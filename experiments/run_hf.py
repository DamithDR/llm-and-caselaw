# -*- coding: utf-8 -*-
"""Run the jurisdiction-advice prompt against open-weight models on Hugging Face.

Two backends:

  --backend local     load the weights with transformers, one prompt at a time
  --backend vllm      batched GPU inference - the right choice on a cluster
  --backend api       call Hugging Face Inference Providers (no local weights)

`local` is right for the small end of the registry. Anything past ~14B generally needs
`api` unless you have the VRAM for it - see `python experiments/models_hf.py` for the
per-model estimate.

Examples
--------
    # everything that fits in 8 GB, all 195 jurisdictions
    # (licence-gated repos are skipped unless you pass --include-gated)
    python experiments/run_hf.py --backend local --max-params 4

    # one model, Africa only, to sanity-check the pipeline
    python experiments/run_hf.py --backend local --ids Qwen/Qwen2.5-0.5B-Instruct \
        --regions Africa --limit 5

    # the big end through hosted inference
    set HF_TOKEN=hf_...
    python experiments/run_hf.py --backend api --min-params 20
"""

import argparse
import gc
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.common import (                           # noqa: E402
    ResultStore, build_prompt, load_jurisdictions, parse_reply,
)
from experiments import models_hf                          # noqa: E402

MAX_NEW_TOKENS = 220          # the reply is a four-field JSON object


# --------------------------------------------------------------------------- local
class LocalBackend(object):
    """transformers on whatever device is available."""

    def __init__(self, model_id, dtype="auto", device_map="auto", load_in_4bit=False):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id = model_id
        self.torch = torch
        print("  loading %s ..." % model_id)
        self.tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        kw = dict(trust_remote_code=True, device_map=device_map)
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )
        else:
            kw["torch_dtype"] = (
                torch.bfloat16 if dtype == "auto" and torch.cuda.is_available() else
                getattr(torch, dtype, None) if dtype != "auto" else torch.float32)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kw)
        self.model.eval()
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token

    def _render(self, system, user):
        """Apply the chat template, folding system into user where unsupported."""
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        try:
            return self.tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True)
        except Exception:
            merged = [{"role": "user", "content": system + "\n\n" + user}]
            return self.tok.apply_chat_template(
                merged, tokenize=False, add_generation_prompt=True)

    def generate(self, system, user):
        text = self._render(system, user)
        enc = self.tok(text, return_tensors="pt", add_special_tokens=False)
        enc = {k: v.to(self.model.device) for k, v in enc.items()}
        with self.torch.inference_mode():
            out = self.model.generate(
                **enc,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,                 # greedy: we want the model's prior
                temperature=None, top_p=None, top_k=None,
                pad_token_id=self.tok.pad_token_id,
            )
        return self.tok.decode(out[0][enc["input_ids"].shape[1]:],
                               skip_special_tokens=True).strip()

    def close(self):
        del self.model
        gc.collect()
        try:
            self.torch.cuda.empty_cache()
        except Exception:
            pass


# --------------------------------------------------------------------------- vllm
class VLLMBackend(object):
    """vLLM - batches the whole jurisdiction sweep in one pass.

    This is the right backend on a GPU cluster: 195 prompts go through together
    instead of one forward pass at a time, so wall time is dominated by loading
    the weights rather than by generation.
    """

    batched = True

    def __init__(self, model_id, tensor_parallel_size=1, gpu_memory_utilization=0.90,
                 max_model_len=4096, dtype="bfloat16"):
        from vllm import LLM, SamplingParams

        self.model_id = model_id
        print("  loading %s into vLLM (TP=%d) ..." % (model_id, tensor_parallel_size))
        self.llm = LLM(
            model=model_id,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            dtype=dtype,
            trust_remote_code=True,
        )
        self.tok = self.llm.get_tokenizer()
        self.params = SamplingParams(temperature=0.0, max_tokens=MAX_NEW_TOKENS)

    def _render(self, system, user):
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        try:
            return self.tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True)
        except Exception:
            merged = [{"role": "user", "content": system + "\n\n" + user}]
            return self.tok.apply_chat_template(
                merged, tokenize=False, add_generation_prompt=True)

    def generate_batch(self, pairs):
        """pairs: [(system, user), ...] -> [text, ...] in the same order."""
        prompts = [self._render(s, u) for s, u in pairs]
        outs = self.llm.generate(prompts, self.params)
        return [(o.outputs[0].text or "").strip() for o in outs]

    def generate(self, system, user):
        return self.generate_batch([(system, user)])[0]

    def close(self):
        del self.llm
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass


# --------------------------------------------------------------------------- hosted
class InferenceAPIBackend(object):
    """Hugging Face Inference Providers - no local weights."""

    def __init__(self, model_id, token=None, provider="auto"):
        from huggingface_hub import InferenceClient
        token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        if not token:
            raise RuntimeError("set HF_TOKEN for --backend api")
        self.model_id = model_id
        self.client = InferenceClient(model=model_id, token=token, provider=provider,
                                      timeout=120)

    def generate(self, system, user):
        r = self.client.chat_completion(
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            max_tokens=MAX_NEW_TOKENS,
            temperature=0.0,
        )
        return (r.choices[0].message.content or "").strip()

    def close(self):
        pass


# --------------------------------------------------------------------------- driver
def run_model(spec, jurisdictions, backend_name, run_tag="", repeats=1, **kw):
    model_id = spec["id"]
    store = ResultStore(model_id, run_tag)
    todo = [(j, r) for j in jurisdictions for r in range(repeats)
            if not store.has(j["name"], r)]
    if not todo:
        print("== %s: already complete (%d rows)" % (model_id, len(store.done)))
        return store
    print("== %s  (%.1fB)  %d calls to make -> %s"
          % (model_id, spec["params_b"], len(todo), os.path.basename(store.path)))

    backends = {"local": LocalBackend, "vllm": VLLMBackend, "api": InferenceAPIBackend}
    try:
        backend = backends[backend_name](model_id, **kw)
    except Exception as e:
        print("   !! could not load: %r" % (e,))
        store.add("*", "*", None, "", 0.0, error="load failed: %r" % (e,))
        return store

    try:
        if getattr(backend, "batched", False):
            prompts = [build_prompt(j["phrase"]) for j, _ in todo]
            t0 = time.time()
            try:
                texts = backend.generate_batch([(p["system"], p["user"]) for p in prompts])
                err = None
            except Exception as e:
                texts, err = [""] * len(todo), repr(e)[:300]
            per = (time.time() - t0) / max(1, len(todo))
            yes = no = bad = 0
            for (j, rep), raw in zip(todo, texts):
                parsed = parse_reply(raw) if err is None else None
                row = store.add(j["name"], j["region"], parsed, raw, per,
                                repeat=rep, error=err,
                                extra={"params_b": spec["params_b"]})
                if row["can_advise"] is True:
                    yes += 1
                elif row["can_advise"] is False:
                    no += 1
                else:
                    bad += 1
            print("   batched %d prompts in %.1fs -> YES=%d NO=%d unusable=%d"
                  % (len(todo), per * len(todo), yes, no, bad))
            return store

        for n, (j, rep) in enumerate(todo, 1):
            p = build_prompt(j["phrase"])
            t0 = time.time()
            raw, err = "", None
            try:
                raw = backend.generate(p["system"], p["user"])
            except Exception as e:
                err = repr(e)[:300]
            parsed = parse_reply(raw) if err is None else None
            row = store.add(j["name"], j["region"], parsed, raw, time.time() - t0,
                            repeat=rep, error=err,
                            extra={"params_b": spec["params_b"]})
            flag = ("YES" if row["can_advise"] is True else
                    "NO " if row["can_advise"] is False else
                    "ERR" if err else "???")
            print("   [%3d/%3d] %-32s %s  %s"
                  % (n, len(todo), j["name"][:32], flag,
                     (row["reason_code"] or row["parse_note"] or "")[:44]))
    finally:
        backend.close()
    return store


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--backend", choices=["local", "vllm", "api"], default="local",
                    help="local=transformers, vllm=batched GPU inference, api=hosted")
    ap.add_argument("--tensor-parallel-size", type=int, default=1,
                    help="vllm only: GPUs to shard the model across")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90,
                    help="vllm only")
    ap.add_argument("--max-model-len", type=int, default=4096, help="vllm only")
    ap.add_argument("--ids", nargs="*", help="explicit model ids from the registry")
    ap.add_argument("--max-params", type=float, help="only models at or below this size (B)")
    ap.add_argument("--min-params", type=float, help="only models at or above this size (B)")
    ap.add_argument("--families", nargs="*")
    ap.add_argument("--include-gated", action="store_true",
                    help="also run licence-gated repos (Llama, Gemma); these need the "
                         "terms accepted on the Hub and `huggingface-cli login`")
    ap.add_argument("--regions", nargs="*", help="e.g. Africa Asia")
    ap.add_argument("--names", nargs="*", help="specific jurisdictions")
    ap.add_argument("--limit", type=int, help="first N jurisdictions (smoke test)")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--run-tag", default="")
    ap.add_argument("--load-in-4bit", action="store_true",
                    help="local only: 4-bit quantisation, ~4x less VRAM")
    ap.add_argument("--dry-run", action="store_true", help="list what would run")
    a = ap.parse_args()

    specs = models_hf.select(max_params=a.max_params, min_params=a.min_params,
                             families=a.families, ids=a.ids,
                             exclude_gated=not a.include_gated)
    if not a.include_gated:
        hidden = len(models_hf.select(max_params=a.max_params, min_params=a.min_params,
                                      families=a.families, ids=a.ids,
                                      exclude_gated=False)) - len(specs)
        if hidden:
            print("skipping %d licence-gated model(s); --include-gated to run them"
                  % hidden)
    jur = load_jurisdictions(regions=a.regions, names=a.names, limit=a.limit)
    if not specs:
        print("no models matched the filters")
        return
    print("%d model(s) x %d jurisdiction(s) x %d repeat(s) = %d calls\n"
          % (len(specs), len(jur), a.repeats, len(specs) * len(jur) * a.repeats))
    if a.dry_run:
        for m in specs:
            print("  %-46s %5.1fB  ~%.0fGB bf16" % (m["id"], m["params_b"], m["vram_gb_bf16"]))
        return

    kw = {}
    if a.backend == "local" and a.load_in_4bit:
        kw["load_in_4bit"] = True
    if a.backend == "vllm":
        kw.update(tensor_parallel_size=a.tensor_parallel_size,
                  gpu_memory_utilization=a.gpu_memory_utilization,
                  max_model_len=a.max_model_len)
    for spec in specs:
        run_model(spec, jur, a.backend, run_tag=a.run_tag, repeats=a.repeats, **kw)

    print("\nrun `python experiments/report.py` for the YES/NO statistics")


if __name__ == "__main__":
    main()
