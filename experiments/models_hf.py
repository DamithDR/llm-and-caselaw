# -*- coding: utf-8 -*-
"""Open-weight model registry, smallest to ~80B parameters.

`params_b` is the total parameter count in billions (for mixture-of-experts models
that is the total, not the active count - `active_b` carries that separately).
`vram_gb_bf16` is a rough working estimate for loading in bfloat16, to decide what
can run locally versus what needs a hosted endpoint.
"""

HF_MODELS = [
    # ---- tiny: will run on CPU, useful as a floor for the YES/NO signal
    dict(id="Qwen/Qwen2.5-0.5B-Instruct",        params_b=0.5,  family="Qwen2.5"),
    dict(id="HuggingFaceTB/SmolLM2-1.7B-Instruct", params_b=1.7, family="SmolLM2"),
    dict(id="meta-llama/Llama-3.2-1B-Instruct",  params_b=1.2,  family="Llama-3.2", gated=True),
    dict(id="Qwen/Qwen2.5-1.5B-Instruct",        params_b=1.5,  family="Qwen2.5"),

    # ---- small: single consumer GPU
    dict(id="meta-llama/Llama-3.2-3B-Instruct",  params_b=3.2,  family="Llama-3.2", gated=True),
    dict(id="Qwen/Qwen2.5-3B-Instruct",          params_b=3.1,  family="Qwen2.5"),
    dict(id="microsoft/Phi-3.5-mini-instruct",   params_b=3.8,  family="Phi-3.5"),
    dict(id="google/gemma-2-2b-it",              params_b=2.6,  family="Gemma-2", gated=True),

    # ---- mid: ~16-24 GB
    dict(id="mistralai/Mistral-7B-Instruct-v0.3", params_b=7.2, family="Mistral"),
    dict(id="Qwen/Qwen2.5-7B-Instruct",          params_b=7.6,  family="Qwen2.5"),
    dict(id="meta-llama/Llama-3.1-8B-Instruct",  params_b=8.0,  family="Llama-3.1", gated=True),
    dict(id="google/gemma-2-9b-it",              params_b=9.2,  family="Gemma-2", gated=True),
    dict(id="mistralai/Mistral-Nemo-Instruct-2407", params_b=12.2, family="Mistral"),
    dict(id="Qwen/Qwen2.5-14B-Instruct",         params_b=14.8, family="Qwen2.5"),
    dict(id="microsoft/phi-4",                   params_b=14.7, family="Phi-4"),

    # ---- large: multi-GPU or a hosted endpoint
    dict(id="google/gemma-2-27b-it",             params_b=27.2, family="Gemma-2", gated=True),
    dict(id="Qwen/Qwen2.5-32B-Instruct",         params_b=32.8, family="Qwen2.5"),
    dict(id="mistralai/Mixtral-8x7B-Instruct-v0.1", params_b=46.7, active_b=12.9,
         family="Mixtral", moe=True),
    dict(id="meta-llama/Llama-3.3-70B-Instruct", params_b=70.6, family="Llama-3.3", gated=True),
    dict(id="Qwen/Qwen2.5-72B-Instruct",         params_b=72.7, family="Qwen2.5"),
]

for _m in HF_MODELS:
    _m.setdefault("gated", False)
    _m.setdefault("moe", False)
    _m.setdefault("active_b", _m["params_b"])
    # weights in bf16 (2 bytes/param) plus ~15% for activations, KV cache and overhead
    _m["vram_gb_bf16"] = round(_m["params_b"] * 2 * 1.15, 1)

HF_MODELS.sort(key=lambda m: m["params_b"])

BY_ID = {m["id"]: m for m in HF_MODELS}


def select(max_params=None, min_params=None, families=None, ids=None, exclude_gated=True):
    """Filter the registry - typically by what your hardware can actually hold.

    Licence-gated repos are excluded by default: they need the terms accepted on the
    Hub and a logged-in token, so including them silently turns into a wall of 403s.
    Pass exclude_gated=False once you have accepted them.
    """
    out = []
    for m in HF_MODELS:
        if ids and m["id"] not in ids:
            continue
        if max_params is not None and m["params_b"] > max_params:
            continue
        if min_params is not None and m["params_b"] < min_params:
            continue
        if families and m["family"] not in families:
            continue
        if exclude_gated and m["gated"]:
            continue
        out.append(m)
    return out


if __name__ == "__main__":
    print("%-46s %8s %8s %9s %s" % ("MODEL", "PARAMS_B", "ACTIVE_B", "VRAM_BF16", "NOTES"))
    print("-" * 92)
    for m in HF_MODELS:
        notes = " ".join(
            x for x in ("gated" if m["gated"] else "", "MoE" if m["moe"] else "") if x)
        print("%-46s %8.1f %8.1f %8.1fG  %s"
              % (m["id"], m["params_b"], m["active_b"], m["vram_gb_bf16"], notes))
    print("\n%d models, %.1fB - %.1fB parameters"
          % (len(HF_MODELS), HF_MODELS[0]["params_b"], HF_MODELS[-1]["params_b"]))
