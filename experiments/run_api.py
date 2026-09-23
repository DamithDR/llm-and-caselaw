# -*- coding: utf-8 -*-
"""Run the jurisdiction-advice prompt against OpenAI, Anthropic (Claude) and DeepSeek.

API keys are read from the environment - never pass them on the command line, and
never commit them:

    set OPENAI_API_KEY=sk-...
    set ANTHROPIC_API_KEY=sk-ant-...
    set DEEPSEEK_API_KEY=sk-...

    pip install openai anthropic

Examples
--------
    python experiments/run_api.py --providers openai anthropic deepseek
    python experiments/run_api.py --models gpt-4o-mini --regions Africa --limit 5
    python experiments/run_api.py --list
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.common import (                           # noqa: E402
    ResultStore, build_prompt, load_jurisdictions, parse_reply,
)

MAX_TOKENS = 300

# Edit freely - these are the defaults, not a fixed set.
API_MODELS = [
    dict(id="gpt-4o-mini",            provider="openai",    label="GPT-4o mini"),
    dict(id="gpt-4o",                 provider="openai",    label="GPT-4o"),
    dict(id="claude-3-5-haiku-latest", provider="anthropic", label="Claude 3.5 Haiku"),
    dict(id="claude-sonnet-4-5",      provider="anthropic", label="Claude Sonnet 4.5"),
    dict(id="deepseek-chat",          provider="deepseek",  label="DeepSeek-V3"),
    dict(id="deepseek-reasoner",      provider="deepseek",  label="DeepSeek-R1"),
]

ENV_KEY = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
}


# --------------------------------------------------------------------------- clients
class OpenAICompatible(object):
    """OpenAI, and DeepSeek which speaks the same protocol on a different base_url."""

    def __init__(self, model_id, api_key, base_url=None, json_mode=True):
        from openai import OpenAI
        self.model_id = model_id
        self.json_mode = json_mode
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=120.0,
                             max_retries=3)

    def generate(self, system, user):
        kw = dict(
            model=self.model_id,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            max_tokens=MAX_TOKENS,
            temperature=0.0,
        )
        if self.json_mode:
            kw["response_format"] = {"type": "json_object"}
        try:
            r = self.client.chat.completions.create(**kw)
        except Exception as e:
            # reasoning models reject temperature / response_format; retry plainly
            if self.json_mode or "temperature" in repr(e):
                kw.pop("response_format", None)
                kw.pop("temperature", None)
                r = self.client.chat.completions.create(**kw)
            else:
                raise
        return (r.choices[0].message.content or "").strip()


class AnthropicClient(object):
    def __init__(self, model_id, api_key):
        import anthropic
        self.model_id = model_id
        self.client = anthropic.Anthropic(api_key=api_key, timeout=120.0, max_retries=3)

    def generate(self, system, user):
        r = self.client.messages.create(
            model=self.model_id,
            system=system,
            messages=[{"role": "user", "content": user}],
            max_tokens=MAX_TOKENS,
            temperature=0.0,
        )
        return "".join(b.text for b in r.content if getattr(b, "type", "") == "text").strip()


def make_client(spec):
    key = os.environ.get(ENV_KEY[spec["provider"]])
    if not key:
        raise RuntimeError("%s is not set" % ENV_KEY[spec["provider"]])
    if spec["provider"] == "openai":
        return OpenAICompatible(spec["id"], key)
    if spec["provider"] == "deepseek":
        return OpenAICompatible(spec["id"], key, base_url="https://api.deepseek.com",
                                json_mode=not spec["id"].endswith("reasoner"))
    if spec["provider"] == "anthropic":
        return AnthropicClient(spec["id"], key)
    raise ValueError(spec["provider"])


# --------------------------------------------------------------------------- driver
def run_model(spec, jurisdictions, run_tag="", repeats=1, sleep=0.0):
    store = ResultStore(spec["id"], run_tag)
    todo = [(j, r) for j in jurisdictions for r in range(repeats)
            if not store.has(j["name"], r)]
    if not todo:
        print("== %s: already complete (%d rows)" % (spec["id"], len(store.done)))
        return store
    print("== %s [%s]  %d calls -> %s"
          % (spec["id"], spec["provider"], len(todo), os.path.basename(store.path)))
    try:
        client = make_client(spec)
    except Exception as e:
        print("   !! %r" % (e,))
        return store

    for n, (j, rep) in enumerate(todo, 1):
        p = build_prompt(j["phrase"])
        t0 = time.time()
        raw, err = "", None
        try:
            raw = client.generate(p["system"], p["user"])
        except Exception as e:
            err = repr(e)[:300]
        parsed = parse_reply(raw) if err is None else None
        row = store.add(j["name"], j["region"], parsed, raw, time.time() - t0,
                        repeat=rep, error=err,
                        extra={"provider": spec["provider"]})
        flag = ("YES" if row["can_advise"] is True else
                "NO " if row["can_advise"] is False else
                "ERR" if err else "???")
        print("   [%3d/%3d] %-32s %s  %s"
              % (n, len(todo), j["name"][:32], flag,
                 (row["reason_code"] or row["parse_note"] or err or "")[:44]))
        if sleep:
            time.sleep(sleep)
    return store


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--providers", nargs="*",
                    choices=["openai", "anthropic", "deepseek"],
                    help="run every configured model for these providers")
    ap.add_argument("--models", nargs="*", help="explicit model ids")
    ap.add_argument("--regions", nargs="*")
    ap.add_argument("--names", nargs="*")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--sleep", type=float, default=0.0, help="seconds between calls")
    ap.add_argument("--run-tag", default="")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()

    if a.list:
        print("%-26s %-11s %-22s %s" % ("MODEL", "PROVIDER", "ENV VAR", "KEY SET?"))
        print("-" * 72)
        for m in API_MODELS:
            env = ENV_KEY[m["provider"]]
            print("%-26s %-11s %-22s %s"
                  % (m["id"], m["provider"], env,
                     "yes" if os.environ.get(env) else "NO"))
        return

    specs = API_MODELS
    if a.models:
        want = set(a.models)
        specs = [m for m in specs if m["id"] in want]
    if a.providers:
        specs = [m for m in specs if m["provider"] in set(a.providers)]
    if not specs:
        print("no models matched; try --list")
        return

    missing = {m["provider"] for m in specs if not os.environ.get(ENV_KEY[m["provider"]])}
    if missing:
        print("warning: no API key for %s - those models will be skipped\n"
              % ", ".join(sorted(missing)))

    jur = load_jurisdictions(regions=a.regions, names=a.names, limit=a.limit)
    print("%d model(s) x %d jurisdiction(s) x %d repeat(s) = %d calls\n"
          % (len(specs), len(jur), a.repeats, len(specs) * len(jur) * a.repeats))
    for spec in specs:
        run_model(spec, jur, run_tag=a.run_tag, repeats=a.repeats, sleep=a.sleep)

    print("\nrun `python experiments/report.py` for the YES/NO statistics")


if __name__ == "__main__":
    main()
