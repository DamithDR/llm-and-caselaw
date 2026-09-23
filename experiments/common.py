# -*- coding: utf-8 -*-
"""Shared plumbing for the jurisdiction-advice experiments.

Response parsing, incremental result storage with resume, and the YES/NO statistics.
Both runners (open models via Hugging Face, and the commercial APIs) write the same
row shape, so results/ can be aggregated in one pass.
"""

import json
import os
import re
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
sys.path.insert(0, ROOT)

from prompts.jurisdiction_advice_prompt import (          # noqa: E402
    build_prompt, REASON_CODES, SCENARIO_ID,
)

VALID_CODES = set(REASON_CODES)
CONFIDENCES = ("HIGH", "MEDIUM", "LOW")


# --------------------------------------------------------------------------- inputs
def load_jurisdictions(regions=None, names=None, limit=None):
    """The jurisdiction list the prompts are rendered for."""
    path = os.path.join(ROOT, "prompts", "jurisdictions.json")
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)
    if regions:
        want = {r.casefold() for r in regions}
        rows = [r for r in rows if r["region"].casefold() in want]
    if names:
        want = {n.casefold() for n in names}
        rows = [r for r in rows if r["name"].casefold() in want]
    return rows[:limit] if limit else rows


def slug(text):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(text)).strip("_")


# --------------------------------------------------------------------------- parsing
_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.S | re.I)


def _candidate_blobs(text):
    """JSON-ish substrings, most likely first: fenced block, then balanced braces."""
    text = (text or "").strip()
    for m in _FENCE.finditer(text):
        yield m.group(1)
    depth, start = 0, None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth:
                depth -= 1
                if depth == 0 and start is not None:
                    yield text[start:i + 1]
    yield text


def parse_reply(text):
    """Parse a model reply into the four fields.

    Returns a dict with can_advise (True/False/None), reason_code, reason,
    confidence and parse_ok. Never raises: an unparseable reply is a result,
    not an error, and must be counted rather than dropped.
    """
    out = {
        "can_advise": None, "reason_code": None, "reason": None,
        "confidence": None, "parse_ok": False, "parse_note": None,
    }
    for blob in _candidate_blobs(text):
        try:
            obj = json.loads(blob)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        low = {str(k).strip().lower(): v for k, v in obj.items()}
        raw = low.get("can_advise")
        if raw is None:
            continue
        if isinstance(raw, bool):
            out["can_advise"] = raw
        else:
            s = str(raw).strip().upper()
            if s.startswith("Y") or s == "TRUE":
                out["can_advise"] = True
            elif s.startswith("N") or s == "FALSE":
                out["can_advise"] = False
            else:
                continue
        code = str(low.get("reason_code") or "").strip().upper()
        out["reason_code"] = code or None
        if code and code not in VALID_CODES:
            out["parse_note"] = "reason_code not in the defined set"
        conf = str(low.get("confidence") or "").strip().upper()
        out["confidence"] = conf or None
        if conf and conf not in CONFIDENCES:
            out["parse_note"] = (out["parse_note"] or "") + "; confidence off-scale"
        reason = low.get("reason")
        out["reason"] = str(reason).strip() if reason is not None else None
        out["parse_ok"] = True
        return out

    # No JSON at all. Fall back to a bare YES/NO so refusals still count.
    head = (text or "").strip().upper()[:400]
    if re.search(r"\bCAN[_ ]?ADVISE\b\W{0,4}(YES|TRUE)\b", head) or head.startswith("YES"):
        out["can_advise"] = True
        out["parse_note"] = "no JSON; recovered YES from free text"
    elif re.search(r"\bCAN[_ ]?ADVISE\b\W{0,4}(NO|FALSE)\b", head) or head.startswith("NO"):
        out["can_advise"] = False
        out["parse_note"] = "no JSON; recovered NO from free text"
    else:
        out["parse_note"] = "no JSON object and no YES/NO found"
    return out


# --------------------------------------------------------------------------- storage
class ResultStore(object):
    """Append-only JSONL per model, with resume.

    A long sweep will be interrupted - rate limits, OOM, a dropped connection.
    Re-running skips whatever already succeeded instead of paying for it twice.
    """

    def __init__(self, model_id, run_tag=""):
        os.makedirs(RESULTS, exist_ok=True)
        name = slug(model_id) + (("__" + slug(run_tag)) if run_tag else "")
        self.path = os.path.join(RESULTS, name + ".jsonl")
        self.model_id = model_id
        self.done = set()
        if os.path.exists(self.path):
            for line in open(self.path, encoding="utf-8"):
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if r.get("error") is None:
                    self.done.add((r.get("jurisdiction"), r.get("repeat", 0)))

    def has(self, jurisdiction, repeat=0):
        return (jurisdiction, repeat) in self.done

    def add(self, jurisdiction, region, parsed, raw, elapsed,
            repeat=0, error=None, extra=None):
        row = {
            "model": self.model_id,
            "scenario_id": SCENARIO_ID,
            "jurisdiction": jurisdiction,
            "region": region,
            "repeat": repeat,
            "can_advise": parsed.get("can_advise") if parsed else None,
            "reason_code": parsed.get("reason_code") if parsed else None,
            "reason": parsed.get("reason") if parsed else None,
            "confidence": parsed.get("confidence") if parsed else None,
            "parse_ok": bool(parsed.get("parse_ok")) if parsed else False,
            "parse_note": parsed.get("parse_note") if parsed else None,
            "elapsed_s": round(elapsed, 2),
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "error": error,
            "raw": (raw or "")[:4000],
        }
        if extra:
            row.update(extra)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        if error is None:
            self.done.add((jurisdiction, repeat))
        return row


def load_results(path=None):
    """Every result row written so far."""
    rows = []
    files = [path] if path else [
        os.path.join(RESULTS, f) for f in sorted(os.listdir(RESULTS))
        if f.endswith(".jsonl")
    ] if os.path.isdir(RESULTS) else []
    for p in files:
        for line in open(p, encoding="utf-8"):
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


# --------------------------------------------------------------------------- stats
def summarise(rows):
    """YES/NO counts per model, plus reason codes, confidence and failures."""
    import collections
    by_model = collections.OrderedDict()
    for r in rows:
        m = by_model.setdefault(r.get("model", "?"), {
            "model": r.get("model", "?"), "n": 0, "yes": 0, "no": 0,
            "unparsed": 0, "errors": 0,
            "reason_codes": collections.Counter(),
            "confidence": collections.Counter(),
            "by_region": collections.defaultdict(lambda: {"yes": 0, "no": 0, "n": 0}),
            "yes_list": [], "no_list": [],
        })
        m["n"] += 1
        if r.get("error"):
            m["errors"] += 1
            continue
        ca = r.get("can_advise")
        reg = m["by_region"][r.get("region") or "?"]
        reg["n"] += 1
        if ca is True:
            m["yes"] += 1
            reg["yes"] += 1
            m["yes_list"].append(r.get("jurisdiction"))
        elif ca is False:
            m["no"] += 1
            reg["no"] += 1
            m["no_list"].append(r.get("jurisdiction"))
        else:
            m["unparsed"] += 1
        if r.get("reason_code"):
            m["reason_codes"][r["reason_code"]] += 1
        if r.get("confidence"):
            m["confidence"][r["confidence"]] += 1
    for m in by_model.values():
        answered = m["yes"] + m["no"]
        m["answered"] = answered
        m["yes_rate"] = (100.0 * m["yes"] / answered) if answered else 0.0
    return by_model
