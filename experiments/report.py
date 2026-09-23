# -*- coding: utf-8 -*-
"""YES/NO statistics over everything in results/.

    python experiments/report.py                 # summary table
    python experiments/report.py --by-region
    python experiments/report.py --reasons
    python experiments/report.py --jurisdictions # which countries models refuse
    python experiments/report.py --csv results/summary.csv
"""

import argparse
import collections
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.common import RESULTS, load_results, summarise   # noqa: E402

REGIONS = ["Africa", "Asia", "Europe", "North America", "South America", "Oceania"]


def bar(pct, width=22):
    filled = int(round(pct / 100.0 * width))
    return "#" * filled + "." * (width - filled)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--by-region", action="store_true")
    ap.add_argument("--reasons", action="store_true")
    ap.add_argument("--jurisdictions", action="store_true")
    ap.add_argument("--csv", metavar="PATH")
    a = ap.parse_args()

    if not os.path.isdir(RESULTS) or not os.listdir(RESULTS):
        print("no results yet - run experiments/run_hf.py or experiments/run_api.py")
        return
    rows = load_results()
    stats = summarise(rows)
    if not stats:
        print("results/ has no parseable rows")
        return

    print("=" * 96)
    print("CAN THE MODEL ADVISE IN THIS JURISDICTION?   %d responses across %d model(s)"
          % (len(rows), len(stats)))
    print("=" * 96)
    print("%-42s %5s %5s %5s %6s %6s  %s"
          % ("MODEL", "N", "YES", "NO", "YES%", "BAD", "YES RATE"))
    print("-" * 96)
    for m in sorted(stats.values(), key=lambda m: -m["yes_rate"]):
        print("%-42s %5d %5d %5d %5.1f%% %6d  %s"
              % (m["model"][-42:], m["n"], m["yes"], m["no"], m["yes_rate"],
                 m["unparsed"] + m["errors"], bar(m["yes_rate"])))
    print("-" * 96)
    tot_y = sum(m["yes"] for m in stats.values())
    tot_n = sum(m["no"] for m in stats.values())
    tot_a = tot_y + tot_n
    print("%-42s %5d %5d %5d %5.1f%% %6d"
          % ("ALL MODELS", len(rows), tot_y, tot_n,
             (100.0 * tot_y / tot_a) if tot_a else 0.0,
             sum(m["unparsed"] + m["errors"] for m in stats.values())))
    print("\nYES = model says it can give advice specific to that jurisdiction.")
    print("BAD = unparseable replies plus API/load errors.")

    if a.by_region:
        print("\n" + "=" * 96)
        print("YES RATE BY REGION")
        print("=" * 96)
        header = "%-42s" % "MODEL" + "".join("%14s" % r[:13] for r in REGIONS)
        print(header)
        print("-" * len(header))
        for m in sorted(stats.values(), key=lambda m: -m["yes_rate"]):
            line = "%-42s" % m["model"][-42:]
            for reg in REGIONS:
                d = m["by_region"].get(reg)
                ans = (d["yes"] + d["no"]) if d else 0
                line += "%13s " % (
                    ("%.0f%% (%d)" % (100.0 * d["yes"] / ans, ans)) if ans else "-")
            print(line)

    if a.reasons:
        print("\n" + "=" * 96)
        print("REASON CODES")
        print("=" * 96)
        for m in stats.values():
            print("\n%s" % m["model"])
            total = sum(m["reason_codes"].values()) or 1
            for code, c in m["reason_codes"].most_common():
                print("   %-30s %5d  %5.1f%%" % (code, c, 100.0 * c / total))
            if m["confidence"]:
                print("   confidence: " + ", ".join(
                    "%s=%d" % (k, v) for k, v in m["confidence"].most_common()))

    if a.jurisdictions:
        print("\n" + "=" * 96)
        print("BY JURISDICTION - how many models would advise")
        print("=" * 96)
        per = collections.defaultdict(lambda: {"yes": 0, "no": 0, "region": "?"})
        for r in rows:
            if r.get("error") or r.get("can_advise") is None:
                continue
            d = per[r["jurisdiction"]]
            d["region"] = r.get("region") or "?"
            d["yes" if r["can_advise"] else "no"] += 1
        ranked = sorted(per.items(),
                        key=lambda kv: (kv[1]["yes"] / max(1, kv[1]["yes"] + kv[1]["no"]),
                                        kv[1]["yes"]))
        print("\n-- lowest YES rate (models decline to advise) --")
        for name, d in ranked[:25]:
            n = d["yes"] + d["no"]
            print("   %-34s %-14s %2d/%2d models  %5.1f%%"
                  % (name[:34], d["region"], d["yes"], n, 100.0 * d["yes"] / n))
        print("\n-- highest YES rate --")
        for name, d in ranked[-15:][::-1]:
            n = d["yes"] + d["no"]
            print("   %-34s %-14s %2d/%2d models  %5.1f%%"
                  % (name[:34], d["region"], d["yes"], n, 100.0 * d["yes"] / n))

    if a.csv:
        import csv
        with open(a.csv, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["model", "n", "answered", "yes", "no", "yes_rate_pct",
                        "unparsed", "errors"] + [r + "_yes_pct" for r in REGIONS])
            for m in sorted(stats.values(), key=lambda m: -m["yes_rate"]):
                row = [m["model"], m["n"], m["answered"], m["yes"], m["no"],
                       round(m["yes_rate"], 1), m["unparsed"], m["errors"]]
                for reg in REGIONS:
                    d = m["by_region"].get(reg)
                    ans = (d["yes"] + d["no"]) if d else 0
                    row.append(round(100.0 * d["yes"] / ans, 1) if ans else "")
                w.writerow(row)
        print("\nwrote %s" % a.csv)


if __name__ == "__main__":
    main()
