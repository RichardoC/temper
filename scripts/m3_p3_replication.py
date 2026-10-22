#!/usr/bin/env python3
"""Flip-set replication: a second game on every flip (design §4.2, §5 P3 D).

The SHIP gate demands ">= 2 confirmed-attributed flips", and the ATTRIBUTION
gate demands "flip-set replication green". A flip is a single Bernoulli draw on
a deterministic cell, so one game cannot confirm it — replication means running
the SAME cell config on the flip set again with a freshly derived seed and
checking the flip survives.

This script does the analysis side of that: given the primary cell's results
and a replication run's results (both emission-rule cells, one row per
problem), plus the set of problems that flipped, it reports per flip whether
the flip REPRODUCED. A flip is ``confirmed`` when the replication run on that
problem is a pass; it is ``confirmed_attributed`` when it ALSO lies outside the
ladder-reachable / type-quartet cohorts (i.e. it is evidence for the in-loop
mechanism rather than for diversity or knowledge delivery).

    temper/.venv/bin/python scripts/m3_p3_replication.py \\
        --primary /tmp/m3_p3/c-full-emission-results.json \\
        --replication /tmp/m3_p3/c-full-repl-emission-results.json \\
        --flips HumanEval_19_sort_numbers,HumanEval_21_rescale_to_unit \\
        --json /tmp/m3_p3/replication.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.m3_p3_analysis import cohort_tags  # noqa: E402
from temper.multipl_e.bench import load_problems  # noqa: E402


def load_passers(path: str) -> dict:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    out = {}
    for row in d["per_problem"]:
        st = row.get("statuses_official") or []
        out[row["name"]] = {
            "pass": any(s == "OK" for s in st),
            "statuses": st,
            "emitted": row.get("emitted"),
            "emitted_temp": row.get("emitted_temp"),
        }
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--primary", required=True,
                    help="the primary cell's emission results JSON")
    ap.add_argument("--replication", required=True,
                    help="the replication run's emission results JSON")
    ap.add_argument("--flips", required=True,
                    help="comma-separated problem names (the flip set)")
    ap.add_argument("--require-outside-cohort", action="store_true", default=True,
                    help="count only flips outside the ladder-reachable / "
                         "type-quartet cohorts as ATTRIBUTED (default: on)")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    primary = load_passers(args.primary)
    repl = load_passers(args.replication)
    flips = [f.strip() for f in args.flips.split(",") if f.strip()]
    known = {td["name"] for td in load_problems("humaneval-rs")}
    unknown = [f for f in flips if f not in known]
    if unknown:
        raise SystemExit(f"unknown problem names in --flips: {unknown}")

    print(f"flip-set replication: {len(flips)} flips")
    print(f"  primary    : {args.primary}")
    print(f"  replication: {args.replication}")
    rows, n_green = [], 0
    for name in flips:
        p0 = primary.get(name, {})
        p1 = repl.get(name, {})
        reproduced = bool(p1.get("pass"))
        tags = cohort_tags(name)
        outside = ("type_quartet" not in tags and "p1_reachable" not in tags)
        confirmed = reproduced
        attributed = reproduced and outside
        n_green += 1 if reproduced else 0
        rows.append({
            "problem": name, "cohorts": tags,
            "primary_pass": bool(p0.get("pass")),
            "replication_pass": reproduced,
            "replication_statuses": p1.get("statuses"),
            "replication_emitted": p1.get("emitted"),
            "replication_temp": p1.get("emitted_temp"),
            "confirmed": confirmed,
            "outside_ladder_reachable": outside,
            "confirmed_attributed": attributed,
        })
        print(f"  {name:44s} primary={'PASS' if p0.get('pass') else 'fail'} "
              f"repl={'PASS' if reproduced else 'fail'} "
              f"{'GREEN' if confirmed else 'NOT-REPRODUCED'}"
              f"{' (attributed)' if attributed else ''}")

    all_green = n_green == len(flips) and len(flips) > 0
    print(f"\nreplicated {n_green}/{len(flips)} flips -> "
          f"{'GREEN' if all_green else 'NOT GREEN'}")
    payload = {"primary": args.primary, "replication": args.replication,
               "problems": rows, "n_flips": len(flips),
               "n_replicated": n_green, "all_green": all_green,
               "n_confirmed_attributed": sum(
                   1 for r in rows if r["confirmed_attributed"])}
    if args.json:
        Path(args.json).write_text(json.dumps(payload, indent=1), encoding="utf-8")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
