#!/usr/bin/env python3
"""M3.1 same-draws rule delta: P-prop-engine vs its own P-prop-screen TWIN.

Design §3's secondary comparison and the P3 canvas-pair-diff discipline applied
to M3.1. The twin reranks the ENGINE's OWN banked ensemble with the offline
property rule; the engine cell emits with the same rule in-loop. Because both
sides see the same draws, every divergence is either an R3 measurement
difference (same canvas, different property evidence) or a rule-wiring
difference (a bug) — never draw luck.

Three classes, and they mean different things:

  SAME_EVIDENCE  the same canvas with the same property evidence and the same
                 container status — the engine and the offline rule agree.
  SAME_OUTCOME   a different canvas, same container status. Expected: R3's
                 evidence is measured in-loop at production cost, and the twin
                 re-measures it offline, so a timeout/abstention can differ.
  DIVERGED       a different container STATUS. THE forensics cell: either the
                 engine's R3 saw something the offline pass did not (abstention
                 asymmetry, a harness crash in one path), or the emission rule
                 fired differently (wiring bug — must be explained).

A problem whose emitted canvas the twin cannot locate is MISSING data, not a
divergence (§P3's lesson: manufacturing forensics out of an absent log is worse
than reporting the absence).

    temper/.venv/bin/python scripts/m3_31_twin_diff.py \\
        --engine-detail bench/multipl-e/m3-31/p-prop-engine-detail.json \\
        --twin-detail bench/multipl-e/m3-31/p-prop-engine-twin-detail.json \\
        --engine-results bench/multipl-e/m3-31/p-prop-engine-results.json \\
        --twin-results bench/multipl-e/m3-31/p-prop-engine-twin-property-results.json \\
        --json bench/multipl-e/m3-31/p-prop-engine-twin-diff.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _by_problem(detail: dict) -> Dict[str, dict]:
    return {r["problem"]: r for r in detail.get("attribution") or []}


def _status_map(results_path: str) -> Dict[str, dict]:
    d = json.loads(Path(results_path).read_text(encoding="utf-8"))
    out = {}
    for row in d["per_problem"]:
        st = row.get("statuses_official") or []
        out[row["name"]] = {"status": (st[0] if st else None),
                            "excluded": bool(row.get("n_excluded")),
                            "emitted": row.get("emitted")}
    return out


def classify(engine_row: dict, twin_row: dict,
             engine_status: dict, twin_status: dict) -> dict:
    e_pick = engine_row.get("prop_pick")
    t_pick = twin_row.get("prop_pick")
    e_out = ("EXCLUDED" if engine_status.get("excluded")
             else engine_status.get("status"))
    t_out = ("EXCLUDED" if twin_status.get("excluded")
             else twin_status.get("status"))
    e_rates = dict(zip([str(i) for i in range(len(engine_row["canvas_pass_rates"]))],
                       engine_row["canvas_pass_rates"]))
    t_rates = dict(zip([str(i) for i in range(len(twin_row["canvas_pass_rates"]))],
                       twin_row["canvas_pass_rates"]))
    if e_pick is None or t_pick is None:
        verdict = "MISSING"
    elif e_pick == t_pick and e_rates.get(str(e_pick)) == t_rates.get(str(t_pick)):
        verdict = "SAME_EVIDENCE"
    elif e_out == t_out:
        verdict = "SAME_OUTCOME"
    else:
        verdict = "DIVERGED"
    return {
        "problem": engine_row["problem"],
        "engine_pick": e_pick, "engine_status": e_out,
        "engine_pass_rate": e_rates.get(str(e_pick)),
        "twin_pick": t_pick, "twin_status": t_out,
        "twin_pass_rate": t_rates.get(str(t_pick)),
        "same_pick": e_pick == t_pick,
        "same_property_evidence": (e_rates == t_rates),
        "verdict": verdict,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine-detail", required=True,
                    help="the ENGINE's own oracle/detail JSON (banked canvases)")
    ap.add_argument("--twin-detail", required=True,
                    help="the offline twin's detail JSON over the same canvases")
    ap.add_argument("--engine-results", required=True)
    ap.add_argument("--twin-results", required=True)
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    engine = _by_problem(json.loads(Path(args.engine_detail).read_text()))
    twin = _by_problem(json.loads(Path(args.twin_detail).read_text()))
    e_status = _status_map(args.engine_results)
    t_status = _status_map(args.twin_results)

    rows = []
    counts: Dict[str, int] = {}
    for name in sorted(set(engine) | set(twin)):
        if name not in engine or name not in twin:
            rows.append({"problem": name, "verdict": "MISSING"})
            counts["MISSING"] = counts.get("MISSING", 0) + 1
            continue
        r = classify(engine[name], twin[name], e_status.get(name, {}),
                     t_status.get(name, {}))
        rows.append(r)
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1

    diverged = [r for r in rows if r["verdict"] == "DIVERGED"]
    report = {
        "engine_detail": args.engine_detail, "twin_detail": args.twin_detail,
        "counts": counts, "rows": rows,
        "n_problems": len(rows),
        "diverged": [r["problem"] for r in diverged],
        "note": ("SAME_EVIDENCE = same canvas and same evidence; SAME_OUTCOME = "
                 "different canvas, same status (expected when R3's in-loop and "
                 "offline measurement differ by an abstention); DIVERGED = "
                 "different status and must be explained as either R3 "
                 "measurement asymmetry or an emission-wiring bug, never "
                 "reported as mechanism evidence."),
    }
    print("=== engine vs twin (same draws, rule delta) ===")
    for k in ("SAME_EVIDENCE", "SAME_OUTCOME", "DIVERGED", "MISSING"):
        print(f"  {k:14s} {counts.get(k, 0)}")
    for r in diverged:
        print(f"\n  DIVERGED {r['problem']}: engine p{r['engine_pick']} "
              f"({r['engine_status']}, rate {r['engine_pass_rate']}) vs twin "
              f"p{r['twin_pick']} ({r['twin_status']}, rate {r['twin_pass_rate']})")
        if r["same_property_evidence"]:
            print("    same per-canvas property evidence -> RULE WIRING BUG, "
                  "escalate")
        else:
            print("    different property evidence -> R3 measurement asymmetry "
                  "(in-loop vs offline abstention)")
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(report, indent=1), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
