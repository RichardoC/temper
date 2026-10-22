#!/usr/bin/env python3
"""P1 probe reachability curve + the rev1 stratified GO/PARTIAL/NO-GO rule.

Reads a P1 run record (9 problems x 3 temps x K=4, seed-paired across temps) and
its container results, and reports:

  * the per-(problem, temp) reachability P(>=1 passing canvas of K), which is
    the curve the design's §5 P1 gate is read from;
  * per-problem MAX-PASS-TEMP (the highest temperature at which the problem
    produced a passing canvas — the "reachable but only when hot" signature);
  * the container-status histogram per temp;
  * the cohort-stratified verdict: GO = >=4/9 overall AND >=2/4 of the type
    quartet (0/4/20/21) AND >=1 non-type; PARTIAL = 2-3 overall (forces the
    §6.1 EV rewrite); NO-GO = <2 (cite the s512-recheck clause, do not run it).

    temper/.venv/bin/python scripts/m3_p1_reachability.py \
        --run bench/multipl-e/runs/<ts>-p1-probe.jsonl \
        --results bench/multipl-e/results/<ts>.json \
        --export bench/multipl-e/exports/<ts>-p1-probe-norepair
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REPO_ROOT = Path(__file__).resolve().parents[1]

# The design's §5 P1 cohorts, by HumanEval NUMBER.
TYPE_QUARTET = (0, 4, 20, 21)
NON_TYPE = (11, 18, 24, 17, 19)


def _num(problem_name: str) -> int:
    import re
    m = re.match(r"^HumanEval_(\d+)_", problem_name)
    if not m:
        raise ValueError(f"cannot read a HumanEval number from {problem_name!r}")
    return int(m.group(1))


def load_run(path: Path) -> dict:
    meta, rows, done = None, [], None
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        o = json.loads(line)
        if o.get("type") == "run":
            meta = o
        elif o.get("type") == "candidate":
            rows.append(o)
        elif o.get("type") == "done":
            done = o
    if meta is None:
        raise SystemExit(f"{path}: no run meta line")
    return {"meta": meta, "rows": rows, "done": done}


def load_results(path: Path) -> dict:
    d = json.loads(path.read_text(encoding="utf-8"))
    return {row["name"]: row for row in d["per_problem"]}


def build_grid(run: dict, results: dict, manifest: dict) -> dict:
    """{(problem, temp): {pass, statuses, n_ok, n_canvases, seeds}}.

    The join is authoritative on the EXPORT MANIFEST, not on the run record's
    own extract_ok flags: the export re-extracts, so a row the driver accepted
    can still be excluded at export (measured here: 18 driver-failures vs 20
    export-exclusions). A positional zip against a naive filter would shift
    every later status within a problem by the difference — the excluded-rows
    hole has to be reconstructed from the manifest's `source_rows`, which names
    exactly which candidate row produced each completion, in file order.
    """
    by_problem = defaultdict(list)
    for r in run["rows"]:
        by_problem[r["problem"]].append(r)
    grid = {}
    for problem, rows in by_problem.items():
        rows = sorted(rows, key=lambda r: r["i"])
        by_i = {r["i"]: r for r in rows}
        row = results.get(problem)
        statuses = list(row["statuses_official"]) if row else []
        # manifest -> which candidate index produced completion j (file order)
        exported_indices = []
        for prob in manifest.get("problems", []):
            if prob["name"] == problem:
                exported_indices = [sr["i"] for sr in prob["source_rows"]]
                break
        status_of_row = {}
        for jdx, i_idx in enumerate(exported_indices):
            status_of_row[i_idx] = (statuses[jdx] if jdx < len(statuses)
                                    else None)
        for r in rows:
            st = status_of_row.get(r["i"], "EXCLUDED")
            key = (problem, float(r["temperature"]))
            cell = grid.setdefault(key, {"statuses": [], "n_canvases": 0,
                                         "seeds": []})
            cell["statuses"].append(st)
            cell["n_canvases"] += 1
            cell["seeds"].append(r["seed"])
    for key, cell in grid.items():
        cell["n_ok"] = sum(1 for s in cell["statuses"] if s == "OK")
        cell["pass"] = cell["n_ok"] >= 1
    return grid


def classify(problem: str, grid: dict, temps: list) -> dict:
    per_temp = {}
    max_pass_temp = None
    for t in temps:
        cell = grid.get((problem, float(t)))
        ok = bool(cell and cell["pass"])
        per_temp[t] = {"pass": ok,
                       "n_ok": cell["n_ok"] if cell else 0,
                       "n_canvases": cell["n_canvases"] if cell else 0}
        if ok:
            max_pass_temp = t
    return {
        "problem": problem,
        "number": _num(problem),
        "cohort": "type_quartet" if _num(problem) in TYPE_QUARTET else "non_type",
        "per_temp": per_temp,
        "reachable": any(v["pass"] for v in per_temp.values()),
        "max_pass_temp": max_pass_temp,
        # the curve shape: does reachability RISE with temperature?
        "reachability_curve": [1 if per_temp[t]["pass"] else 0 for t in temps],
    }


def verdict(rows: list) -> dict:
    reach = [r for r in rows if r["reachable"]]
    n_total = len(rows)
    quartet = [r for r in rows if r["cohort"] == "type_quartet" and r["reachable"]]
    non_type = [r for r in rows if r["cohort"] == "non_type" and r["reachable"]]
    if len(reach) >= 4 and len(quartet) >= 2 and len(non_type) >= 1:
        call = "GO"
        text = (f"GO: {len(reach)}/{n_total} reachable, "
                f"{len(quartet)}/4 of the type quartet, "
                f"{len(non_type)}/5 non-type")
    elif len(reach) >= 2:
        call = "PARTIAL"
        text = (f"PARTIAL ({len(reach)}/{n_total}): the design's §6.1 EV must be "
                "REWRITTEN from this measured reachability before any C-cell "
                "launches — no inherited thresholds")
    else:
        call = "NO-GO"
        text = (f"NO-GO ({len(reach)}/{n_total}): the distribution lacks "
                "alternatives. Per design §5 P1 the next step is an s512-variant "
                "recheck on exactly the failing cohort (~+3 h, mode-at-coarser-"
                "commit-grid vs genuinely absent) — OUT OF THIS LANE'S SCOPE, "
                "cited not run.")
    return {"call": call, "text": text, "n_reachable": len(reach),
            "n_total": n_total, "n_quartet": len(quartet),
            "n_non_type": len(non_type),
            "reachable_problems": [r["problem"] for r in reach]}


def _cell_text(cell: dict) -> str:
    """Column text for one (problem, temp) cell: PASS, or n_ok/n_canvases."""
    if cell["pass"]:
        return f"{'PASS':<6}"
    return f"{cell['n_ok']}/{cell['n_canvases']:<4}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--export", required=True,
                    help="the export dir whose manifest.json names the "
                         "completion->candidate-row mapping")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    run = load_run(Path(args.run))
    results = load_results(Path(args.results))
    manifest = json.loads(
        (Path(args.export) / "manifest.json").read_text(encoding="utf-8"))
    meta = run["meta"]
    temps = sorted({float(t) for t in (meta.get("temperatures") or [])})
    print(f"P1 probe: {Path(args.run).name}")
    print(f"  problems={meta.get('n_problems')} temps={temps} "
          f"K_effective={meta.get('K_effective')} "
          f"seed_paired_across_temps={meta.get('seed_paired_across_temps')}")
    do = run["done"] or {}
    print(f"  rows={len(run['rows'])} nfe_total={do.get('nfe_total')} "
          f"gen_wall={do.get('gen_wall_s_total')}s "
          f"extract_ok={do.get('extract_ok_rows')}")
    if not meta.get("seed_paired_across_temps"):
        print("  WARNING: seed pairing is not recorded — the temp curve is "
              "confounded with seed; report it as such.")

    grid = build_grid(run, results, manifest)
    problems = meta.get("problem_names") or sorted({r["problem"] for r in run["rows"]})
    rows = [classify(p, grid, temps) for p in problems]

    print("\n=== reachability curve: P(>=1 passing canvas of K) per problem x temp ===")
    hdr = f"{'problem':44s} {'cohort':14s} " + " ".join(f"T{t:<5}" for t in temps) + "  max-pass-T  curve"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        cells = " ".join(_cell_text(r["per_temp"][t]) for t in temps)
        curve = "".join("^" if c else "." for c in r["reachability_curve"])
        max_t = str(r["max_pass_temp"]) if r["max_pass_temp"] is not None else "-"
        print(f"{r['problem']:44s} {r['cohort']:14s} {cells}  "
              f"{max_t:11s} {curve}")
    print("-" * len(hdr))

    print("\n=== container status histogram per temperature ===")
    for t in temps:
        hist = Counter()
        for (problem, temp), cell in grid.items():
            if temp == float(t):
                hist.update(cell["statuses"])
        total = sum(hist.values())
        parts = " ".join(f"{k}={v}" for k, v in
                         sorted(hist.items(), key=lambda kv: str(kv[0])))
        print(f"  T={t}: {total} canvases  {parts}")

    v = verdict(rows)
    print(f"\n=== design §5 P1 verdict ===\n  {v['text']}")
    print(f"  reachable: {', '.join(v['reachable_problems']) or 'none'}")

    if args.json:
        Path(args.json).write_text(json.dumps(
            {"meta": {k: meta.get(k) for k in
                      ("n_problems", "temperatures", "K_effective",
                       "seed_paired_across_temps")},
             "rows": rows, "verdict": v}, indent=1), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
