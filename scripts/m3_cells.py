#!/usr/bin/env python3
"""D6: paired + Bayesian cell statistics for M3 (docs/m3-design.md §4.2, §7).

rev1's gates are PAIRED, not pooled: per-problem binary outcomes are
deterministic single draws, not exchangeable Bernoulli draws, so the primary
decision quantities are (1) the per-problem migration matrix against a banked
anchor and (2) a one-sided sign test / McNemar on the discordant pairs. The
pooled Jeffreys posterior is retained as a REPORTING CONVENIENCE only, printed
with its iid caveat — it must never decide a gate (rev1 R4).

Usage — one or more cells, each "label=results.json[:run.jsonl]":

    # the §4.2 reference values, reproduced from the corrected anchors
    scripts/m3_cells.py --reproduce

    # a paired comparison (steered cell vs a banked anchor)
    scripts/m3_cells.py \
        s256=bench/multipl-e/results/20261021T144401Z.json \
        s512=bench/multipl-e/results/20261021T145224Z.json \\
        --primary s512 s256 \
        --secondary s256 s512

Results JSONs carry official statuses; honest counting folds the manifest's
excluded rows back in as failures (a problem with zero scored rows and
n_excluded == K is an exclusion = failure), which is the design's counting.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]

# The corrected-basis anchors (docs/m2.4b-report.md five-cell table). Every
# §4.2 reference number is reproducible from these files alone.
CORRECTED = {
    "s96": ("bench/multipl-e/results/20261021T144931Z.json",
            "bench/multipl-e/runs/20261019T103020Z-b2pilot.jsonl"),
    "s128": ("bench/multipl-e/results/20261021T145038Z.json",
             "bench/multipl-e/runs/20261019T183015Z-s128.jsonl"),
    "s256": ("bench/multipl-e/results/20261021T144401Z.json",
             "bench/multipl-e/runs/20261020T100251Z-s256.jsonl"),
    "s512": ("bench/multipl-e/results/20261021T145224Z.json",
             "bench/multipl-e/runs/20261020T225623Z-s512.jsonl"),
}

# §4.2 / §10.5 reference values (rev1 text): Jeffreys Beta(0.5+x, 0.5+25-x).
JEFFREYS_REFERENCES = {
    13: {"P>0.44": 0.79},
    14: {"P>0.44": 0.886},
    16: {"P>0.44": 0.978},
    18: {"P>0.64": 0.795},
}


# --------------------------------------------------------------------------
# cell loading
# --------------------------------------------------------------------------


def load_cell(spec: str) -> dict:
    """Parse 'label=results.json[:run.jsonl]' -> a cell dict with per-problem
    honest binary outcomes (pass = at least one OK canvas under K)."""
    if "=" not in spec:
        raise SystemExit(f"cell {spec!r}: expected label=results.json[:run.jsonl]")
    label, path = spec.split("=", 1)
    results_path, _, run_path = path.partition(":")
    p = Path(results_path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    d = json.loads(p.read_text(encoding="utf-8"))
    per = {}
    for row in d["per_problem"]:
        st = row["statuses_official"]
        n_ok = sum(1 for s in st if s == "OK")
        n_scored = row["n_scored"]
        n_excl = row.get("n_excluded", 0)
        # honest counting: excluded rows are scored failures, so a problem is
        # a pass iff it has >=1 OK among everything it should have produced
        per[row["name"]] = {
            "statuses": st,
            "n_scored": n_scored,
            "n_excluded": n_excl,
            "n_ok": n_ok,
            "pass": n_ok >= 1,
            "all_pass": n_ok > 0 and n_ok == n_scored and n_excl == 0,
        }
    return {
        "label": label,
        "results_path": str(p.relative_to(REPO_ROOT)) if p.is_relative_to(REPO_ROOT) else str(p),
        "run_path": run_path or None,
        "per_problem": per,
        "n_problems": len(per),
        "honest_p1": (d["aggregate"].get("honest-official", {}).get("pass@1")),
        "official_p1": d["aggregate"].get("official", {}).get("pass@1"),
        "n_excluded_rows": sum(v["n_excluded"] for v in per.values()),
        "nfe_per_problem": _nfe_per_problem(run_path),
        "gen_wall_h": _gen_wall_h(run_path),
    }


def _read_run(run_path: str | None):
    if not run_path:
        return None
    p = Path(run_path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    if not p.exists():
        return None
    meta, rows, done = None, [], None
    for line in p.read_text(encoding="utf-8").splitlines():
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
    return {"meta": meta, "rows": rows, "done": done}


def _nfe_per_problem(run_path: str | None):
    run = _read_run(run_path)
    if not run or not run["rows"]:
        return None
    k = (run["meta"] or {}).get("K") or 1
    nfe = sum((r.get("nfe") or 0) for r in run["rows"])
    return round(nfe / max(1, len(run["rows"])) * k, 1)


def _gen_wall_h(run_path: str | None):
    run = _read_run(run_path)
    if not run:
        return None
    done = run["done"] or {}
    wall = done.get("gen_wall_s_total") or sum(
        (r.get("gen_wall_s") or 0.0) for r in run["rows"])
    return round(wall / 3600.0, 2) if wall else None


# --------------------------------------------------------------------------
# paired statistics
# --------------------------------------------------------------------------


def migration_matrix(a: dict, b: dict) -> dict:
    """Per-problem pass/fail migration a -> b, plus the 2x2 table."""
    shared = sorted(set(a["per_problem"]) & set(b["per_problem"]))
    rows = []
    table = {"pass_pass": 0, "pass_fail": 0, "fail_pass": 0, "fail_fail": 0}
    for name in shared:
        pa = a["per_problem"][name]["pass"]
        pb = b["per_problem"][name]["pass"]
        key = ("pass_" if pa else "fail_") + ("pass" if pb else "fail")
        table[key] += 1
        rows.append({"problem": name, "a": pa, "b": pb, "transition": key})
    return {"rows": rows, "table": table, "n_shared": len(shared)}


def sign_test_one_sided(n_better: int, n_worse: int) -> float:
    """Exact one-sided sign test p-value: H1 = b beats a (n_better > n_worse).

    Under H0 each discordant pair is a fair coin, so
    p = P(X >= n_better | n = n_better + n_worse, p = 1/2).
    Ties (concordant pairs) are excluded, as the sign test requires.
    """
    n = n_better + n_worse
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(n_better, n + 1))
    return tail / (2 ** n)


def mcnemar_exact(n_better: int, n_worse: int) -> float:
    """Two-sided exact McNemar p-value (binomial on the discordant pairs)."""
    n = n_better + n_worse
    if n == 0:
        return 1.0
    k = min(n_better, n_worse)
    tail = sum(math.comb(n, i) for i in range(0, k + 1))
    return min(1.0, 2.0 * tail / (2 ** n))


def jeffreys_posterior(x: int, n: int = 25, draws: int = 2_000_000,
                       seed: int = 20261021) -> dict:
    """Pooled Jeffreys Beta(0.5+x, 0.5+n-x) reference — REPORTING ONLY.

    rev1 demoted this to a reporting convenience: treating per-problem
    deterministic draws as iid Bernoulli overstates confidence. Never a gate.
    """
    rng = np.random.default_rng(seed)
    d = rng.beta(0.5 + x, 0.5 + n - x, size=draws)
    return {
        "x": x, "n": n,
        "mean": round(float(d.mean()), 4),
        "P>0.44": round(float((d > 0.44).mean()), 3),
        "P>0.64": round(float((d > 0.64).mean()), 3),
        "P>0.72": round(float((d > 0.72).mean()), 3),
        "caveat": ("reporting-only: pooled iid Beta over deterministic "
                   "single per-problem draws (rev1 §4.2)"),
    }


def detectable_effect_statement(n_pairs: int) -> str:
    """Ex-ante detectable-effect note every gate line carries (§4.2)."""
    return (f"one-sided alpha 0.05 at n={n_pairs} paired draws needs >=5 "
            "discordant problems all in one direction (P=1/32); fewer flips, "
            "or flips split between directions, cannot clear the sign test — "
            "every gate line carries that caveat (rev1 §4.2)")


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------


def cell_block(cell: dict) -> dict:
    """The §4.2 per-cell block: NFE, wall, verifier wall, exclusion counts."""
    return {
        "label": cell["label"],
        "n_problems": cell["n_problems"],
        "honest_p1": cell["honest_p1"],
        "official_p1": cell["official_p1"],
        "n_excluded_rows": cell["n_excluded_rows"],
        "nfe_per_problem": cell["nfe_per_problem"],
        "gen_wall_h": cell["gen_wall_h"],
        "n_passers": sum(1 for v in cell["per_problem"].values() if v["pass"]),
    }


def print_cell(cell: dict) -> None:
    b = cell_block(cell)
    print(f"[{b['label']}] problems={b['n_problems']} "
          f"passers={b['n_passers']} hon_p@1={b['honest_p1']} "
          f"(official {b['official_p1']}) excl_rows={b['n_excluded_rows']} "
          f"NFE/prob={b['nfe_per_problem']} gen_wall={b['gen_wall_h']}h")
    j = jeffreys_posterior(b["n_passers"])
    print(f"  Jeffreys pooled (REPORTING ONLY): mean={j['mean']} "
          f"P(>0.44)={j['P>0.44']} P(>0.64)={j['P>0.64']} P(>0.72)={j['P>0.72']}")
    print(f"  caveat: {j['caveat']}")


def print_paired(a: dict, b: dict, *, primary: bool = False) -> dict:
    m = migration_matrix(a, b)
    t = m["table"]
    p_one = sign_test_one_sided(t["fail_pass"], t["pass_fail"])
    p_two = mcnemar_exact(t["fail_pass"], t["pass_fail"])
    tag = "PRIMARY" if primary else "secondary (exploratory)"
    print(f"\n--- paired {a['label']} -> {b['label']}  [{tag}] ---")
    print(f"  table: pass->pass {t['pass_pass']}  pass->fail {t['pass_fail']}  "
          f"fail->pass {t['fail_pass']}  fail->fail {t['fail_fail']}  "
          f"(n={m['n_shared']})")
    print(f"  sign test one-sided p={p_one:.4f}   "
          f"McNemar exact two-sided p={p_two:.4f}")
    print(f"  {detectable_effect_statement(m['n_shared'])}")
    flips = [r["problem"] for r in m["rows"] if r["transition"] !=
             ("pass_pass" if r["a"] else "fail_fail")]
    if flips:
        print(f"  discordant problems: {', '.join(flips)}")
    return {"table": t, "sign_test_p_one_sided": round(p_one, 6),
            "mcnemar_exact_p_two_sided": round(p_two, 6),
            "n_shared": m["n_shared"], "discordant": flips,
            "detectable_effect": detectable_effect_statement(m["n_shared"])}


def reproduce_references() -> int:
    """D6 acceptance: reproduce the §4.2 reference values from the corrected
    anchors exactly (honest p@1 0.16/0.20/0.44/0.64 and the Jeffreys table)."""
    print("=== corrected-basis anchors (docs/m2.4b-report.md) ===")
    cells = {}
    for label in ("s96", "s128", "s256", "s512"):
        results, run = CORRECTED[label]
        cells[label] = load_cell(f"{label}={results}:{run}")
        print_cell(cells[label])
    expected = {"s96": 0.16, "s128": 0.20, "s256": 0.44, "s512": 0.64}
    ok = True
    print("\n=== anchor reproduction vs docs/m2.4b-report.md ===")
    for label, hon in expected.items():
        got = cells[label]["honest_p1"]
        good = got is not None and abs(got - hon) < 5e-5
        ok &= good
        print(f"  {label:5s}: honest p@1 {got} expected {hon} "
              f"{'OK' if good else 'MISMATCH'}")
    print("\n=== Jeffreys reference values (§4.2 / §10.5) ===")
    for x, refs in JEFFREYS_REFERENCES.items():
        j = jeffreys_posterior(x)
        for key, val in refs.items():
            got = j[key]
            good = abs(got - val) <= 0.006
            ok &= good
            print(f"  x={x:2d} {key}: {got} expected {val} "
                  f"{'OK' if good else 'MISMATCH'}")
    print(f"\nD6 reference reproduction: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cells", nargs="*", help="label=results.json[:run.jsonl]")
    ap.add_argument("--reproduce", action="store_true",
                    help="reproduce the §4.2 reference values from the anchors")
    ap.add_argument("--primary", nargs=2, metavar=("A", "B"),
                    help="the pre-registered primary paired comparison A -> B")
    ap.add_argument("--secondary", nargs=2, action="append", default=[],
                    metavar=("A", "B"), help="secondary/exploratory comparison")
    ap.add_argument("--json", default=None, help="write the report as JSON")
    args = ap.parse_args(argv)

    if args.reproduce:
        return reproduce_references()
    if not args.cells:
        ap.print_help()
        return 2

    cells = {}
    for spec in args.cells:
        c = load_cell(spec)
        cells[c["label"]] = c
        print_cell(c)

    report = {"cells": {k: cell_block(v) for k, v in cells.items()},
              "primary": None, "secondary": []}
    if args.primary:
        a, b = args.primary
        for lbl in (a, b):
            if lbl not in cells:
                raise SystemExit(f"--primary names unknown cell {lbl!r}")
        report["primary"] = print_paired(cells[a], cells[b], primary=True)
    for a, b in args.secondary:
        for lbl in (a, b):
            if lbl not in cells:
                raise SystemExit(f"--secondary names unknown cell {lbl!r}")
        report["secondary"].append(print_paired(cells[a], cells[b]))
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=1), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
