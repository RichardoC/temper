#!/usr/bin/env python3
"""M3.0 P3 PHASE D: the pre-registered paired analysis and the SHIP gate.

Design §4.2 fixes the decision quantities and their order of authority:

  1. **Paired migration matrix + one-sided sign test (PRIMARY).** Per-problem
     outcomes are deterministic single draws, not exchangeable Bernoulli draws,
     so the sign test on discordant pairs decides; the pooled Jeffreys
     posterior is printed REPORTING-ONLY with its iid caveat.
  2. **The pre-registered primary comparison is C-full@2048 vs C-null@2048.**
     Its outcome decides the mechanism question (§0.5b). Every other
     comparison is secondary/exploratory and labelled so.

This script runs the whole matrix, decomposes flips by cohort, and evaluates the
gates against their EXACT pre-registered text:

  SHIP  : C-full@2048 >= 15/25 honest WITH paired margins (zero net
          regressions among the 11 corrected-s256 passers, sign test vs
          anchor-1024 reported) and >= 2 confirmed-attributed flips.
  ATTRIB: C-full > max(C-null, C-screen) on the paired matrix, margin >= +2
          problems, with at least one flip OUTSIDE the hint-fixable /
          ladder-reachable cohorts, flip-set replication green.
  STRONG: >= 18/25 at <= 2048 with the same paired discipline.

Every gate line carries the ex-ante detectable-effect statement: at n=25 paired
draws, alpha=0.05 one-sided needs >= 5 discordant problems all in one direction.

    temper/.venv/bin/python scripts/m3_p3_analysis.py \\
        --cells anchor1024=<results>:<run> anchor2048=<results>:<run> \\
                cnull=<results> cscreen=<results> cfull=<results>:<run> \\
        --primary cfull cnull --json /tmp/m3_p3_analysis.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]

# ---- the pre-registered cohort table (design §6.1, by HumanEval NUMBER) ----
TYPE_QUARTET = (0, 4, 20, 21)           # the E0282/type-trap quartet
SEMANTIC_TRIO = (11, 18, 24)            # wrong-but-confident
EXCLUSION_PAIR = (17, 19)               # prose/fence pathology
# The 17-reachable class: the P1 probe found 17 reachable at T=0.8 while every
# banked unsteered cell excluded it — the design classified it as a prompt-shape
# pathology, the probe showed it is a DIVERSITY problem. It therefore joins the
# ladder-reachable set, which makes the attribution gate's outside-cohort
# requirement STRICTER than the design text assumed (m3.0-report.md).
P1_REACHABLE = (4, 11, 17, 20, 24)
P1_UNREACHABLE = (0, 18, 19, 21)
# The 11 corrected-s256 passers (the zero-regression guard's denominator).
S256_PASSERS = (1, 2, 3, 5, 7, 8, 13, 14, 15, 16, 23)


def _num(name: str) -> int:
    m = re.match(r"^HumanEval_(\d+)_", name)
    if not m:
        raise ValueError(f"cannot read a HumanEval number from {name!r}")
    return int(m.group(1))


def cohort_of(name: str) -> str:
    n = _num(name)
    if n in TYPE_QUARTET:
        return "type_quartet"
    if n in SEMANTIC_TRIO:
        return "semantic_trio"
    if n in EXCLUSION_PAIR:
        return "exclusion_pair"
    if n in S256_PASSERS:
        return "structural_base"
    return "other"


def cohort_tags(name: str) -> List[str]:
    """Every cohort label a problem belongs to (they overlap by design)."""
    n = _num(name)
    tags = [cohort_of(name)]
    if n in P1_REACHABLE:
        tags.append("p1_reachable")
    if n in P1_UNREACHABLE:
        tags.append("p1_unreachable")
    if n in S256_PASSERS:
        tags.append("s256_passer")
    return tags


def ladder_reachable_cohorts() -> set:
    """The cohorts the attribution gate says a flip must NOT be confined to.

    A flip inside the type-trap quartet could be knowledge delivery (C-hint's
    question); a flip inside the ladder-reachable set could be diversity alone
    (C-null's question). The gate wants at least one flip that is NEITHER —
    that is what stops "C-null 14, C-full 15" from being sold as a mechanism.
    """
    return {"type_quartet"} | {f"p1_reachable:{n}" for n in P1_REACHABLE}


# --------------------------------------------------------------------------
# cell loading
# --------------------------------------------------------------------------


def load_cell(spec: str) -> dict:
    """'label=results.json[:run.jsonl]' -> per-problem honest pass/fail."""
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
        st = row.get("statuses_official") or []
        n_ok = sum(1 for s in st if s == "OK")
        per[row["name"]] = {
            "statuses": st,
            "n_ok": n_ok,
            "n_scored": row.get("n_scored", len(st)),
            "n_excluded": row.get("n_excluded", 0),
            "pass": n_ok >= 1,
            "emitted": row.get("emitted"),
            "emitted_temp": row.get("emitted_temp"),
        }
    cell = {
        "label": label,
        "results_path": str(p),
        "per_problem": per,
        "n_problems": len(per),
        "honest_p1": (d.get("aggregate", {}).get("honest-official", {})
                      .get("pass@1")),
        "n_passers": sum(1 for v in per.values() if v["pass"]),
        "run_path": run_path or None,
    }
    if cell["honest_p1"] is None and cell["n_problems"]:
        cell["honest_p1"] = round(cell["n_passers"] / cell["n_problems"], 4)
    return cell


# --------------------------------------------------------------------------
# paired statistics (D6 parity — the same formulas as scripts/m3_cells.py)
# --------------------------------------------------------------------------


def migration(a: dict, b: dict) -> dict:
    shared = sorted(set(a["per_problem"]) & set(b["per_problem"]))
    table = {"pass_pass": 0, "pass_fail": 0, "fail_pass": 0, "fail_fail": 0}
    rows = []
    for name in shared:
        pa = a["per_problem"][name]["pass"]
        pb = b["per_problem"][name]["pass"]
        key = ("pass_" if pa else "fail_") + ("pass" if pb else "fail")
        table[key] += 1
        rows.append({"problem": name, "number": _num(name), "a": pa, "b": pb,
                     "transition": key, "cohorts": cohort_tags(name),
                     "cohort": cohort_of(name)})
    return {"rows": rows, "table": table, "n_shared": len(shared)}


def sign_test_one_sided(n_better: int, n_worse: int) -> float:
    n = n_better + n_worse
    if n == 0:
        return 1.0
    return sum(__import__("math").comb(n, k) for k in range(n_better, n + 1)) / (2 ** n)


def jeffreys(x: int, n: int = 25, draws: int = 2_000_000,
             seed: int = 20261021) -> dict:
    """REPORTING ONLY (rev1 §4.2): pooled iid Beta over deterministic draws."""
    rng = np.random.default_rng(seed)
    d = rng.beta(0.5 + x, 0.5 + n - x, size=draws)
    return {"x": x, "n": n, "mean": round(float(d.mean()), 4),
            "P>0.44": round(float((d > 0.44).mean()), 3),
            "P>0.64": round(float((d > 0.64).mean()), 3),
            "P>0.72": round(float((d > 0.72).mean()), 3)}


def detectable(n: int) -> str:
    need = 5 if n >= 25 else max(1, int(np.ceil(n / 5)))
    return (f"one-sided alpha 0.05 at n={n} paired draws needs >=~{need} "
            "discordant problems all in one direction; fewer flips, or flips "
            "split between directions, cannot clear the sign test (rev1 §4.2)")


def paired_report(a: dict, b: dict) -> dict:
    m = migration(a, b)
    t = m["table"]
    p_one = sign_test_one_sided(t["fail_pass"], t["pass_fail"])
    flips_up = [r["problem"] for r in m["rows"] if r["transition"] == "fail_pass"]
    flips_down = [r["problem"] for r in m["rows"] if r["transition"] == "pass_fail"]
    return {
        "a": a["label"], "b": b["label"],
        "a_passers": a["n_passers"], "b_passers": b["n_passers"],
        "delta": b["n_passers"] - a["n_passers"],
        "table": t,
        "sign_test_p_one_sided": round(p_one, 6),
        "n_discordant": t["fail_pass"] + t["pass_fail"],
        "flips_up": flips_up, "flips_down": flips_down,
        "detectable_effect": detectable(m["n_shared"]),
        "rows": m["rows"],
    }


def print_paired(rep: dict, *, primary: bool) -> None:
    tag = "PRIMARY (pre-registered)" if primary else "secondary (exploratory)"
    print(f"\n--- {rep['a']} -> {rep['b']}  [{tag}] ---")
    print(f"  {rep['a']}={rep['a_passers']}/25  {rep['b']}={rep['b_passers']}/25  "
          f"delta={rep['delta']:+d}")
    t = rep["table"]
    print(f"  migration: pass->pass {t['pass_pass']}  pass->fail {t['pass_fail']}  "
          f"fail->pass {t['fail_pass']}  fail->fail {t['fail_fail']}")
    print(f"  sign test one-sided p={rep['sign_test_p_one_sided']:.4f} "
          f"(n_discordant={rep['n_discordant']})")
    if rep["flips_up"]:
        print(f"  flips UP  : {', '.join(rep['flips_up'])}")
    if rep["flips_down"]:
        print(f"  flips DOWN: {', '.join(rep['flips_down'])}")
    print(f"  {rep['detectable_effect']}")


# --------------------------------------------------------------------------
# gates
# --------------------------------------------------------------------------


def gate_ship(cfull: dict, anchor1024: dict, cnull: dict,
              replication: Optional[dict]) -> dict:
    """Design §4.2 SHIP GATE, evaluated against its exact text."""
    n = cfull["n_passers"]
    regressions = [name for name in cfull["per_problem"]
                   if _num(name) in S256_PASSERS
                   and not cfull["per_problem"][name]["pass"]]
    vs_anchor = paired_report(anchor1024, cfull)
    n_confirmed = 0
    if replication:
        n_confirmed = sum(1 for v in replication.get("problems", [])
                          if v.get("confirmed_attributed"))
    checks = {
        "passers_ge_15": n >= 15,
        "zero_net_regressions_on_11_s256_passers": len(regressions) == 0,
        "sign_test_vs_anchor1024_reported": True,
        "confirmed_attributed_flips_ge_2": n_confirmed >= 2,
    }
    return {
        "gate": "SHIP (M3.0, design §4.2)",
        "criterion": ("C-full@2048 >= 15/25 honest WITH paired margins: zero net "
                      "regressions among the 11 corrected-s256 passers, sign test "
                      "vs anchor-1024 reported, and >= 2 confirmed-attributed "
                      "flips (replication >= 2 games/problem on the flip set)"),
        "passers": n, "checks": checks, "regressions": regressions,
        "n_confirmed_attributed_flips": n_confirmed,
        "sign_test_vs_anchor1024": vs_anchor["sign_test_p_one_sided"],
        "pass": all(checks.values()),
        "underpowered_note": ("underpowered for effects < +3 problems — every "
                              "ship/no-ship line says so"),
    }


def gate_attribution(cfull: dict, cnull: dict, cscreen: dict,
                     replication: Optional[dict]) -> dict:
    """Design §4.2 ATTRIBUTION GATE + the report's stricter reading of it."""
    vs_null = paired_report(cnull, cfull)
    vs_screen = paired_report(cscreen, cfull)
    margin_null = vs_null["delta"]
    margin_screen = vs_screen["delta"]
    # outside-cohort requirement: at least one flip (vs the BETTER null) that is
    # neither in the type-trap quartet nor in the P1 ladder-reachable set
    better = cnull if cnull["n_passers"] >= cscreen["n_passers"] else cscreen
    vs_better = paired_report(better, cfull)
    outside = []
    for name in vs_better["flips_up"]:
        tags = cohort_tags(name)
        if "type_quartet" in tags:
            continue
        if "p1_reachable" in tags:
            continue
        outside.append(name)
    supported = []
    if replication:
        supported = [v["problem"] for v in replication.get("problems", [])
                     if v.get("confirmed_attributed")]
    checks = {
        "cfull_gt_max_nulls_by_ge_2": (cfull["n_passers"] - max(
            cnull["n_passers"], cscreen["n_passers"])) >= 2,
        "margin_vs_cnull_ge_2": margin_null >= 2,
        "margin_vs_cscreen_ge_2": margin_screen >= 2,
        "at_least_one_flip_outside_ladder_reachable": len(outside) >= 1,
        "flip_set_replication_green": bool(replication
                                           and replication.get("all_green")),
    }
    return {
        "gate": "ATTRIBUTION (M3.0, design §4.2 — the causal story)",
        "criterion": ("C-full@2048 > max(C-null, C-screen) with margin >= +2 "
                      "problems, at least one flip OUTSIDE the hint-fixable / "
                      "ladder-reachable cohorts, flip-set replication green"),
        "cfull": cfull["n_passers"], "cnull": cnull["n_passers"],
        "cscreen": cscreen["n_passers"],
        "margin_vs_cnull": margin_null, "margin_vs_cscreen": margin_screen,
        "outside_cohort_flips": outside,
        "replicated_attributed_flips": supported,
        "checks": checks, "pass": all(checks.values()),
        "error_forensics_note": (
            "C-full - C-screen is the pre-registered mechanism delta: a "
            "divergence between the offline counterfactual and the in-loop "
            "engine is ENGINE forensics (design rev2 V3), not mechanism "
            "evidence — diff each problem's emitted canvas pair table."),
    }


def gate_strong(cfull: dict) -> dict:
    n = cfull["n_passers"]
    return {
        "gate": "STRONG (M3.0, design §4.2)",
        "criterion": "C-full >= 18/25 at <= 2048 NFE with the same paired discipline",
        "passers": n, "pass": n >= 18,
    }


def cohort_decomposition(cfull: dict, cnull: dict, cscreen: dict,
                         anchor1024: dict) -> dict:
    """Flips decomposed by cohort (design §5 P3 / report requirement)."""
    out = {}
    for label, base in (("vs_anchor1024", anchor1024), ("vs_cnull", cnull),
                        ("vs_cscreen", cscreen)):
        rep = paired_report(base, cfull)
        by_cohort: Dict[str, dict] = {}
        for name in rep["flips_up"]:
            c = cohort_of(name)
            by_cohort.setdefault(c, {"up": [], "down": []})["up"].append(name)
        for name in rep["flips_down"]:
            c = cohort_of(name)
            by_cohort.setdefault(c, {"up": [], "down": []})["down"].append(name)
        out[label] = {"delta": rep["delta"], "by_cohort": by_cohort}
    return out


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cells", nargs="+", required=True,
                    help="label=results.json[:run.jsonl]")
    ap.add_argument("--primary", nargs=2, metavar=("A", "B"),
                    help="the pre-registered primary comparison (cfull cnull)")
    ap.add_argument("--anchor1024", default="anchor1024",
                    help="cell label of the corrected s256 anchor (the 0.44 line)")
    ap.add_argument("--replication", default=None,
                    help="flip-set replication JSON (see --replication-from)")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    cells = {}
    for spec in args.cells:
        c = load_cell(spec)
        cells[c["label"]] = c

    print("=== cells ===")
    for c in cells.values():
        j = jeffreys(c["n_passers"])
        print(f"[{c['label']:12s}] passers={c['n_passers']}/25 "
              f"honest_p@1={c['honest_p1']}")
        print(f"  Jeffreys pooled REPORTING-ONLY: mean={j['mean']} "
              f"P(>0.44)={j['P>0.44']} P(>0.64)={j['P>0.64']} "
              f"P(>0.72)={j['P>0.72']}  [iid caveat: deterministic single "
              "per-problem draws]")

    report = {"cells": {k: {"passers": v["n_passers"],
                            "honest_p1": v["honest_p1"],
                            "jeffreys_reporting_only": jeffreys(v["n_passers"])}
                        for k, v in cells.items()},
              "paired": {}, "gates": {}, "cohorts": {}}

    if args.primary:
        a, b = args.primary
        rep = paired_report(cells[a], cells[b])
        print_paired(rep, primary=True)
        report["paired"]["primary"] = rep

    # every other pre-registered/secondary comparison the report needs
    plan = [("anchor1024", "anchor2048"), ("anchor1024", "cnull"),
            ("anchor2048", "cnull"), ("cnull", "cfull"),
            ("cscreen", "cfull"), ("anchor1024", "cfull")]
    for a, b in plan:
        if a in cells and b in cells and [a, b] != args.primary:
            rep = paired_report(cells[a], cells[b])
            print_paired(rep, primary=False)
            report["paired"][f"{a}->{b}"] = rep

    replication = None
    if args.replication:
        replication = json.loads(Path(args.replication).read_text(encoding="utf-8"))

    print("\n=== gates (design §4.2) ===")
    if all(k in cells for k in ("cfull", "cnull", "cscreen", "anchor1024")):
        for g, fn in (("ship", lambda: gate_ship(cells["cfull"],
                                                 cells[args.anchor1024],
                                                 cells["cnull"], replication)),
                      ("attribution", lambda: gate_attribution(
                          cells["cfull"], cells["cnull"], cells["cscreen"],
                          replication)),
                      ("strong", lambda: gate_strong(cells["cfull"]))):
            res = fn()
            report["gates"][g] = res
            print(f"\n[{res['gate']}]  => {'PASS' if res['pass'] else 'FAIL'}")
            print(f"  criterion: {res['criterion']}")
            if "checks" in res:
                for k, v in res["checks"].items():
                    print(f"    {'OK ' if v else 'NO '} {k}")
            if g == "ship":
                print(f"  passers={res['passers']} regressions={res['regressions']}"
                      f" confirmed_attributed_flips="
                      f"{res['n_confirmed_attributed_flips']}")
                print(f"  {res['underpowered_note']}")
            if g == "attribution":
                print(f"  C-full={res['cfull']} C-null={res['cnull']} "
                      f"C-screen={res['cscreen']} margins="
                      f"{res['margin_vs_cnull']:+d}/{res['margin_vs_cscreen']:+d}")
                print(f"  outside-cohort flips: {res['outside_cohort_flips'] or 'NONE'}")
                print(f"  {res['error_forensics_note']}")
        report["cohorts"] = cohort_decomposition(
            cells["cfull"], cells["cnull"], cells["cscreen"],
            cells[args.anchor1024])
        print("\n=== flips decomposed by cohort ===")
        for label, d in report["cohorts"].items():
            print(f"  {label} (delta {d['delta']:+d}):")
            for cohort, fl in sorted(d["by_cohort"].items()):
                print(f"    {cohort:18s} up={fl['up']} down={fl['down']}")

    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=1), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
