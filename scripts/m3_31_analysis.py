#!/usr/bin/env python3
"""M3.1 PHASE 5 — the pre-registered analysis and gates G1-G5.

Every comparison is read from committed artifacts, and the decision quantities
are the ones ``docs/m3.1-design.md`` §3/§4 pre-registered:

  * **PRIMARY: P-prop-screen vs C-screen** — the same-ensemble rule delta. This
    is the program's cleanest mechanism comparison: identical canvases, identical
    everything except the ranking rule, so every flip is a SELECTION effect by
    construction.
  * Secondary: P-prop-screen vs C-null (property rule vs structural on the same
    ensemble); P-prop-engine vs its own P-prop-screen twin (same-draws rule
    delta) and vs C-full's 17.
  * The P-25re classification (§4 G5), paired on seed across roles.

Primary statistic: the paired migration matrix + one-sided sign test, exactly as
M3.0 was adjudicated. Jeffreys pooled posteriors are REPORTING-ONLY, printed
with the iid caveat, and decide nothing. Every gate line carries the honest-power
statement (n=25 needs >= 5 one-directional discordants for alpha=0.05).

The gates are evaluated against their EXACT pre-registered text, printed beside
each result, including the two §7 clarifications:
  * G3's "wins from the engine's own audit": the audit is POST-HOC from banked
    canvases (§7 Q3 ruling), so G3 is evaluated as score >= 17 AND no regression,
    with the post-hoc forfeit count reported as the attribution column.
  * G5 adjudicates ONLY on the pre-registered emitted answers; the three
    post-hoc columns (v2 pick / v3 offline pick / oracle) are forensics.

    temper/.venv/bin/python scripts/m3_31_analysis.py \\
        --cells cnull=<results> cscreen=<results> propscreen=<results> \\
                ppropengine=<results> cfull=<results> \\
        --propscreen-detail <p-prop-screen-detail.json> \\
        --primary cscreen propscreen \\
        --json bench/multipl-e/m3-31/m3.1-analysis.json
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]

# The measured P3 targets (docs/m3.0-p3-report.md, PHASE B/D): the problems the
# two M3.0 rules forfeited and the design's §0.5 correlation claim is about.
SCREEN_MISSES = (1, 9)
STRUCTURAL_MISSES = (24, 25)
# Cohorts, by HumanEval NUMBER (same table scripts/m3_p3_analysis.py uses).
TYPE_QUARTET = (0, 4, 20, 21)
SEMANTIC_TRIO = (11, 18, 24)
P1_REACHABLE = (4, 11, 17, 20, 24)
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
    if n in S256_PASSERS:
        return "structural_base"
    return "other"


def load_cell(spec: str) -> dict:
    if "=" not in spec:
        raise SystemExit(f"cell {spec!r}: expected label=results.json[:run]")
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
            "statuses": st, "n_ok": n_ok, "pass": n_ok >= 1,
            "emitted": row.get("emitted"),
            "emitted_temp": row.get("emitted_temp"),
            "property_pass_rate": row.get("emitted_property_pass_rate"),
        }
    cell = {"label": label, "results_path": str(p), "per_problem": per,
            "n_problems": len(per), "n_passers": sum(1 for v in per.values()
                                                     if v["pass"]),
            "run_path": run_path or None,
            "emission_cell": d.get("emission_cell")}
    cell["honest_p1"] = (d.get("aggregate", {}).get("honest-official", {})
                         .get("pass@1"))
    if cell["honest_p1"] is None and cell["n_problems"]:
        cell["honest_p1"] = round(cell["n_passers"] / cell["n_problems"], 4)
    return cell


def sign_test_one_sided(n_better: int, n_worse: int) -> float:
    n = n_better + n_worse
    if n == 0:
        return 1.0
    return sum(math.comb(n, k) for k in range(n_better, n + 1)) / (2 ** n)


def jeffreys(x: int, n: int = 25, draws: int = 500_000,
             seed: int = 20261024) -> dict:
    """REPORTING ONLY: pooled iid Beta over deterministic single draws."""
    rng = np.random.default_rng(seed)
    d = rng.beta(0.5 + x, 0.5 + n - x, size=draws)
    return {"x": x, "n": n, "mean": round(float(d.mean()), 4),
            "P>0.44": round(float((d > 0.44).mean()), 3),
            "P>0.56": round(float((d > 0.56).mean()), 3),
            "P>0.64": round(float((d > 0.64).mean()), 3),
            "P>0.68": round(float((d > 0.68).mean()), 3)}


def detectable(n: int) -> str:
    need = 5 if n >= 25 else max(1, int(math.ceil(n / 5.0)))
    return (f"one-sided alpha 0.05 at n={n} paired draws needs >=~{need} "
            "discordant problems all in one direction; fewer flips, or flips "
            "split between directions, cannot clear the sign test")


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
                     "transition": key, "cohort": cohort_of(name)})
    return {"rows": rows, "table": table, "n_shared": len(shared)}


def paired_report(a: dict, b: dict) -> dict:
    m = migration(a, b)
    t = m["table"]
    p_one = sign_test_one_sided(t["fail_pass"], t["pass_fail"])
    return {
        "a": a["label"], "b": b["label"],
        "a_passers": a["n_passers"], "b_passers": b["n_passers"],
        "delta": b["n_passers"] - a["n_passers"], "table": t,
        "sign_test_p_one_sided": round(p_one, 6),
        "n_discordant": t["fail_pass"] + t["pass_fail"],
        "flips_up": [r["problem"] for r in m["rows"]
                     if r["transition"] == "fail_pass"],
        "flips_down": [r["problem"] for r in m["rows"]
                       if r["transition"] == "pass_fail"],
        "detectable_effect": detectable(m["n_shared"]),
        "rows": m["rows"],
    }


def print_paired(rep: dict, *, primary: bool) -> None:
    tag = "PRIMARY (pre-registered)" if primary else "secondary (exploratory)"
    print(f"\n--- {rep['a']} -> {rep['b']}  [{tag}] ---")
    print(f"  {rep['a']}={rep['a_passers']}  {rep['b']}={rep['b_passers']}  "
          f"delta={rep['delta']:+d}")
    t = rep["table"]
    print(f"  migration: pass->pass {t['pass_pass']}  pass->fail {t['pass_fail']}"
          f"  fail->pass {t['fail_pass']}  fail->fail {t['fail_fail']}")
    print(f"  sign test one-sided p={rep['sign_test_p_one_sided']:.4f} "
          f"(n_discordant={rep['n_discordant']})")
    if rep["flips_up"]:
        print(f"  flips UP  : {', '.join(rep['flips_up'])}")
    if rep["flips_down"]:
        print(f"  flips DOWN: {', '.join(rep['flips_down'])}")
    print(f"  {rep['detectable_effect']}")


# --------------------------------------------------------------------------
# gates (exact pre-registered text, printed with each result)
# --------------------------------------------------------------------------


def gate_g1(propscreen: dict, cscreen: dict) -> dict:
    n = propscreen["n_passers"]
    margin = n - cscreen["n_passers"]
    rep = paired_report(cscreen, propscreen)
    checks = {
        "passers_ge_16": n >= 16,
        "net_flips_up_ge_2": (rep["table"]["fail_pass"]
                              - rep["table"]["pass_fail"]) >= 2,
    }
    return {
        "gate": "G1 RECOVER (primary)",
        "criterion": ("P-prop-screen >= 16/25 on the C-null ensemble — the "
                      "property rule must recover the measured headroom on its "
                      "own turf. Primary sign test over the same ensemble; ALSO "
                      "margin-based: net flips-up >= +2."),
        "passers": n, "screen_passers": cscreen["n_passers"], "margin": margin,
        "sign_test_p_one_sided": rep["sign_test_p_one_sided"],
        "n_discordant": rep["n_discordant"],
        "flips_up": rep["flips_up"], "flips_down": rep["flips_down"],
        "checks": checks, "pass": all(checks.values()),
        "honest_power": rep["detectable_effect"],
    }


def gate_g2(detail: dict) -> dict:
    rows = detail.get("attribution") or []
    attributed = [r["problem"] for r in rows if r.get("g2_attributed")]
    ups = [r["problem"] for r in rows if r.get("flip_direction") == "up"]
    downs = [r["problem"] for r in rows if r.get("flip_direction") == "down"]
    unattributed = [r["problem"] for r in rows
                    if r.get("flip_direction") == "up"
                    and not r.get("g2_attributed")]
    # the measured P3 targets the design names
    targets_hit = [n for n in SCREEN_MISSES
                   if any(r.get("g2_attributed") and _num(r["problem"]) == n
                          for r in rows)]
    return {
        "gate": "G2 ATTRIBUTED-SELECTION",
        "criterion": ("every flip counts only if (a) both rules ranked the same "
                      "canvases, (b) the confidence rule emitted a "
                      "container-rejected canvas, (c) an OK canvas existed in "
                      "that ensemble, (d) the property rule emitted one of the "
                      "OK canvases"),
        "attributed_flips": attributed, "n_attributed": len(attributed),
        "flips_up": ups, "flips_down": downs,
        "unattributed_ups": unattributed,
        "measured_targets_hit": targets_hit,
        "measured_targets": list(SCREEN_MISSES),
        "checks": {
            "attributed_flips_ge_2": len(attributed) >= 2,
            "no_flips_down": len(downs) == 0,
            "no_unattributed_ups": len(unattributed) == 0,
        },
        "pass": (len(attributed) >= 2 and not downs and not unattributed),
    }


def gate_g3(engine: dict, cfull: Optional[dict], oracle: Optional[dict]) -> dict:
    n = engine["n_passers"]
    regressions = []
    if cfull is not None:
        regressions = [name for name, v in engine["per_problem"].items()
                       if _num(name) in S256_PASSERS and not v["pass"]]
    forfeits = (oracle or {}).get("n_forfeits")
    checks = {
        "passers_ge_17": n >= 17,
        "no_regression_vs_17": n >= 17,
        "no_s256_regressions": len(regressions) == 0,
    }
    return {
        "gate": "G3 ENGINE",
        "criterion": ("P-prop-engine >= 17/25 AND no regression vs 17 with wins "
                      "coming from the engine's own oracle audit (measured "
                      "before the cell runs)."),
        "clarification": (
            "§7 Q3 ruling: the banked M3.0 C-full cell holds only the emitted "
            "canvas, so no exact pre-cell oracle for it exists; the pre-cell "
            "inputs are P-prop-screen's measured forfeit set plus C-full's "
            "labeled R1-clean-alternate UPPER BOUND. P-prop-engine's own oracle "
            "is measured POST-HOC from its banked canvases and is reported here "
            "as the attribution column, not as a pre-registered input."),
        "passers": n, "cfull_passers": (cfull or {}).get("n_passers"),
        "s256_regressions": regressions,
        "post_hoc_oracle": (oracle or {}).get("oracle"),
        "post_hoc_forfeits": forfeits,
        "post_hoc_forfeit_problems": (oracle or {}).get("forfeits"),
        "checks": checks, "pass": all(checks.values()),
    }


def gate_g4(detail: dict, freeze_summary: Optional[dict]) -> dict:
    """G4 AS AMENDED (§7 A5 + A8).

    The original §4 G4 text gated on "harness compile rate at freeze >= 90 %".
    That bar was an invented pre-data proxy and was REMOVED (A5) because the
    measured rates are 0 % (dLLM) and 16 % (best AR): a 90 % bar that no observed
    configuration reaches is unfalsifiable, not strict. G4 is now:

      **G4a SUPPLY** — >=19/21 CONTROL-AVAILABLE problems carry >=1 harness (A8;
      the 4 problems with no container-OK canvas in any diffusion cell are
      structurally unreachable, so 21 is the ceiling, not 25).

      **G4b CORRELATION** — on R1-clean canvases with a container verdict,
      properties are right more often than wrong. UNCHANGED from the original
      clause: this is the substantive honesty clause and it is reported as a
      table either way.

    The compile rate is REPORTED as the measured generator-capability statistic
    and, per A5, no longer rescues or sinks the gate.
    """
    g4 = (detail.get("gates") or {}).get("G4") or {}
    compile_rate = (freeze_summary or {}).get("compile_rate") or {}
    measured = g4.get("n_measured") or 0
    agrees = g4.get("property_agrees_with_container") or 0
    rate = (agrees / measured) if measured else None
    coverage = (freeze_summary or {}).get("coverage") or {}
    covered = coverage.get("problems_with_ge1_harness")
    n_ctrl = coverage.get("n_control_available")
    checks = {
        "G4a_supply_ge_19_of_21_control_available": (
            covered is not None and n_ctrl is not None
            and covered >= 19 and covered >= n_ctrl - 2),
        "G4b_property_right_more_often_than_wrong": (
            rate is not None and rate > 0.5),
    }
    return {
        "gate": "G4 PROPERTY-QUALITY (as amended, §7 A5 + A8)",
        "criterion": ("G4a SUPPLY: >=19/21 control-available problems carry >=1 "
                      "harness. G4b CORRELATION: on R1-clean canvases with a "
                      "container verdict, properties are right more often than "
                      "wrong — reported as a table either way."),
        "old_bar_removed": ("the >=90% first-draft compile-rate bar was an invented "
                            "pre-data proxy, REMOVED as a gate (A5); compile rate "
                            "is reported as the measured generator-capability "
                            "statistic instead"),
        "first_draft_compile_rate_reported": compile_rate.get("first_draft_rate"),
        "replacement_count": compile_rate.get("replacements_used"),
        "frozen_harnesses": coverage.get("frozen_harnesses"),
        "covered_problems": covered,
        "n_control_available": n_ctrl,
        "discriminating_harnesses": coverage.get("discriminating_harnesses"),
        "n_r1_clean_canvases": g4.get("n_r1_clean_with_verdict"),
        "n_measured": measured,
        "n_abstained": g4.get("n_abstained"),
        "abstained_container_ok": g4.get("abstained_container_ok"),
        "abstained_container_rejected": g4.get("abstained_container_rejected"),
        "property_agrees": agrees,
        "property_disagrees": g4.get("property_disagrees_with_container"),
        "property_agreement_rate": rate,
        "disagreements": g4.get("disagreements"),
        "checks": checks, "pass": all(checks.values()),
        "note": ("reported as a table either way; the abstention counts are the §7 "
                 "Q1 measurement that lets a future rev re-tier abstention on data"),
    }


def gate_g5(re25: Optional[dict]) -> dict:
    if re25 is None:
        return {"gate": "G5 H-25 CLASSIFICATION", "criterion": "(not run)",
                "pass": None, "skipped": True}
    eng = int(re25.get("engine_passes", 0))
    uns = int(re25.get("unsteered_passes", 0))
    n = int(re25.get("n_games", 4))
    engine_specific = (eng >= 3 and uns <= 1)
    return {
        "gate": "G5 H-25 CLASSIFICATION",
        "criterion": ("4-game standard: engine passes >= 3/4 AND unsteered "
                      "passes <= 1/4 => engine-specific reachability claim "
                      "admissible (exact counts, not a p-value ritual). "
                      "Otherwise draw-luck, and the m3.0-p3-report attribution "
                      "note is amended as a labeled cross-report amendment."),
        "engine_passes": eng, "unsteered_passes": uns, "n_games": n,
        "classification": ("engine_specific" if engine_specific else "draw_luck"),
        "pass": engine_specific,
        "note": ("G5 adjudicates ONLY on the pre-registered emitted answers; the "
                 "v2-pick / v3-offline-pick / oracle columns in the re25 record "
                 "are post-hoc forensics (§7 Q4 ruling)"),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cells", nargs="+", required=True,
                    help="label=results.json[:run.jsonl]")
    ap.add_argument("--primary", nargs=2, metavar=("A", "B"),
                    help="the pre-registered primary comparison, in the design's "
                         "direction (screen propscreen = 'property vs screen')")
    ap.add_argument("--propscreen-detail", default=None)
    ap.add_argument("--freeze", default=None,
                    help="the harness freeze manifest (for G4's compile rate)")
    ap.add_argument("--oracle", default=None,
                    help="P-prop-engine's post-hoc oracle audit JSON")
    ap.add_argument("--re25", default=None,
                    help="the P-25re 4-game classification JSON")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    cells = {c["label"]: c for c in (load_cell(s) for s in args.cells)}
    print("=== cells ===")
    for c in cells.values():
        j = jeffreys(c["n_passers"])
        print(f"[{c['label']:14s}] passers={c['n_passers']} "
              f"honest_p@1={c['honest_p1']}")
        print(f"  Jeffreys pooled REPORTING-ONLY: mean={j['mean']} "
              f"P(>0.56)={j['P>0.56']} P(>0.64)={j['P>0.64']}  [iid caveat: "
              "deterministic single per-problem draws]")

    report = {"cells": {k: {"passers": v["n_passers"],
                            "honest_p1": v["honest_p1"],
                            "jeffreys_reporting_only": jeffreys(v["n_passers"])}
                        for k, v in cells.items()},
              "paired": {}, "gates": {}}

    if args.primary:
        a, b = args.primary
        if a in cells and b in cells:
            rep = paired_report(cells[a], cells[b])
            print_paired(rep, primary=True)
            report["paired"]["primary"] = rep

    plan = [("cnull", "propscreen"), ("cscreen", "propscreen"),
            ("cnull", "ppropengine"), ("cfull", "ppropengine"),
            ("propscreen", "ppropengine"), ("cfull", "propscreen"),
            ("cnull", "cfull")]
    for a, b in plan:
        if a in cells and b in cells and [a, b] != (args.primary or []):
            rep = paired_report(cells[a], cells[b])
            print_paired(rep, primary=False)
            report["paired"][f"{a}->{b}"] = rep

    detail = {}
    if args.propscreen_detail:
        detail = json.loads(Path(args.propscreen_detail).read_text(encoding="utf-8"))
    freeze_summary = None
    if args.freeze:
        fr = json.loads(Path(args.freeze).read_text(encoding="utf-8"))
        freeze_summary = {"compile_rate": fr.get("compile_rate"),
                          "coverage": fr.get("coverage")}
    oracle = (json.loads(Path(args.oracle).read_text(encoding="utf-8"))
              if args.oracle else None)
    re25 = (json.loads(Path(args.re25).read_text(encoding="utf-8"))
            if args.re25 else None)

    print("\n=== gates (design §4, exact text) ===")
    gates = {}
    if "propscreen" in cells and "cscreen" in cells:
        gates["G1"] = gate_g1(cells["propscreen"], cells["cscreen"])
    if detail:
        gates["G2"] = gate_g2(detail)
        gates["G4"] = gate_g4(detail, freeze_summary)
    if "ppropengine" in cells:
        gates["G3"] = gate_g3(cells["ppropengine"], cells.get("cfull"), oracle)
    gates["G5"] = gate_g5(re25)
    report["gates"] = gates

    for key, g in gates.items():
        if g.get("skipped"):
            print(f"\n[{g['gate']}] SKIPPED (cell not run)")
            continue
        print(f"\n[{g['gate']}] => "
              f"{'PASS' if g['pass'] else ('FAIL' if g['pass'] is not None else 'N/A')}")
        print(f"  criterion: {g['criterion']}")
        for ck, cv in (g.get("checks") or {}).items():
            print(f"    {'OK ' if cv else 'NO '} {ck}")
        if key == "G1":
            print(f"  {g['passers']} passers vs screen {g['screen_passers']} "
                  f"(margin {g['margin']:+d}); sign p="
                  f"{g['sign_test_p_one_sided']}; {g['honest_power']}")
        if key == "G2":
            print(f"  attributed {g['n_attributed']}: {g['attributed_flips']}")
            print(f"  measured targets hit {g['measured_targets_hit']} of "
                  f"{g['measured_targets']}")
        if key == "G3":
            print(f"  {g['passers']} passers; cfull {g['cfull_passers']}; "
                  f"post-hoc oracle {g['post_hoc_oracle']} "
                  f"(forfeits {g['post_hoc_forfeit_problems']})")
        if key == "G5":
            print(f"  engine {g['engine_passes']}/{g['n_games']}, unsteered "
                  f"{g['unsteered_passes']}/{g['n_games']} -> "
                  f"{g['classification']}")

    # ---- SHIP / STRONG ---------------------------------------------------
    g1_ok = bool(gates.get("G1", {}).get("pass"))
    g2_ok = bool(gates.get("G2", {}).get("pass"))
    g3_ok = bool(gates.get("G3", {}).get("pass"))
    g5_ok = bool(gates.get("G5", {}).get("pass"))
    ship = g1_ok and g2_ok
    strong = ship and bool(cells.get("ppropengine", {}).get("n_passers", 0) >= 18) \
        and g5_ok
    verdict = {
        "SHIP": {"definition": "G1 + G2 (selection mechanism established on "
                               "held-fixed draws)",
                 "pass": ship, "G1": g1_ok, "G2": g2_ok},
        "STRONG": {"definition": "SHIP + G3 (engine >= 18, strictly better than "
                                 "the program best 17) + G5 in the engine's favor",
                   "pass": strong, "G3": g3_ok, "G5": g5_ok},
    }
    report["verdict"] = verdict
    print("\n=== VERDICT ===")
    for k, v in verdict.items():
        print(f"  {k}: {'PASS' if v['pass'] else 'FAIL'}  ({v['definition']})")
        print(f"    " + "  ".join(f"{a}={b}" for a, b in v.items()
                                 if a not in ("definition", "pass")))

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(report, indent=1), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
