#!/usr/bin/env python3
"""M3.1 PHASE 2 — P-prop-screen: rerank a BANKED ensemble with the property rule.

Design §3's primary cell. This is the cleanest mechanism comparison in the
program's history: the canvases are the M3.0 Phase-A C-null ensemble, already
generated and already container-scored, and the ONLY thing that changes is the
emission rule. Every flip is a SELECTION effect by construction — no draw
variance, no temperature cliff, no luck.

What it does, per problem:

  1. recover the ensemble from the banked run record + its FULL-ensemble export
     manifest + results JSON (the same join ``m3_p3_emissions.py`` does, so the
     two cells read per-canvas container statuses identically);
  2. recompute R1-clean offline with host rustc (the same predicate the in-loop
     engine applies), because R3 requires R1-clean canvases;
  3. run the FROZEN harness set against each R1-clean canvas (R3, 256 cases),
     logging every call to the D2 R3 schema;
  4. emit per the frozen v3 rule, then container-score the emitted set in ONE
     Rancher window (``--emit-only`` writes the export for that window);
  5. report the v2 screen / v3 property picks side by side, the oracle count,
     and the flip attribution (G2) — all before the 15 h engine cell launches.

The emitted canvases are exported and RE-SCORED rather than inheriting the
banked per-canvas status (§7 note 3, approved): a deterministic scorer on
identical text must agree, and the cross-check is recorded. A nonzero
disagreement means export/driver drift and is escalated, never absorbed.

    # 1. compute the v3 emissions (CPU + cargo, no GPU)
    temper/.venv/bin/python scripts/m3_31_propscreen.py --stage compute \\
        --run bench/multipl-e/runs/20261022T220753Z-c-null.jsonl \\
        --export bench/multipl-e/exports/20261023T130450Z-c-null-norepair \\
        --results bench/multipl-e/results/20261023T130739Z.json \\
        --out-dir bench/multipl-e/m3-31

    # 2. write the emitted-only export (one canvas per problem) for the window
    temper/.venv/bin/python scripts/m3_31_propscreen.py --stage export \\
        --out-dir bench/multipl-e/m3-31

    # 3. after the container window, fold the fresh results back in
    temper/.venv/bin/python scripts/m3_31_propscreen.py --stage score \\
        --out-dir bench/multipl-e/m3-31 --emitted-results <results.json>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from temper.multipl_e.bench import load_problems  # noqa: E402
from temper.steer.emission import (  # noqa: E402
    EMISSION_PROPERTY,
    EMISSION_SCREEN,
    property_tier,
    rank_by_rule,
    rank_property,
    rank_screen,
    select_by_rule,
    select_structural,
)
from temper.steer.property import (  # noqa: E402
    R3Log,
    R3Verdict,
    load_harness_freeze,
    make_pub_program,
    run_r3,
)
from temper.steer.verdict import (  # noqa: E402
    final_checkpoint_clean,
    ghost_close,
    load_verdict_config,
    r1_typecheck,
)
from temper.steer.view import SteerProblemView  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]


class Canvas:
    """One finished canvas of the ensemble, as the four emission rules see it.

    ``property_pass_rate`` is filled by R3 for R1-clean canvases and stays
    ``None`` for the rest — a canvas R1 rejected is not a candidate at all, so
    R3 never judged it (design §2.3's precondition).
    """

    __slots__ = ("index", "temperature", "seed", "mean_commit_conf", "raw_path",
                 "raw_text", "final_clean", "status", "excluded",
                 "r1_error_codes", "property_pass_rate", "property_statuses",
                 "property_measured", "property_passed", "property_wall_s",
                 "property_minimal_inputs")

    def __init__(self, row: dict):
        self.index = int(row["i"])
        self.temperature = float(row["temperature"])
        self.seed = int(row.get("seed") or 0)
        self.mean_commit_conf = row.get("mean_commit_conf")
        self.raw_path = row.get("raw_path")
        self.raw_text = ""
        self.final_clean = False
        self.status: Optional[str] = None
        self.excluded = False
        self.r1_error_codes: List[str] = []
        self.property_pass_rate: Optional[float] = None
        self.property_statuses: List[str] = []
        self.property_measured = 0
        self.property_passed = 0
        self.property_wall_s = 0.0
        self.property_minimal_inputs: List[str] = []


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


def read_text(raw_path: str) -> str:
    p = Path(raw_path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    return p.read_text(encoding="utf-8") if p.exists() else ""


def build_ensembles(run: dict, manifest: Optional[dict],
                    results: Optional[dict]) -> Dict[str, List[Canvas]]:
    """{problem: [Canvas]} with per-canvas container status + the R1 screen.

    ``manifest``/``results`` may be ``None`` for the P-prop-screen-twin over the
    engine's own banked canvases, where the container statuses come from the
    engine cell's own emission results instead (see ``attach_results``).
    """
    cfg = load_verdict_config()
    views = {td["name"]: SteerProblemView.from_task(td)
             for td in load_problems("humaneval-rs")}
    position_of: Dict[str, Dict[int, int]] = {}
    excluded_at: Dict[str, set] = {}
    for prob in (manifest or {}).get("problems", []):
        position_of[prob["name"]] = {int(sr["i"]): j
                                     for j, sr in enumerate(prob["source_rows"])}
    for ex in (manifest or {}).get("excluded", []):
        excluded_at.setdefault(ex.get("problem"), set()).add(int(ex.get("i")))
    statuses_of = {row["name"]: list(row.get("statuses_official") or [])
                   for row in (results or {}).get("per_problem", [])}

    ensembles: Dict[str, List[Canvas]] = {}
    for row in run["rows"]:
        name = row["problem"]
        c = Canvas(row)
        c.raw_text = read_text(c.raw_path) if c.raw_path else ""
        c.excluded = c.index in excluded_at.get(name, set())
        pos = position_of.get(name, {}).get(c.index)
        if pos is not None:
            sts = statuses_of.get(name, [])
            if pos < len(sts):
                c.status = sts[pos]
        view = views.get(name)
        if view is not None and c.raw_text:
            try:
                verdict = r1_typecheck(
                    ghost_close(view, c.raw_text),
                    hole_proximity=cfg.hole_proximity_lines)
                c.final_clean = final_checkpoint_clean(verdict)
                c.r1_error_codes = list(verdict.error_codes)[:6]
            except Exception as exc:      # a verifier crash is never clean
                c.final_clean = False
                c.r1_error_codes = [f"verifier:{type(exc).__name__}"]
        ensembles.setdefault(name, []).append(c)
    for name in ensembles:
        ensembles[name].sort(key=lambda c: c.index)
    return ensembles


def attach_results(ensembles: Dict[str, List[Canvas]], results: dict) -> None:
    """Stamp container statuses onto canvases from a per-canvas export manifest.

    Used for the TWIN over an engine cell's own banked canvases, where the
    ``<out>.canvases/<problem>__p<i>.txt`` files are the ensemble and the
    statuses come from a fresh all-canvas scoring window.
    """
    per = {row["name"]: row for row in results.get("per_problem", [])}
    for name, canvases in ensembles.items():
        row = per.get(name)
        if row is None:
            continue
        by_index = {int(k): v for k, v in (row.get("statuses_by_index") or {}).items()}
        for c in canvases:
            if c.index in by_index:
                c.status = by_index[c.index]


def load_canvases_dir(run_or_dir: Path, meta: dict) -> Dict[str, List[Canvas]]:
    """Rebuild an ensemble from an engine cell's banked canvas files (Q3).

    The engine seam writes every particle's final text to
    ``<out>.canvases/<problem>__p<i>.txt``; the ladder temperatures come from
    the run meta so each canvas carries the temperature the emission rules
    rank on.
    """
    d = Path(str(run_or_dir) + ".canvases")
    if not d.is_dir():
        raise SystemExit(f"{d}: no banked canvases — the cell did not bank them "
                         "(Q3 clarification requires it for M3.1 cells)")
    ladder = (meta.get("diversity_ladder") or meta.get("temperatures") or [])
    out: Dict[str, List[Canvas]] = {}
    for f in sorted(d.glob("*.txt")):
        stem = f.stem
        if "__p" not in stem:
            continue
        name, _, idx_s = stem.rpartition("__p")
        c = Canvas({"i": int(idx_s),
                    "temperature": (ladder[int(idx_s)] if int(idx_s) < len(ladder)
                                    else 0.0),
                    "seed": int(idx_s), "mean_commit_conf": None,
                    "raw_path": str(f)})
        c.raw_text = f.read_text(encoding="utf-8")
        out.setdefault(name, []).append(c)
    for name in out:
        out[name].sort(key=lambda c: c.index)
    return out


def r1_screen(ensembles: Dict[str, List[Canvas]]) -> None:
    cfg = load_verdict_config()
    views = {td["name"]: SteerProblemView.from_task(td)
             for td in load_problems("humaneval-rs")}
    for name, canvases in ensembles.items():
        view = views[name]
        for c in canvases:
            if c.final_clean or not c.raw_text:
                continue
            try:
                verdict = r1_typecheck(ghost_close(view, c.raw_text),
                                       hole_proximity=cfg.hole_proximity_lines)
                c.final_clean = final_checkpoint_clean(verdict)
                c.r1_error_codes = list(verdict.error_codes)[:6]
            except Exception as exc:
                c.final_clean = False
                c.r1_error_codes = [f"verifier:{type(exc).__name__}"]


def run_r3_over(ensembles: Dict[str, List[Canvas]], frozen: dict,
                log: R3Log, *, workdir: Path | str | None = None,
                problems: Optional[Sequence[str]] = None) -> dict:
    """R3 every R1-clean canvas with the frozen harness set (design §2.3)."""
    cfg = load_verdict_config()
    cases = int(cfg.raw["m3_1"]["r3"]["cases"])
    timeout_ms = int(cfg.raw["m3_1"]["r3"]["timeout_ms"])
    totals = {"calls": 0, "measured": 0, "passed": 0, "abstained": 0,
              "all_abstained_canvases": 0, "wall_s": 0.0, "canvases_r3ed": 0,
              "problems_zero_harness": []}
    for name in (problems or sorted(ensembles)):
        harnesses = frozen.get(name) or []
        if not harnesses:
            totals["problems_zero_harness"].append(name)
            continue
        views = {td["name"]: SteerProblemView.from_task(td)
                 for td in load_problems("humaneval-rs")}
        view = views[name]
        for c in ensembles.get(name, []):
            if not c.final_clean or not c.raw_text:
                continue
            try:
                program = make_pub_program(view, c.raw_text)
            except Exception as exc:
                log.write({"kind": "r3_error", "problem": name,
                           "canvas": f"p{c.index}",
                           "error": f"{type(exc).__name__}: {exc}"})
                continue
            verdict = run_r3(program.text, harnesses,
                             canvas_ref=f"p{c.index}", problem=name,
                             cases=cases, timeout_ms=timeout_ms,
                             workdir=workdir)
            c.property_pass_rate = verdict.pass_rate
            c.property_statuses = list(verdict.statuses)
            c.property_measured = verdict.n_measured
            c.property_passed = verdict.n_passed
            c.property_wall_s = verdict.wall_s
            c.property_minimal_inputs = list(verdict.minimal_inputs)
            log.record_verdict(verdict, cell="P-prop-screen")
            if verdict.all_abstained:
                log.write({"kind": "r3_abstain", "problem": name,
                           "canvas": f"p{c.index}",
                           "statuses": list(verdict.statuses),
                           "pass_rate": None})
                totals["all_abstained_canvases"] += 1
            totals["canvases_r3ed"] += 1
            totals["calls"] += verdict.n_calls
            totals["measured"] += verdict.n_measured
            totals["passed"] += verdict.n_passed
            totals["abstained"] += verdict.n_abstained
            totals["wall_s"] += verdict.wall_s
    totals["wall_s"] = round(totals["wall_s"], 2)
    totals["verdict_config_sha256"] = cfg.sha256
    totals["r3_cases"] = cases
    return totals


def emit_row(name: str, chosen: Optional[Canvas], rule: str) -> dict:
    """One problem's emitted outcome under one rule, in results-JSON shape."""
    if chosen is None:
        return {"name": name, "statuses_official": [], "n_scored": 0,
                "n_excluded": 1, "emitted": None, "rule": rule}
    if chosen.excluded or chosen.status is None:
        return {"name": name, "statuses_official": [], "n_scored": 0,
                "n_excluded": 1, "emitted": chosen.index, "rule": rule,
                "emitted_temp": chosen.temperature,
                "emitted_conf": chosen.mean_commit_conf,
                "emitted_clean": chosen.final_clean,
                "emitted_r1_codes": chosen.r1_error_codes,
                "emitted_property_pass_rate": chosen.property_pass_rate,
                "emitted_property_statuses": chosen.property_statuses}
    return {"name": name, "statuses_official": [chosen.status], "n_scored": 1,
            "n_excluded": 0, "emitted": chosen.index, "rule": rule,
            "emitted_temp": chosen.temperature,
            "emitted_conf": chosen.mean_commit_conf,
            "emitted_clean": chosen.final_clean,
            "emitted_r1_codes": chosen.r1_error_codes,
            "emitted_property_pass_rate": chosen.property_pass_rate,
            "emitted_property_statuses": chosen.property_statuses}


def results_payload(label: str, rows: List[dict], source: dict) -> dict:
    n_ok = sum(1 for r in rows if "OK" in r["statuses_official"])
    n = len(rows)
    return {
        "type": "multipl-e-results",
        "created_utc": source.get("created_utc"),
        "input_dir": source.get("input_dir"),
        "versions": ["official"],
        "emission_cell": label,
        "note": ("EMISSION-RULE CELL: one emitted canvas per problem, so each "
                 "problem contributes exactly one scored row (or one honest "
                 "exclusion). This is design 4.1's per-problem binary."),
        "per_problem": rows,
        "aggregate": {
            "official": {"pass@1": round(n_ok / n, 4) if n else None},
            "honest-official": {"pass@1": round(n_ok / n, 4) if n else None,
                                "n_excluded_rows": sum(r["n_excluded"]
                                                       for r in rows)},
        },
    }


# --------------------------------------------------------------------------
# G2 attribution + G4 correlation table
# --------------------------------------------------------------------------


def attribution_row(name: str, canvases: List[Canvas],
                    screen_pick: Optional[Canvas],
                    prop_pick: Optional[Canvas]) -> dict:
    """One problem's G2 attribution record (design §4 G2, clauses a-d)."""
    ok_canvases = [c for c in canvases if c.status == "OK"]
    screen_ok = (screen_pick is not None and screen_pick.status == "OK")
    prop_ok = (prop_pick is not None and prop_pick.status == "OK")
    # G2(a): "both rules ranked the same 8 canvases" — the same POOL, not the
    # same ORDER. The order is what the rules are allowed to disagree about (it
    # is the whole point); a different pool would mean the rules were ranking
    # different ensembles and no flip would be comparable.
    same_pool = (sorted(c.index for c in rank_screen(canvases))
                 == sorted(c.index for c in
                           rank_by_rule(canvases, EMISSION_PROPERTY)))
    flip = (screen_pick is not None and prop_pick is not None
            and screen_pick.index != prop_pick.index)
    up = (not screen_ok) and prop_ok
    down = screen_ok and not prop_ok
    return {
        "problem": name,
        "n_canvases": len(canvases),
        "n_r1_clean": sum(1 for c in canvases if c.final_clean),
        "n_container_ok": len(ok_canvases),
        "ok_indices": [c.index for c in ok_canvases],
        "any_ok": bool(ok_canvases),
        "screen_pick": screen_pick.index if screen_pick else None,
        "screen_temp": screen_pick.temperature if screen_pick else None,
        "screen_status": screen_pick.status if screen_pick else None,
        "screen_conf": screen_pick.mean_commit_conf if screen_pick else None,
        "screen_property_pass_rate": (screen_pick.property_pass_rate
                                      if screen_pick else None),
        "prop_pick": prop_pick.index if prop_pick else None,
        "prop_temp": prop_pick.temperature if prop_pick else None,
        "prop_status": prop_pick.status if prop_pick else None,
        "prop_conf": prop_pick.mean_commit_conf if prop_pick else None,
        "prop_pass_rate": prop_pick.property_pass_rate if prop_pick else None,
        "prop_statuses": prop_pick.property_statuses if prop_pick else [],
        "flip": flip,
        "flip_direction": ("up" if up else ("down" if down else
                                            ("sideways" if flip else ""))),
        # G2's four clauses, evaluated literally
        "g2_a_same_pool": same_pool,
        "g2_b_screen_emitted_rejected": not screen_ok,
        "g2_c_ok_canvas_existed": bool(ok_canvases),
        "g2_d_property_emitted_ok": prop_ok,
        "g2_attributed": bool(up and same_pool and (not screen_ok)
                              and ok_canvases and prop_ok),
        "canvas_statuses": [c.status for c in canvases],
        "canvas_temps": [c.temperature for c in canvases],
        "canvas_clean": [c.final_clean for c in canvases],
        "canvas_pass_rates": [c.property_pass_rate for c in canvases],
        "canvas_property_statuses": [c.property_statuses for c in canvases],
    }


def g4_table(rows: List[dict]) -> dict:
    """G4's property/correlation table (design §4 G4, §0.5).

    On R1-clean canvases where the property verdict and the confidence ranking
    DISAGREE, which one agrees with the container? Measured per canvas, so the
    table is a real correlation and not a winner-take-all tally.
    """
    considered = []          # R1-clean canvases with a container verdict
    for r in rows:
        clean = r["canvas_clean"]
        for i, status in enumerate(r["canvas_statuses"]):
            if not clean[i]:
                continue
            if status is None:
                continue
            rates = r["canvas_pass_rates"]
            considered.append({
                "problem": r["problem"],
                "index": i,
                "temperature": r["canvas_temps"][i],
                "status": status,
                "container_ok": status == "OK",
                "property_pass_rate": rates[i],
                "property_tier": property_tier(_TierState(rates[i])),
                "property_statuses": r["canvas_property_statuses"][i],
            })
    measured = [c for c in considered if c["property_pass_rate"] is not None]
    # "the properties are right" = the property verdict's own preference among
    # an OK and a REJECTED canvas agrees with the container
    prop_right = sum(1 for c in measured
                     if (c["property_pass_rate"] > 0) == c["container_ok"])
    prop_wrong = len(measured) - prop_right
    abstained = [c for c in considered if c["property_pass_rate"] is None]
    # the same question for the tier-1 (all-abstain) canvases: are they more
    # often OK or rejected? §7 Q1 requires this to be measured before any rev2
    # re-tiering can even be argued
    abstain_ok = sum(1 for c in abstained if c["container_ok"])
    # disagreements only: a canvas the property rung judged PASS but which the
    # container rejected, or vice versa
    disagreements = [c for c in measured
                     if (c["property_pass_rate"] > 0) != c["container_ok"]]
    return {
        "n_r1_clean_with_verdict": len(considered),
        "n_measured": len(measured),
        "n_abstained": len(abstained),
        "property_agrees_with_container": prop_right,
        "property_disagrees_with_container": prop_wrong,
        "property_agreement_rate": (round(prop_right / len(measured), 4)
                                    if measured else None),
        "disagreements": disagreements,
        "abstained_canvases": len(abstained),
        "abstained_container_ok": abstain_ok,
        "abstained_container_rejected": len(abstained) - abstain_ok,
        "note": ("§7 Q1: the all-abstain counts are reported so a future rev can "
                 "argue re-tiering them as negative evidence from data; until "
                 "then §2.3's abstention semantics stand."),
        "canvases": considered,
    }


class _TierState:
    """Tiny adapter so :func:`temperature.steer.emission.property_tier` can be
    reused on a plain pass-rate value."""
    __slots__ = ("property_pass_rate",)

    def __init__(self, rate):
        self.property_pass_rate = rate


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True,
                    choices=("compute", "export", "score"))
    ap.add_argument("--run", default=None,
                    help="the BANKED run record whose ensemble is reranked")
    ap.add_argument("--canvases-from", default=None,
                    help="an engine cell's run record: rerank its banked "
                         "<out>.canvases/ ensemble (the twin)")
    ap.add_argument("--export", default=None,
                    help="the FULL-ensemble export dir (manifest.json)")
    ap.add_argument("--results", default=None,
                    help="the FULL-ensemble container results JSON")
    ap.add_argument("--label", default="P-prop-screen")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--out-prefix", default="p-prop-screen")
    ap.add_argument("--emitted-results", default=None,
                    help="--stage score: the container results of the emitted "
                         "export")
    ap.add_argument("--freeze", default=None, help="harness freeze manifest")
    ap.add_argument("--r3-workdir", default=None)
    ap.add_argument("--problems", default=None,
                    help="comma-separated problem names to restrict to")
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_verdict_config()

    if args.stage == "score":
        return _stage_score(args, out_dir, cfg)

    if args.stage == "export":
        return _stage_export(args, out_dir)

    # ---- compute ---------------------------------------------------------
    if not args.run and not args.canvases_from:
        raise SystemExit("--stage compute needs --run or --canvases-from")
    frozen = load_harness_freeze(args.freeze)
    n_frozen = sum(len(v) for v in frozen.values())
    print(f"frozen harnesses: {n_frozen} over {len(frozen)} problems")

    if args.canvases_from:
        src = Path(args.canvases_from)
        if not src.is_absolute():
            src = REPO_ROOT / src
        meta2, _ = _read_meta(src)
        ensembles = load_canvases_dir(src, meta2)
        source_desc = {"canvases_from": str(src)}
        base_results = None
    else:
        run = load_run(Path(args.run))
        manifest = (json.loads((REPO_ROOT / args.export / "manifest.json").read_text())
                    if args.export else None)
        results = (json.loads((REPO_ROOT / args.results).read_text())
                   if args.results else None)
        ensembles = build_ensembles(run, manifest, results)
        source_desc = {"run": str(args.run), "export": str(args.export),
                       "results": str(args.results)}
        base_results = results

    if args.problems:
        keep = {s.strip() for s in args.problems.split(",") if s.strip()}
        ensembles = {k: v for k, v in ensembles.items() if k in keep}

    r1_screen(ensembles)
    r3_log = R3Log(out_dir / f"{args.out_prefix}-r3.jsonl")
    print(f"running R3 over {sum(len(v) for v in ensembles.values())} canvases "
          f"({sum(1 for v in ensembles.values() for c in v if c.final_clean)} "
          "R1-clean)...")
    totals = run_r3_over(ensembles, frozen, r3_log, workdir=args.r3_workdir)
    print(f"  R3: {totals['calls']} calls, {totals['measured']} measured, "
          f"{totals['passed']} passed, {totals['abstained']} abstained, "
          f"wall {totals['wall_s']}s")
    if totals["problems_zero_harness"]:
        print(f"  !! zero-harness problems: {totals['problems_zero_harness']}")

    # ---- the four rule tables -------------------------------------------
    rule_rows = {rule: [] for rule in
                 ("structural", EMISSION_SCREEN, EMISSION_PROPERTY)}
    attribution = []
    for name in sorted(ensembles):
        ens = ensembles[name]
        if not ens:
            continue
        structural = select_structural(ens)
        screen_pick = select_by_rule(ens, EMISSION_SCREEN)
        prop_pick = select_by_rule(ens, EMISSION_PROPERTY)
        rule_rows["structural"].append(emit_row(name, structural, "structural"))
        rule_rows[EMISSION_SCREEN].append(
            emit_row(name, screen_pick, EMISSION_SCREEN))
        rule_rows[EMISSION_PROPERTY].append(
            emit_row(name, prop_pick, EMISSION_PROPERTY))
        attribution.append(attribution_row(name, ens, screen_pick, prop_pick))

    n_of = lambda rows: sum(1 for r in rows if "OK" in r["statuses_official"])
    print(f"\n=== rule tables over {len(attribution)} problems ===")
    print(f"  structural (C-null semantics) : {n_of(rule_rows['structural'])}"
          f"/{len(attribution)}")
    print(f"  screen      (v2)              : {n_of(rule_rows[EMISSION_SCREEN])}"
          f"/{len(attribution)}")
    print(f"  property    (v3)              : {n_of(rule_rows[EMISSION_PROPERTY])}"
          f"/{len(attribution)}")

    # ---- oracle + G1/G2 --------------------------------------------------
    oracle = sum(1 for r in attribution if r["any_ok"])
    g2 = [r for r in attribution if r["g2_attributed"]]
    flips = [r for r in attribution if r["flip"]]
    ups = [r for r in flips if r["flip_direction"] == "up"]
    downs = [r for r in flips if r["flip_direction"] == "down"]
    prop_passes = n_of(rule_rows[EMISSION_PROPERTY])
    screen_passes = n_of(rule_rows[EMISSION_SCREEN])
    g1 = {"gate": "G1 RECOVER", "variant": args.label,
          "criterion": "P-prop-screen >= 16/25 on the C-null ensemble",
          "passers": prop_passes, "n_problems": len(attribution),
          "oracle": oracle, "pass": prop_passes >= 16,
          "margin_vs_v2_screen": prop_passes - screen_passes,
          "power": ("at n=25 with ~2-4 discordant pairs conventional alpha=0.05 "
                    "is unreachable; the pre-registered decision quantity is the "
                    "+2 net-flip margin WITH attribution (§4 G1)")}
    g2rec = {"gate": "G2 ATTRIBUTED-SELECTION",
             "criterion": ("every flip counts only if (a) both rules ranked the "
                           "same canvases, (b) the confidence rule emitted a "
                           "container-rejected canvas, (c) an OK canvas existed, "
                           "(d) the property rule emitted one of them"),
             "attributed_flips": [r["problem"] for r in g2],
             "n_attributed": len(g2),
             "flips_up": [r["problem"] for r in ups],
             "flips_down": [r["problem"] for r in downs],
             "unattributed_ups": [r["problem"] for r in ups
                                  if not r["g2_attributed"]],
             "pass": len(g2) >= 2 and not downs,
             "margin_net_flips": len(ups) - len(downs)}
    g4 = g4_table(attribution)

    print(f"\noracle (any container-OK canvas in the ensemble): {oracle}"
          f"/{len(attribution)}")
    print(f"flips v2->v3: up {len(ups)} down {len(downs)}; attributed {len(g2)}")
    print(f"  flips UP  : {[r['problem'] for r in ups]}")
    print(f"  flips DOWN: {[r['problem'] for r in downs]}")
    print(f"G4 property agreement on R1-clean canvases: "
          f"{g4['property_agrees_with_container']}/{g4['n_measured']} "
          f"(rate {g4['property_agreement_rate']}); abstained "
          f"{g4['abstained_canvases']} of which OK {g4['abstained_container_ok']}")
    print(f"\nG1: {prop_passes}/{len(attribution)} vs 16 -> "
          f"{'PASS' if g1['pass'] else 'FAIL'}")
    print(f"G2: {len(g2)} attributed flips -> "
          f"{'PASS' if g2rec['pass'] else 'FAIL'}")

    # ---- persist ---------------------------------------------------------
    detail_path = out_dir / f"{args.out_prefix}-detail.json"
    detail_path.write_text(json.dumps({
        "cell": args.label,
        "source": source_desc,
        "verdict_config_sha256": cfg.sha256,
        "verdict_config_version": cfg.version,
        "emission_rule": cfg.emission,
        "r3_totals": totals,
        "rules": {k: {"passers": n_of(v), "rows": v}
                  for k, v in rule_rows.items()},
        "oracle": oracle,
        "attribution": attribution,
        "gates": {"G1": g1, "G2": g2rec, "G4": g4},
    }, indent=1), encoding="utf-8")
    for rule, rows in rule_rows.items():
        slug = {"structural": "structural",
                EMISSION_SCREEN: "screen",
                EMISSION_PROPERTY: "property"}[rule]
        payload = results_payload(f"{args.label}-{slug}", rows,
                                  base_results or {"created_utc": None})
        (out_dir / f"{args.out_prefix}-{slug}-results.json").write_text(
            json.dumps(payload, indent=1), encoding="utf-8")
    print(f"\nwrote {detail_path}")
    print(f"wrote {out_dir}/{args.out_prefix}-{{structural,screen,property}}-results.json")
    return 0


def _read_meta(path: Path) -> Tuple[dict, List[dict]]:
    meta, rows = None, []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        o = json.loads(line)
        if o.get("type") == "run":
            meta = o
        elif o.get("type") == "candidate":
            rows.append(o)
    if meta is None:
        raise SystemExit(f"{path}: no run meta line")
    return meta, rows


def _stage_export(args, out_dir: Path) -> int:
    """Write the emitted-only export for the container window.

    One completion per problem — the v3 emission — so the window scores exactly
    the canvases the rule picked. The export instrument is REUSED unchanged:
    this stage writes a filtered run record holding only the emitted rows and
    hands it to ``export_runs``, so extraction, repair, body-fragment reduction
    and stop-residue handling are byte-for-byte the pipeline that produced the
    banked ensemible's completions. Reimplementing that reduction here would
    make the rescore a different measurement from the bank it is cross-checked
    against.
    """
    from temper.multipl_e.export import export_runs

    detail = json.loads((out_dir / f"{args.out_prefix}-detail.json")
                        .read_text(encoding="utf-8"))
    rows = detail["rules"][detail["emission_rule"]]["rows"]
    source = detail["source"]
    if source.get("run"):
        loaded = load_run(Path(source["run"]))
        meta, banked_rows = loaded["meta"], loaded["rows"]
    else:
        meta, banked_rows = _synthetic_meta(source), []

    by_key = {(r["problem"], int(r["i"])): r for r in banked_rows}
    filtered = [{"type": "run"}]
    filtered[0].update(meta)
    filtered[0]["cell"] = f"{detail['cell']}-emitted"
    filtered[0]["note"] = ("filtered record: exactly the v3-emitted canvas per "
                           "problem, so the container scores the emission")
    for row in rows:
        name = row["name"]
        emitted = row.get("emitted")
        if emitted is None:
            continue
        banked = by_key.get((name, int(emitted)))
        if banked is None:
            continue
        filtered.append(dict(banked, problem=name, i=int(emitted)))

    if not meta.get("problem_names"):
        meta["problem_names"] = [r["name"] for r in rows]
    filt_path = out_dir / f"{args.out_prefix}-emitted-run.jsonl"
    filt_path.write_text("\n".join(json.dumps(r) for r in filtered) + "\n",
                         encoding="utf-8")
    exp = out_dir / f"{args.out_prefix}-emitted-export"
    man = export_runs([filt_path], exp, benchmark="humaneval-rs",
                      repair=False)
    n = sum(p["completions"] for p in man["problems"])
    print(f"wrote {exp}: {n} emitted completions over "
          f"{len(man['problems'])} problems, {len(man['excluded'])} exclusions")
    for ex in man["excluded"]:
        print(f"  EXCLUDED {ex.get('problem')}: {ex.get('reason')}")
    (out_dir / f"{args.out_prefix}-emitted-manifest.json").write_text(
        json.dumps(man, indent=1), encoding="utf-8")
    return 0


def _synthetic_meta(source: dict) -> dict:
    return {"type": "run", "benchmark": "humaneval-rs",
            "cell": source.get("canvases_from", ""),
            "problem_names": []}


def _canvas_text(source: dict, name: str, index: int) -> str:
    if source.get("run"):
        run = load_run(Path(source["run"]))
        for row in run["rows"]:
            if row["problem"] == name and int(row["i"]) == index:
                return read_text(row.get("raw_path") or "")
        return ""
    if source.get("canvases_from"):
        f = Path(str(source["canvases_from"]) + ".canvases") / f"{name}__p{index}.txt"
        return f.read_text(encoding="utf-8") if f.exists() else ""
    return ""


def _extract(text: str) -> str:
    from temper.model.extract import extract_rust
    if not text:
        return ""
    ex = extract_rust(text)
    return ex.code if ex.ok else ""


def _stage_score(args, out_dir: Path, cfg) -> int:
    """Fold the emitted-export's fresh container results back into the cell.

    The emissions are container-CONFIRMED here, not inherited: the cross-check
    against the banked per-canvas statuses must agree exactly (deterministic
    scorer on identical text), and any disagreement is reported loudly rather
    than absorbed (§7 note 3).
    """
    if not args.emitted_results:
        raise SystemExit("--stage score needs --emitted-results")
    res = json.loads(Path(args.emitted_results).read_text(encoding="utf-8"))
    detail_path = out_dir / f"{args.out_prefix}-detail.json"
    detail = json.loads(detail_path.read_text(encoding="utf-8"))
    rows = detail["rules"][detail["emission_rule"]]["rows"]
    fresh = {r["name"]: list(r.get("statuses_official") or [])
             for r in res["per_problem"]}
    disagreements = []
    n_ok = 0
    for row in rows:
        name = row["name"]
        sts = fresh.get(name, [])
        row["rescored_statuses_official"] = sts
        row["rescored_ok"] = bool(sts) and sts[0] == "OK"
        if row["rescored_ok"]:
            n_ok += 1
        banked = row["statuses_official"]
        if sts and banked and sts[0].split(":")[0] != banked[0].split(":")[0]:
            disagreements.append({"problem": name,
                                  "banked": banked[0], "rescored": sts[0]})
    detail["rescored"] = {
        "results_path": str(args.emitted_results),
        "passers": n_ok,
        "n_problems": len(rows),
        "pass@1": round(n_ok / len(rows), 4) if rows else None,
        "disagreements_vs_banked": disagreements,
        "n_disagreements": len(disagreements),
    }
    detail["gates"]["G1"]["rescored_passers"] = n_ok
    detail["gates"]["G1"]["rescored_pass"] = n_ok >= 16
    detail_path.write_text(json.dumps(detail, indent=1), encoding="utf-8")
    print(f"rescored emissions: {n_ok}/{len(rows)} OK "
          f"(pass@1 {detail['rescored']['pass@1']})")
    print(f"cross-check vs banked per-canvas statuses: "
          f"{len(disagreements)} disagreements")
    for d in disagreements:
        print(f"  DISAGREE {d['problem']}: banked {d['banked']} vs "
              f"rescored {d['rescored']}")
    if disagreements:
        print("\nESCALATE: a deterministic scorer must agree on identical text; "
              "a disagreement means export/driver drift. Do not proceed to the "
              "engine cell until this is explained.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
