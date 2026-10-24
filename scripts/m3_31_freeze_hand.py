#!/usr/bin/env python3
"""Freeze the HAND-AUTHORED harness set with BOTH controls (§7 step 2, branch C).

Reads the committed harness sources (``harnesses/m3-31/cases``), validates each
one with the two controls, and writes the frozen manifest the selection cells
consume. Ground truth for the positive control is the BANKED C-null container
verdicts — container evidence, not property evidence — so using it to validate an
instrument cannot contaminate the G4b correlation table.

    temper/.venv/bin/python scripts/m3_31_freeze_hand.py \\
        --cases harnesses/m3-31/cases \\
        --run bench/multipl-e/runs/20261022T220753Z-c-null.jsonl \\
        --export bench/multipl-e/exports/20261023T130450Z-c-null-norepair \\
        --results bench/multipl-e/results/20261023T130739Z.json \\
        --manifest harness_freeze.json --json bench/multipl-e/m3-31/freeze-report.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from temper.multipl_e.bench import load_benchmark, load_problems  # noqa: E402
from temper.steer.property import (  # noqa: E402
    HARNESS_FREEZE_SCHEMA,
    Harness,
    compute_harness_freeze_sha256,
    strategy_hash_of,
)
from temper.steer.verdict import load_verdict_config  # noqa: E402
from temper.steer.view import SteerProblemView  # noqa: E402
from scripts.m3_31_harness import validate_two_controls  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
PROBLEM_NUMBERS = tuple(range(22)) + (23, 24, 25)


def _load_banked(run_path: Path, export_dir: Path, results_path: Path):
    """{problem: [(index, text, container_status)]} from the banked ensemble."""
    meta, rows = None, []
    for line in run_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        o = json.loads(line)
        if o.get("type") == "run":
            meta = o
        elif o.get("type") == "candidate":
            rows.append(o)
    manifest = json.loads((export_dir / "manifest.json").read_text(encoding="utf-8"))
    results = json.loads(results_path.read_text(encoding="utf-8"))
    position_of = {p["name"]: {int(sr["i"]): j
                               for j, sr in enumerate(p["source_rows"])}
                   for p in manifest.get("problems", [])}
    statuses_of = {r["name"]: list(r.get("statuses_official") or [])
                   for r in results["per_problem"]}
    out: Dict[str, List[dict]] = {}
    for row in rows:
        name = row["problem"]
        idx = int(row["i"])
        rp = row.get("raw_path")
        text = ""
        if rp:
            p = Path(rp)
            if not p.is_absolute():
                p = REPO_ROOT / p
            text = p.read_text(encoding="utf-8") if p.exists() else ""
        pos = position_of.get(name, {}).get(idx)
        status = None
        sts = statuses_of.get(name, [])
        if pos is not None and pos < len(sts):
            status = sts[pos]
        out.setdefault(name, []).append({"index": idx, "text": text,
                                         "status": status})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--export", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--extra-bank-file", default=None,
                    help="a JSON list of {run, export, results} dicts to add as "
                         "positive-control ground-truth banks")
    ap.add_argument("--extra-bank", action="append", default=[],
                    metavar="RUN:EXPORT:RESULTS",
                    help="additional BANKED diffusion ensembles whose container-OK "
                         "canvases may serve as positive-control ground truth. "
                         "Without this, 9 of 25 problems have NO container-OK "
                         "canvas anywhere in the C-null ensemble and their "
                         "positive control cannot run at all.")
    ap.add_argument("--manifest", default=str(REPO_ROOT / "harness_freeze.json"))
    ap.add_argument("--json", default=None)
    ap.add_argument("--workdir", default=None)
    args = ap.parse_args(argv)

    cases = Path(args.cases)
    banked = _load_banked(Path(args.run), Path(args.export), Path(args.results))
    extra_specs = list(args.extra_bank)
    if args.extra_bank_file:
        for d in json.loads(Path(args.extra_bank_file).read_text()):
            extra_specs.append(f"{d['run']}:{d['export']}:{d['results']}")
    for spec in extra_specs:
        r, e, res = spec.split(":")
        extra = _load_banked(Path(r), Path(e), Path(res))
        for name, cs in extra.items():
            banked.setdefault(name, [])
            known = {c["index"] for c in banked[name]}
            # re-key the extra ensemble's indices so they cannot collide with
            # the primary ensemble's; only the STATUS and TEXT matter to a control
            for c in cs:
                if c["index"] in known:
                    c = dict(c, index=c["index"] + 100)
                banked[name].append(c)
    views = {td["name"]: SteerProblemView.from_task(td)
             for td in load_problems("humaneval-rs",
                                     problem_numbers=list(PROBLEM_NUMBERS))}
    cfg = load_verdict_config()

    frozen: Dict[str, List[Harness]] = {}
    report_rows: List[dict] = []
    rewrite_counts: Dict[str, int] = {}
    rewrite_log: List[dict] = []
    for name in [views[n].name for n in views]:
        canvases = banked.get(name, [])
        ok = next((c for c in canvases if c["status"] == "OK" and c["text"]), None)
        # A container-REJECTED canvas: any non-OK verdict, INCLUDING an export
        # exclusion (status None), which is a scored failure under honest
        # counting. Requiring only `status not in (None,"OK")` skipped exactly
        # those and left most problems with no negative control at all.
        # The negative control must be a canvas the harness can actually JUDGE:
        # a SyntaxError canvas cannot even link, so `fail` is unreachable and it
        # validates nothing about the property's discriminating power. Prefer a
        # canvas whose verdict shows it COMPILED but was rejected (Exception /
        # Wrong / a container failure that is not a syntax error), and fall back
        # to any non-OK text only if no compiled-but-rejected one exists.
        rejects = [c for c in canvases
                   if c["text"] and c["status"] not in (None, "OK")
                   and "SyntaxError" not in str(c["status"])]
        # NO fallback to a SyntaxError canvas: the harness cannot link against
        # one, so running the control there produces a `None` runtime and would
        # read as a harness defect rather than as "this control cannot be run".
        # An absent control is reported as unavailable (None), not as False.
        kept: List[Harness] = []
        for f in sorted(cases.glob(f"{name}__h*.rs")):
            slot = int(f.stem.rsplit("__h", 1)[1])
            src = f.read_text(encoding="utf-8")
            res = validate_two_controls(
                views[name], src,
                ok_canvas=(ok["text"] if ok else None),
                reject_canvases=[c["text"] for c in rejects],
                workdir=args.workdir, cases=64, timeout_ms=1000)
            row = {"problem": name, "slot": slot, **res,
                   "validated_ok_canvas_index": ok["index"] if ok else None,
                   "validated_ok_canvas_status": ok["status"] if ok else None,
                   "validated_reject_canvas_count": len(rejects)}
            report_rows.append(row)
            if res["validated"]:
                kept.append(Harness(problem=name, slot=slot, seed=slot,
                                    source=src,
                                    strategy_hash=strategy_hash_of(src),
                                    source_path=str(f.resolve().relative_to(REPO_ROOT))))
            else:
                bad_checks = [k for k, v in res["checks"].items()
                              if v is False]
                print(f"  NOT VALIDATED {name} h{slot}: {bad_checks}")
        frozen[name] = kept
        from scripts.m3_31_handwritten import REWRITES
        n_rew = max((REWRITES.get((name, r["slot"]), 0) for r in report_rows
                     if r["problem"] == name), default=0)
        rewrite_counts[name] = 1 + n_rew
        for (pname, slot), reasons in REWRITES.items():
            if pname == name:
                for i, why in enumerate(reasons):
                    rewrite_log.append({"problem": pname, "slot": slot,
                                        "rewrite": i + 1, "reason": why})
        print(f"{name:44s} validated {len(kept)}/2  rewrites={n_rew}")

    covered = sum(1 for v in frozen.values() if v)
    total = sum(len(v) for v in frozen.values())
    # G4a AS AMENDED (§7 A8, operator ruling 2026-10-24): >=19/21
    # CONTROL-AVAILABLE problems, i.e. problems for which a container-OK
    # diffusion canvas exists ANYWHERE in the program's banks. The original
    # >=23/25 was an invented threshold and is structurally unreachable: 4
    # problems {0, 18, 19, 21} have no container-OK canvas in any diffusion
    # cell — exactly P1's unreachable-at-all-temps set, found independently by a
    # different instrument. G1's own threshold stays ABSOLUTE (>=16/25).
    control_available = sorted(
        n for n in frozen
        if any(r["checks"]["passes_a_container_ok_canvas"] is not None
               for r in report_rows if r["problem"] == n))
    n_ctrl = len(control_available)
    g4a = n_ctrl > 0 and covered >= 19 and covered >= n_ctrl - 2
    payload = {
        "schema": HARNESS_FREEZE_SCHEMA,
        "version": "m3.1-c-phase-v1",
        "frozen_utc": "2026-10-24",
        "note": (
            "M3.1 C-phase freeze: HAND-AUTHORED harnesses (design §7 A6). "
            "Authoring discipline: every harness was written from the problem's "
            "docstring and signature ONLY, before any canvas for that problem was "
            "inspected, and the whole set was committed before the first R3 call "
            "— so the freeze ORDER proves it was not fitted to observed canvases "
            "or to the watchlist problems {1, 9, 24, 25}. Validation is the TWO-"
            "CONTROL protocol: abort on the todo!() stub (negative) and separate "
            "a banked container-OK canvas from a banked container-rejected one "
            "(positive)."),
        "harness_origin": "hand-authored (not model-generated)",
        "generator_status_reference": "bench/multipl-e/m3-31/q-gen-negative.json",
        "verdict_config_sha256": cfg.sha256,
        "controls": {
            "negative": "compiles against the todo!() signature stub and does not PASS it",
            "positive": ("PASSES a banked container-OK canvas; ground truth is the "
                         "container's verdict, so this cannot contaminate the G4b "
                         "correlation table"),
            "reject_side_amendment": (
                "§7 A7 (operator ruling 2026-10-24): the reject-side clause is NO LONGER "
                "GATED. Measured reason: the dominant not-validated class was correct-by-"
                "docstring harnesses passing near-miss reject canvases (HE_0/3/4/5/8/13), "
                "which is the intrinsic limitation of property evidence that G4b exists to "
                "measure — demanding a zero miss rate at freeze would only converge by "
                "fitting properties to the reject pool, the exact canvas-knowledge leak the "
                "docstring-only discipline forbids. The reject side is now REPORTED as a "
                "per-harness miss rate over the problem's compiled-but-rejected canvases, "
                "published here and carried into the G4b narrative."),
        },
        "positive_control_pool": {
            "definition": (
                "GROUND TRUTH for the positive control = a canvas whose CONTAINER verdict is "
                "OK, drawn from the union of every BANKED diffusion (non-Qwen) ensemble named "
                "to this run. It is container evidence, never property evidence, so it cannot "
                "contaminate the G4b correlation table."),
            "primary_pool": "the C-null ensemble (--run/--export/--results)",
            "extra_pools": [d.get("export") for d in
                            ([json.loads(Path(args.extra_bank_file).read_text())]
                             if args.extra_bank_file else [{}])[0]] or [],
            "why_union": (
                "9 of 25 problems have NO container-OK canvas in the C-null ensemble alone, so "
                "a C-null-only pool would cap coverage at 16/25 before any harness is judged. "
                "The union is the largest legitimate ground truth the program owns."),
            "per_problem_ok_in_union": {
                n: sum(1 for c in banked.get(n, []) if c["status"] == "OK" and c["text"])
                for n in sorted(views)},
        },
        "coverage": {
            "problems": len(frozen),
            "problems_with_ge1_harness": covered,
            "frozen_harnesses": total,
            "G4a_supply_gate": (
                "§7 A8 (operator ruling 2026-10-24): >=19/21 CONTROL-AVAILABLE problems "
                "carry >=1 harness. Original >=23/25 was an invented threshold and is "
                "structurally unreachable — 4 problems {0,18,19,21} have NO container-OK "
                "canvas in any diffusion cell (independently P1's unreachable-at-all-temps "
                "set). G1's own threshold stays ABSOLUTE: >=16/25 by container verdict."),
            "control_available_problems": control_available,
            "n_control_available": n_ctrl,
            "G4a_pass": g4a,
            # the four measured forfeit problems from M3.0-P3 (§7 A5)
            "watchlist": {f"HumanEval_{k}": len(frozen.get(
                next((x for x in views if x.startswith(f"HumanEval_{k}_")), ""), []))
                for k in (1, 9, 24, 25)},
            # how many FROZEN harnesses were shown to DISCRIMINATE (pass a
            # container-OK canvas and fail a compiled-but-rejected one) rather
            # than merely compile. A harness validated only against the stub has
            # not been shown to separate good from wrong.
            "discriminating_harnesses": sum(
                1 for r in report_rows if r.get("validated") and r.get("discriminating")),
            "controls_unavailable": sum(
                1 for r in report_rows
                if not all(r.get("controls_available", {}).values())),
        },
        "problems": [
            {"name": n, "harnesses": [h.record() for h in hs]}
            for n, hs in sorted(frozen.items())
        ],
        "rejects_miss_rate": {
            "per_harness": [
                {"problem": r["problem"], "slot": r["slot"],
                 "miss_rate": (r.get("reject_side") or {}).get("miss_rate"),
                 "n_judged": (r.get("reject_side") or {}).get("n_judged"),
                 "n_missed": (r.get("reject_side") or {}).get("n_missed"),
                 "validated": r.get("validated")}
                for r in report_rows],
            "mean": (round(sum((r.get("reject_side") or {}).get("miss_rate")
                               for r in report_rows
                               if (r.get("reject_side") or {}).get("miss_rate") is not None)
                            / max(1, sum(1 for r in report_rows
                                         if (r.get("reject_side") or {}).get("miss_rate") is not None)), 4)),
        },
        # §7 A7 ruling (3): DIRECTION-NEUTRAL divergence rows. These are NOT
        # predictions about which rule wins — that is the measurement. A property
        # that rejects a container-OK-but-docstring-violating canvas costs the
        # benchmark score while being truer to the contract, and that COST is as
        # much a finding as a win would be.
        "divergence_rows": [
            {"problem": "HumanEval_10_make_palindrome",
             "what": ("a container-OK canvas implements make_palindrome as "
                      "'cca' -> 'ccac' (not a palindrome) and 'cata' -> 'catatac' "
                      "(the docstring's own worked example says 'catac'); the "
                      "docstring-derived properties REJECT it."),
             "status": "direction-neutral: either emission-rule pick is a finding "
                       "about docstring-vs-benchmark truth"},
            {"problem": "HumanEval_23_strlen",
             "what": ("a container-OK canvas returns strlen of the "
                      "single non-ASCII char U+00A1 as 2 (byte count) while the "
                      "docstring says 'length of given string'; the "
                      "docstring-derived property counts characters and REJECTS it."),
             "status": "direction-neutral: the property is truer to the contract "
                       "and may cost the benchmark score"},
        ],
        # A problem needing many rewrites is a datum: the human-authoring analog
        # of the generator's difficulty.
        "rewrite_counts": {
            "per_problem": rewrite_counts,
            "note": ("counts are 1 (as-first-written) plus each rewrite, with the "
                     "reason for every rewrite in `rewrite_log`"),
        },
        "rewrite_log": rewrite_log,
        "validation_report": report_rows,
    }
    body = {k: v for k, v in payload.items() if k != "sha256"}
    stamped = dict(body, sha256=compute_harness_freeze_sha256(body))
    p = Path(args.manifest)
    p.write_text(json.dumps(stamped, indent=2) + "\n", encoding="utf-8")
    print(f"\ncoverage: {covered}/25 problems, {total} harnesses")
    print(f"G4a AS AMENDED (>=19/21 control-available): {'PASS' if g4a else 'FAIL'}")
    print(f"watchlist {payload['coverage']['watchlist']}")
    print(f"manifest: {p}\nsha256: {stamped['sha256']}")
    if args.json:
        Path(args.json).write_text(json.dumps(stamped, indent=1), encoding="utf-8")
    return 0 if g4a else 1


if __name__ == "__main__":
    raise SystemExit(main())
