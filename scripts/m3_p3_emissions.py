#!/usr/bin/env python3
"""PHASE B: the two emission tables computed from ONE scored 8-canvas ensemble.

M3.0 P3 (design §4.1, §4.3). C-null and C-screen are not two generations — they
are two EMISSION RULES over the same completed 8-canvas C-null ensemble:

  * **C-null** (rev1 §4.1): the T=0.2 structural canvas if it is R1-clean, else
    the lowest-temperature clean canvas. The "diversity ladder only" control.
  * **C-screen** (rev1, review B's null): best R1-clean canvas by mean commit
    confidence, ties to the lowest temperature. The static-final-filter null —
    "temp ladder + final rustc screen, ZERO mid-decode machinery".

Both rules live in ``temper/steer/emission.py`` so this script and the in-loop
C-full engine (:mod:`temper.steer.engine`) run the same code. The screen must
be computed BEFORE C-full's result is analyzed (design §5 P3), so the primary
delta's yardstick exists ahead of the steered rows.

Inputs, all committed artifacts:

  --run       the C-null run record (<ts>-c-null.jsonl)
  --export    the norepair export dir of that record's FULL ensemble, whose
              manifest.json names the completion->candidate-row mapping
  --results   the container results JSON for that export
  --out-dir   where the two emission result JSONs are written

Per-canvas statuses are recovered through the manifest's ``source_rows`` (a
positional zip against a naive filter would shift every later status within a
problem by the excluded-row count — the m3_p1_reachability.py lesson).

    temper/.venv/bin/python scripts/m3_p3_emissions.py \\
        --run bench/multipl-e/runs/<ts>-c-null.jsonl \\
        --export bench/multipl-e/exports/<ts>-c-null-norepair \\
        --results bench/multipl-e/results/<ts>.json \\
        --out-dir /tmp/m3_p3_emissions
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from temper.multipl_e.bench import load_problems  # noqa: E402
from temper.steer.emission import (  # noqa: E402
    STRUCTURAL_TEMP,
    rank_screen,
    select_screen,
    select_structural,
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
    """One completed canvas of the ensemble, as the emission rules see it."""

    __slots__ = ("index", "temperature", "seed", "mean_commit_conf", "raw_path",
                 "raw_text", "final_clean", "status", "excluded", "r1_error_codes")

    def __init__(self, row: dict):
        self.index = int(row["i"])
        self.temperature = float(row["temperature"])
        self.seed = int(row.get("seed") or 0)
        self.mean_commit_conf = row.get("mean_commit_conf")
        self.raw_path = row.get("raw_path")
        self.raw_text = ""
        self.final_clean = False
        self.status: Optional[str] = None      # container status, None = not scored
        self.excluded = False                  # export refused this row
        self.r1_error_codes: List[str] = []


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
        p = REPO_ROOT / raw_path
    return p.read_text(encoding="utf-8") if p.exists() else ""


def build_ensembles(run: dict, manifest: dict, results: dict) -> Dict[str, List[Canvas]]:
    """{problem: [Canvas]} with per-canvas container status + final R1 screen.

    The R1 clean verdict is computed HERE, offline, with host rustc on the
    sidecar text — the same ghost-closure screen the steer loop applies, so
    C-screen and C-full are ranked by the same predicate.
    """
    cfg = load_verdict_config()
    views = {td["name"]: SteerProblemView.from_task(td)
             for td in load_problems("humaneval-rs")}
    # manifest: problem -> {candidate index: completion position}
    position_of: Dict[str, Dict[int, int]] = {}
    excluded_at: Dict[str, set] = {}
    for prob in manifest.get("problems", []):
        position_of[prob["name"]] = {int(sr["i"]): j
                                    for j, sr in enumerate(prob["source_rows"])}
    for ex in manifest.get("excluded", []):
        excluded_at.setdefault(ex.get("problem"), set()).add(int(ex.get("i")))
    statuses_of = {row["name"]: list(row.get("statuses_official") or [])
                   for row in results["per_problem"]}

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
            except Exception as exc:          # a verifier crash is never clean
                c.final_clean = False
                c.r1_error_codes = [f"verifier:{type(exc).__name__}"]
        ensembles.setdefault(name, []).append(c)
    for name in ensembles:
        ensembles[name].sort(key=lambda c: c.index)
    return ensembles


def emit_row(name: str, chosen: Optional[Canvas], ensemble: List[Canvas],
             rule: str) -> dict:
    """One problem's emitted outcome under one rule, in results-JSON shape."""
    if chosen is None:
        return {"name": name, "statuses_official": [], "n_scored": 0,
                "n_excluded": 1, "emitted": None, "rule": rule}
    # honest counting: an export-excluded canvas is a scored FAILURE (its
    # extraction failed), so it appears as an exclusion, not as a scored row
    if chosen.excluded or chosen.status is None:
        return {"name": name, "statuses_official": [], "n_scored": 0,
                "n_excluded": 1, "emitted": chosen.index, "rule": rule,
                "emitted_temp": chosen.temperature,
                "emitted_conf": chosen.mean_commit_conf,
                "emitted_clean": chosen.final_clean,
                "emitted_r1_codes": chosen.r1_error_codes}
    return {"name": name, "statuses_official": [chosen.status], "n_scored": 1,
            "n_excluded": 0, "emitted": chosen.index, "rule": rule,
            "emitted_temp": chosen.temperature,
            "emitted_conf": chosen.mean_commit_conf,
            "emitted_clean": chosen.final_clean,
            "emitted_r1_codes": chosen.r1_error_codes}


def results_payload(label: str, rows: List[dict], source: dict) -> dict:
    """A results JSON the D6 tooling (scripts/m3_cells.py) can load directly."""
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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--export", required=True,
                    help="norepair export dir of the FULL C-null ensemble")
    ap.add_argument("--results", required=True,
                    help="container results JSON for that export")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args(argv)

    run = load_run(Path(args.run))
    manifest = json.loads((Path(args.export) / "manifest.json")
                          .read_text(encoding="utf-8"))
    results = json.loads(Path(args.results).read_text(encoding="utf-8"))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ensembles = build_ensembles(run, manifest, results)
    meta = run["meta"]
    cfg = load_verdict_config()
    print(f"C-null run: {Path(args.run).name}")
    print(f"  problems={meta.get('n_problems')} ladder={meta.get('temperatures')}")
    print(f"  verdict_config_sha256={cfg.sha256}")

    cnull_rows, screen_rows, detail = [], [], []
    for name in meta.get("problem_names") or sorted(ensembles):
        ens = ensembles.get(name, [])
        if not ens:
            print(f"  !! {name}: no canvases in the run record")
            continue
        structural = select_structural(ens)
        screened = select_screen(ens)
        cnull_rows.append(emit_row(name, structural, ens, "structural"))
        screen_rows.append(emit_row(name, screened, ens, "screen"))
        n_clean = sum(1 for c in ens if c.final_clean)
        n_ok = sum(1 for c in ens if c.status == "OK")
        detail.append({
            "problem": name,
            "n_canvases": len(ens),
            "n_r1_clean": n_clean,
            "n_container_ok": n_ok,
            "any_ok": n_ok >= 1,
            "cnull_emitted": structural.index if structural else None,
            "cnull_temp": structural.temperature if structural else None,
            "cnull_status": (structural.status if structural else None),
            "screen_emitted": screened.index if screened else None,
            "screen_temp": screened.temperature if screened else None,
            "screen_status": (screened.status if screened else None),
            "screen_ranking": [c.index for c in rank_screen(ens)],
            "canvas_statuses": [c.status for c in ens],
            "canvas_temps": [c.temperature for c in ens],
            "canvas_clean": [c.final_clean for c in ens],
            "diverged": (structural.index if structural else None)
                        != (screened.index if screened else None),
        })

    cnull = results_payload("C-null", cnull_rows, results)
    screen = results_payload("C-screen", screen_rows, results)
    cnull_path = out_dir / "c-null-emission-results.json"
    screen_path = out_dir / "c-screen-emission-results.json"
    cnull_path.write_text(json.dumps(cnull, indent=1), encoding="utf-8")
    screen_path.write_text(json.dumps(screen, indent=1), encoding="utf-8")

    # ---- report -----------------------------------------------------------
    hdr = (f"{'problem':44s} {'clean':>7s} {'ok':>4s} "
           f"{'C-null':>18s} {'C-screen':>18s}")
    print("\n=== per-problem emission table ===")
    print(hdr)
    print("-" * len(hdr))
    for d in detail:
        def cell(temp, st, em):
            st = st or ("EXCL" if em is not None else "-")
            return f"p{em}@T{temp:<4} {st:<10}" if em is not None else "-"
        print(f"{d['problem']:44s} "
              f"{d['n_r1_clean']:3d}/{d['n_canvases']:<3d} {d['n_container_ok']:3d}  "
              f"{cell(d['cnull_temp'], d['cnull_status'], d['cnull_emitted']):18s} "
              f"{cell(d['screen_temp'], d['screen_status'], d['screen_emitted']):18s}"
              + ("  <-diverged" if d["diverged"] else ""))
    n_cnull = sum(1 for r in cnull_rows if "OK" in r["statuses_official"])
    n_screen = sum(1 for r in screen_rows if "OK" in r["statuses_official"])
    n_problems = len(detail)
    print(f"\nC-null   honest passes: {n_cnull}/{n_problems}")
    print(f"C-screen honest passes: {n_screen}/{n_problems}")
    print(f"emission diverged on {sum(1 for d in detail if d['diverged'])}"
          f"/{n_problems} problems")
    print(f"\nwrote {cnull_path}\nwrote {screen_path}")

    (out_dir / "emission-detail.json").write_text(json.dumps({
        "run": str(args.run), "export": str(args.export),
        "results": str(args.results),
        "verdict_config_sha256": cfg.sha256,
        "structural_temp": STRUCTURAL_TEMP,
        "rows": detail,
        "cnull_passes": n_cnull,
        "screen_passes": n_screen,
    }, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
