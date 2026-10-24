#!/usr/bin/env python3
"""M3.1 oracle audits: what a cell's own ensemble CONTAINED vs what it emitted.

A cell's score is the container verdict of one canvas per problem, so any cell
forfeits whatever it drew but did not emit. The oracle counts the forfeits: how
many problems had at least one container-OK canvas in the ensemble. It is the
ceiling the emission rule could have reached, and the gap between it and the
emitted score is the selection headroom M3.1 exists to measure.

Three audits, each with its own epistemic status (design §3 G3 + §7 Q3 ruling):

**``engine``** — for an M3.1 cell whose seam banked EVERY particle's canvas
(``<out>.canvases/``). All canvases are container-scored in one window, so the
oracle is EXACT and the audit is post-hoc attribution, never a pre-registered
input to a pre-cell claim.

**``cfull-bound``** — for the BANKED M3.0 C-full cell, which saved only the
emitted canvas. A true oracle is impossible there (a sha cannot be
container-scored), so this audit reports the **R1-clean-alternate upper bound**
from the steer logs: for every problem whose emitted canvas was rejected, how
many ALTERNATE particles were R1-clean at the final checkpoint. It is a CEILING
on forfeits, never an oracle count, and the report says so in those words.

**``banked``** — for any M3.0-style cell with a full-ensemble export + results
(C-null, C-screen), where every canvas was already scored.

    # exact audit of an M3.1 engine cell (needs a scored all-canvas window)
    temper/.venv/bin/python scripts/m3_31_oracle.py --mode engine \\
        --run bench/multipl-e/runs/<ts>-p-prop-engine.jsonl \\
        --results <scored all-canvas results.json> \\
        --emitted-results bench/multipl-e/m3-31/p-prop-engine-results.json \\
        --json bench/multipl-e/m3-31/p-prop-engine-oracle.json

    # labeled upper bound for the banked C-full cell
    temper/.venv/bin/python scripts/m3_31_oracle.py --mode cfull-bound \\
        --steer-dir bench/multipl-e/runs/20261023T133243Z-c-full.jsonl.steer \\
        --emitted-results bench/multipl-e/m3-p3/c-full-emission-results.json \\
        --json bench/multipl-e/m3-31/c-full-oracle-bound.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from temper.steer.verdict import (  # noqa: E402
    final_checkpoint_clean,
    ghost_close,
    load_verdict_config,
    r1_typecheck,
)
from temper.steer.view import SteerProblemView  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]


def _read_jsonl(path: Path) -> List[dict]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def _meta_rows(path: Path):
    meta, rows = None, []
    for o in _read_jsonl(path):
        if o.get("type") == "run":
            meta = o
        elif o.get("type") == "candidate":
            rows.append(o)
    if meta is None:
        raise SystemExit(f"{path}: no run meta line")
    return meta, rows


def _emitted(path: str) -> Dict[str, dict]:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    out = {}
    for row in d["per_problem"]:
        st = row.get("statuses_official") or []
        out[row["name"]] = {"pass": any(s == "OK" for s in st),
                            "statuses": st,
                            "emitted": row.get("emitted")}
    return out


# --------------------------------------------------------------------------
# exact audit (M3.1 cells with banked canvases)
# --------------------------------------------------------------------------


def engine_audit(meta: dict, rows: List[dict], results: dict,
                 emitted: Dict[str, dict]) -> dict:
    """Every banked canvas scored -> exact oracle, per-problem forfeit table.

    ``results`` must be the all-canvas scoring of the banked texts, with a
    per-canvas status map. It is produced by ``--mode engine --stage export``
    through the unchanged export instrument; the statuses ride
    ``statuses_by_index`` in the results payload this script's own export step
    writes (one synthetic problem per canvas would break the honest counting,
    so the mapping is explicit instead).
    """
    per = {r["name"]: r for r in results.get("per_problem", [])}
    table = []
    for name in meta.get("problem_names") or sorted(per):
        row = per.get(name, {})
        by_index = {int(k): v for k, v in (row.get("statuses_by_index") or {}).items()}
        ok_indices = [i for i, s in sorted(by_index.items()) if s == "OK"]
        em = emitted.get(name, {})
        emitted_ok = bool(em.get("pass"))
        table.append({
            "problem": name,
            "n_canvases": len(by_index),
            "n_container_ok": len(ok_indices),
            "ok_indices": ok_indices,
            "any_ok": bool(ok_indices),
            "emitted": em.get("emitted"),
            "emitted_ok": emitted_ok,
            "forfeited": bool(ok_indices) and not emitted_ok,
            "emitted_was_ok_canvas": (em.get("emitted") in ok_indices),
        })
    return {
        "mode": "engine",
        "oracle": sum(1 for t in table if t["any_ok"]),
        "emitted": sum(1 for t in table if t["emitted_ok"]),
        "n_problems": len(table),
        "forfeits": [t["problem"] for t in table if t["forfeited"]],
        "n_forfeits": sum(1 for t in table if t["forfeited"]),
        "epistemic_status": ("EXACT — every particle's canvas was banked and "
                             "scored; this is post-hoc attribution, never a "
                             "pre-cell input (§7 Q3 ruling)"),
        "table": table,
    }


# --------------------------------------------------------------------------
# labeled upper bound (the banked M3.0 C-full cell)
# --------------------------------------------------------------------------


def cfull_bound(steer_dir: Path, emitted: Dict[str, dict]) -> dict:
    """R1-clean-alternate UPPER BOUND on C-full's forfeits.

    C-full banked only the emitted canvas, so no exact oracle exists. What IS
    recoverable from the D2 steer logs is each particle's final-checkpoint
    ``r1.ok``: for a problem whose emitted canvas was rejected, an R1-clean
    alternate is a canvas the emission rule COULD have picked. R1-clean is
    necessary but NOT sufficient for container-OK (a clean canvas can still be
    semantically wrong), so this can only over-count — it is a ceiling.
    """
    views = {}
    from temper.multipl_e.bench import load_problems
    for td in load_problems("humaneval-rs"):
        views[td["name"]] = SteerProblemView.from_task(td)
    table = []
    for log_path in sorted(steer_dir.glob("*.jsonl")):
        name = log_path.stem
        events = _read_jsonl(log_path)
        emit = next((e for e in events if e["kind"] == "emit"), {})
        clean = emit.get("final_clean") or []
        emitted_particle = emit.get("particle")
        em = emitted.get(name, {})
        rejected = not bool(em.get("pass"))
        alternates = [i for i in clean if i != emitted_particle]
        table.append({
            "problem": name,
            "n_r1_clean": len(clean),
            "emitted": emitted_particle,
            "emitted_ok": bool(em.get("pass")),
            # an upper bound: a clean alternate COULD have been better
            "clean_alternates": alternates,
            "n_clean_alternates": len(alternates),
            "bound_forfeit": bool(rejected) and bool(alternates),
        })
    return {
        "mode": "cfull-bound",
        "n_problems": len(table),
        "emitted_passers": sum(1 for t in table if t["emitted_ok"]),
        "upper_bound_oracle": sum(1 for t in table if t["emitted_ok"]
                                  or t["bound_forfeit"]),
        "bound_forfeits": [t["problem"] for t in table if t["bound_forfeit"]],
        "n_bound_forfeits": sum(1 for t in table if t["bound_forfeit"]),
        "epistemic_status": (
            "UPPER BOUND, NOT AN ORACLE — the M3.0 C-full cell banked only the "
            "emitted canvas, so its non-emitted canvases cannot be "
            "container-scored. R1-clean is necessary but not sufficient for "
            "container-OK, so this counts a forfeit for every rejected problem "
            "with a compile-clean alternate and therefore can only over-count. "
            "Reported as a ceiling; §7 Q3 ruling rejects regenerating C-full."),
        "table": table,
    }


# --------------------------------------------------------------------------
# banked full-ensemble audit
# --------------------------------------------------------------------------


def banked_audit(manifest: dict, results: dict,
                 emitted: Dict[str, dict]) -> dict:
    statuses_of = {r["name"]: list(r.get("statuses_official") or [])
                   for r in results.get("per_problem", [])}
    table = []
    for prob in manifest.get("problems", []):
        name = prob["name"]
        sts = statuses_of.get(name, [])
        ok_indices = [int(sr["i"]) for j, sr in enumerate(prob["source_rows"])
                      if j < len(sts) and sts[j] == "OK"]
        em = emitted.get(name, {})
        table.append({
            "problem": name,
            "n_canvases": len(sts),
            "n_container_ok": len(ok_indices),
            "ok_indices": ok_indices,
            "any_ok": bool(ok_indices),
            "emitted": em.get("emitted"),
            "emitted_ok": bool(em.get("pass")),
            "forfeited": bool(ok_indices) and not bool(em.get("pass")),
        })
    return {
        "mode": "banked",
        "oracle": sum(1 for t in table if t["any_ok"]),
        "emitted": sum(1 for t in table if t["emitted_ok"]),
        "n_problems": len(table),
        "forfeits": [t["problem"] for t in table if t["forfeited"]],
        "n_forfeits": sum(1 for t in table if t["forfeited"]),
        "epistemic_status": "EXACT — every canvas of the ensemble was scored",
        "table": table,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", required=True,
                    choices=("engine", "cfull-bound", "banked"))
    ap.add_argument("--run", default=None)
    ap.add_argument("--results", default=None,
                    help="all-canvas scoring of the banked canvas texts")
    ap.add_argument("--steer-dir", default=None)
    ap.add_argument("--export", default=None, help="full-ensemble export dir")
    ap.add_argument("--emitted-results", required=True)
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    emitted = _emitted(args.emitted_results)
    if args.mode == "engine":
        meta, _rows = _meta_rows(Path(args.run))
        res = json.loads(Path(args.results).read_text(encoding="utf-8"))
        report = engine_audit(meta, _rows, res, emitted)
    elif args.mode == "cfull-bound":
        report = cfull_bound(Path(args.steer_dir), emitted)
    else:
        manifest = json.loads((Path(args.export) / "manifest.json")
                              .read_text(encoding="utf-8"))
        res = json.loads(Path(args.results).read_text(encoding="utf-8"))
        report = banked_audit(manifest, res, emitted)
    report["verdict_config_sha256"] = load_verdict_config().sha256
    report["emitted_results"] = args.emitted_results

    print(f"=== oracle audit [{report['mode']}] ===")
    print(f"  {report['epistemic_status']}")
    key = ("oracle" if report["mode"] != "cfull-bound" else "upper_bound_oracle")
    print(f"  {key}: {report[key]}/{report['n_problems']}   "
          f"emitted: {report['emitted_passers'] if report['mode'] == 'cfull-bound' else report['emitted']}")
    print(f"  forfeits: {report['n_forfeits'] if 'n_forfeits' in report else report['n_bound_forfeits']}")
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(report, indent=1), encoding="utf-8")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
