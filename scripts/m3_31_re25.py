#!/usr/bin/env python3
"""M3.1 PHASE 4 — P-25re: the HumanEval_25 re-decide, 4 base seeds, BOTH roles.

Design §3/§4 G5. HumanEval_25 is the single P3 flip that failed replication at
2 games, and it is the one the attribution argument most needed: budget-fixable,
P1-unreachable, and the only problem neither knowledge delivery nor the ladder
alone ever produced. G5 re-decides it at the pre-registered 4-game standard:

    engine passes >= 3/4 AND unsteered passes <= 1/4
      => engine-specific reachability claim admissible
    otherwise
      => draw-luck, and m3.0-p3-report.md's attribution note is amended

**Roles (§7 Q4 ruling).** Both roles play the SAME 8-slot rev2 ladder at the
SAME envelope with the SAME base seed in each pair, so the only difference is
the policy:

  * **unsteered** — ``policy_enabled=False``, i.e. exactly C-null semantics
    (the role M3.0 measured as "unsteered"), emitting the T=0.2 structural
    canvas via the ladder run itself (a plain ``bench-multipl-e`` invocation);
  * **engine** — ``run_game`` with policy ON and v3 property emission in-loop at
    the final window (the M3.1 engine).

Seeds 2000..2003, paired on seed across roles (``seed_paired`` logged).

**Post-hoc columns, never pre-registered inputs (§7 Q4 addition 1).** For every
game all 8 canvases are banked and scored, so three labeled columns are
computed: (i) the v2-confidence-screen pick, (ii) the v3 offline property pick,
(iii) the oracle. G5 adjudicates ONLY on the pre-registered emitted answers;
the columns let any outcome be decomposed into draw-luck vs emission-rule vs
engine-pipeline without further GPU (§7 Q4 addition 2).

    # 1. the two role games (GPU, generation phase, Rancher CLOSED)
    temper/.venv/bin/python scripts/m3_31_re25.py --stage generate \\
        --which unsteered --out bench/multipl-e/runs/<ts>-p25re-unsteered.jsonl
    temper/.venv/bin/python scripts/m3_31_re25.py --stage generate \\
        --which engine --out bench/multipl-e/runs/<ts>-p25re-engine.jsonl

    # 2. after the scoring window(s), classify
    temper/.venv/bin/python scripts/m3_31_re25.py --stage classify \\
        --unsteered-results <json> --engine-results <json> \\
        --json bench/multipl-e/m3-31/p25re.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from temper.steer.emission import (  # noqa: E402
    EMISSION_PROPERTY,
    EMISSION_SCREEN,
    property_tier,
    rank_by_rule,
    select_by_rule,
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

PROBLEM_NUMBER = 25
BASE_SEEDS = (2000, 2001, 2002, 2003)
ENVELOPE = 2048
STEPS = 256


def _problem_name() -> str:
    from temper.multipl_e.bench import load_problems
    return load_problems("humaneval-rs",
                         problem_numbers=[PROBLEM_NUMBER])[0]["name"]


# --------------------------------------------------------------------------
# generate
# --------------------------------------------------------------------------


def _generate(args) -> int:
    from temper.multipl_e.bench import run_bench
    from temper.steer.engine import REV2_LADDER, make_steer_sample_fn
    from temper.steer.property import load_harness_freeze

    name = _problem_name()
    out = Path(args.out)
    if not out.is_absolute():
        out = REPO_ROOT / out
    cfg = load_verdict_config()
    print(f"P-25re [{args.which}]: {name} x {len(BASE_SEEDS)} base seeds "
          f"{list(BASE_SEEDS)} x {len(REV2_LADDER)} canvases x {STEPS} steps")
    print(f"  envelope {ENVELOPE} NFE/problem; seed_paired=True; "
          f"config {cfg.sha256[:12]}")

    if args.which == "engine":
        frozen = load_harness_freeze(args.freeze)
        if name not in frozen:
            raise SystemExit(f"{name} has no frozen harness — the engine role "
                             "cannot run a property cell without it")
        fn = make_steer_sample_fn(
            out=out, envelope_nfe=ENVELOPE, steps=STEPS,
            canvas_len=args.canvas, base_seed=BASE_SEEDS[0], alg="entropy",
            harnesses=frozen, log=lambda s: print(s, flush=True))
        run_bench("humaneval-rs", n_problems=None, k=1, steps=STEPS,
                  canvas=args.canvas, out=out, sample_fn=fn,
                  problem_numbers=[PROBLEM_NUMBER],
                  extra_meta={"mode": "steered", "cell": "P-25re-engine",
                              "envelope_nfe": ENVELOPE,
                              "diversity_ladder": list(REV2_LADDER),
                              "steer_config_sha256": cfg.sha256,
                              "kill_clone_enabled": False,
                              "emission": EMISSION_PROPERTY,
                              "seed_paired": True,
                              "base_seeds": list(BASE_SEEDS),
                              "role": "engine"},
                  log=lambda s: print(s, flush=True))
        return 0

    # UNSTEERED role: the plain ladder run (C-null semantics), one canvas per
    # base seed, emitted by the ladder itself — the role M3.0 measured.
    run_bench("humaneval-rs", n_problems=None, k=1, steps=STEPS,
              canvas=args.canvas, base_seed=BASE_SEEDS[0], alg="entropy",
              out=out, problem_numbers=[PROBLEM_NUMBER],
              temperatures=list(REV2_LADDER),
              extra_meta={"mode": "unsteered", "cell": "P-25re-unsteered",
                          "envelope_nfe": ENVELOPE,
                          "diversity_ladder": list(REV2_LADDER),
                          "steer_config_sha256": cfg.sha256,
                          "emission": "structural_t0.2_if_clean (C-null)",
                          "seed_paired": True,
                          "base_seeds": list(BASE_SEEDS),
                          "role": "unsteered"},
              log=lambda s: print(s, flush=True))
    return 0


# --------------------------------------------------------------------------
# classify
# --------------------------------------------------------------------------


def _cells_by_seed(run_path: Path) -> Dict[int, dict]:
    """{base_seed: {canvas index -> info}} from a run record + its sidecars."""
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
    if meta is None:
        raise SystemExit(f"{run_path}: no run meta line")
    out: Dict[int, dict] = {}
    for row in rows:
        seed = int(row.get("seed") or 0)
        raw = row.get("raw_path")
        text = ""
        if raw:
            p = Path(raw)
            if not p.is_absolute():
                p = REPO_ROOT / p
            text = p.read_text(encoding="utf-8") if p.exists() else ""
        out.setdefault(seed, {})[int(row["i"])] = {
            "temperature": float(row.get("temperature") or 0.0),
            "mean_commit_conf": row.get("mean_commit_conf"),
            "text": text,
        }
    return out


def _statuses_by_index(results: dict, name: str) -> Dict[int, str]:
    for row in results.get("per_problem", []):
        if row["name"] == name:
            return {int(k): v for k, v in
                    (row.get("statuses_by_index") or {}).items()}
    return {}


class _Cand:
    __slots__ = ("index", "temperature", "final_clean", "mean_commit_conf",
                 "property_pass_rate")

    def __init__(self, index, temperature, final_clean, mean_commit_conf,
                 property_pass_rate=None):
        self.index = index
        self.temperature = temperature
        self.final_clean = final_clean
        self.mean_commit_conf = mean_commit_conf
        self.property_pass_rate = property_pass_rate


def _classify(args) -> int:
    """G5 on the pre-registered emitted answers + the three post-hoc columns."""
    from temper.multipl_e.bench import load_problems
    from temper.steer.property import (
        load_harness_freeze, make_pub_program, run_r3)

    name = _problem_name()
    view = SteerProblemView.from_task(
        load_problems("humaneval-rs", problem_numbers=[PROBLEM_NUMBER])[0])
    cfg = load_verdict_config()
    un = json.loads(Path(args.unsteered_results).read_text(encoding="utf-8"))
    eng = json.loads(Path(args.engine_results).read_text(encoding="utf-8"))
    un_status = {row["name"]: list(row.get("statuses_official") or [])
                 for row in un["per_problem"]}
    eng_status = {row["name"]: list(row.get("statuses_official") or [])
                  for row in eng["per_problem"]}

    def role_passers(res: dict) -> int:
        """Honest counting: a problem's emitted answer passes or it does not."""
        n = 0
        for row in res["per_problem"]:
            if any(s == "OK" for s in (row.get("statuses_official") or [])):
                n += 1
        return n

    # The role games are ONE problem each over 4 base seeds, so the pass count
    # is the number of games whose emitted answer passed.
    eng_passes = role_passers(eng)
    un_passes = role_passers(un)
    n_games = len(BASE_SEEDS)

    # ---- post-hoc columns (planning support; NOT gate inputs) --------------
    columns = []
    if args.engine_run:
        frozen = load_harness_freeze(args.freeze)
        harnesses = frozen.get(name) or []
        per_seed = _cells_by_seed(Path(args.engine_run))
        all_status = _statuses_by_index(
            json.loads(Path(args.all_canvases_results).read_text()), name) \
            if args.all_canvases_results else {}
        for seed in BASE_SEEDS:
            cells = per_seed.get(seed, {})
            cands: List[_Cand] = []
            for idx, info in sorted(cells.items()):
                clean = False
                rate = None
                if info["text"]:
                    try:
                        v = r1_typecheck(
                            ghost_close(view, info["text"]),
                            hole_proximity=cfg.hole_proximity_lines)
                        clean = final_checkpoint_clean(v)
                    except Exception:
                        clean = False
                if clean and harnesses and info["text"]:
                    try:
                        program = make_pub_program(view, info["text"])
                        verdict = run_r3(program.text, harnesses,
                                         canvas_ref=f"p{idx}", problem=name,
                                         cases=int(cfg.raw["m3_1"]["r3"]["cases"]),
                                         timeout_ms=int(cfg.raw["m3_1"]["r3"]["timeout_ms"]),
                                         workdir=args.r3_workdir)
                        rate = verdict.pass_rate
                    except Exception:
                        rate = None
                cands.append(_Cand(idx, info["temperature"], clean,
                                   info["mean_commit_conf"], rate))
            if not cands:
                continue
            v2 = select_by_rule(cands, EMISSION_SCREEN)
            v3 = select_by_rule(cands, EMISSION_PROPERTY)
            oracle = [c.index for c in cands
                      if all_status.get(c.index) == "OK"]
            columns.append({
                "base_seed": seed,
                "v2_screen_pick": v2.index if v2 else None,
                "v2_status": all_status.get(v2.index) if v2 else None,
                "v3_offline_pick": v3.index if v3 else None,
                "v3_status": all_status.get(v3.index) if v3 else None,
                "oracle_ok_indices": oracle,
                "any_ok": bool(oracle),
            })

    report = {
        "problem": name,
        "base_seeds": list(BASE_SEEDS),
        "seed_paired": True,
        "n_games": n_games,
        "engine_passes": eng_passes,
        "unsteered_passes": un_passes,
        "engine_statuses": eng_status.get(name),
        "unsteered_statuses": un_status.get(name),
        "engine_results": args.engine_results,
        "unsteered_results": args.unsteered_results,
        "post_hoc_columns": columns,
        "post_hoc_note": ("forensics only (§7 Q4 addition 1): G5 adjudicates on "
                          "the emitted answers; these columns decompose any "
                          "outcome into draw-luck vs emission-rule vs engine"),
        "verdict_config_sha256": cfg.sha256,
        "classification": ("engine_specific"
                           if (eng_passes >= 3 and un_passes <= 1)
                           else "draw_luck"),
    }
    print(f"=== P-25re [{name}] ===")
    print(f"  engine    {eng_passes}/{n_games}  {eng_status.get(name)}")
    print(f"  unsteered {un_passes}/{n_games}  {un_status.get(name)}")
    print(f"  G5: {'engine-specific' if report['classification'] == 'engine_specific' else 'DRAW-LUCK'}")
    if columns:
        print("\n  post-hoc columns (forensics, not gate inputs):")
        for c in columns:
            print(f"    seed {c['base_seed']}: v2=p{c['v2_screen_pick']}"
                  f"({c['v2_status']}) v3=p{c['v3_offline_pick']}"
                  f"({c['v3_status']}) oracle={c['oracle_ok_indices']}")
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(report, indent=1), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=("generate", "classify"))
    ap.add_argument("--which", choices=("engine", "unsteered"), default="engine",
                    help="--stage generate: which ROLE to play")
    ap.add_argument("--out", default=None)
    ap.add_argument("--canvas", type=int, default=384)
    ap.add_argument("--freeze", default=None)
    ap.add_argument("--unsteered-results", default=None)
    ap.add_argument("--engine-results", default=None)
    ap.add_argument("--engine-run", default=None,
                    help="the engine role's run record (post-hoc columns)")
    ap.add_argument("--all-canvases-results", default=None,
                    help="the all-canvas scoring of the engine role's banked "
                         "canvases (post-hoc oracle column)")
    ap.add_argument("--r3-workdir", default=None)
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)
    if args.stage == "generate":
        if not args.out:
            raise SystemExit("--stage generate needs --out")
        return _generate(args)
    for req in ("unsteered_results", "engine_results"):
        if not getattr(args, req):
            raise SystemExit(f"--stage classify needs --{req.replace('_', '-')}")
    return _classify(args)


if __name__ == "__main__":
    raise SystemExit(main())
