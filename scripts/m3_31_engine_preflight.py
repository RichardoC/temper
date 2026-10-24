#!/usr/bin/env python3
"""M3.1 P-prop-engine GPU PRE-FLIGHT: verify the R3-wired loop end-to-end.

23.3 h of GPU are pre-approved and the engine cell is 15 h of it, so the cell's
own failure modes get a cheap gate first — the P3 discipline, extended with the
three quantities M3.1 adds:

  * **R3 call count == expected** (harnesses x R1-clean canvases per game). A
    count that is too low means R3 silently did not fire; too high means it
    fired twice or on canvases R1 rejected;
  * **R3 wall share <= 10 %** of the game's generation wall — the D3 criterion
    that caps verifier work, now carrying R3's cargo cost;
  * **no kill fires** and the demoted role still holds (R3 must not have
    re-enabled anything);
  * plus the emitted canvas is the v3 rule's pick over the logged evidence, the
    D2 log replays, and every particle's canvas was banked (Q3).

Design §3: 2 problems x 8 canvases x 32 steps.

    # generation phase (Rancher CLOSED)
    temper/.venv/bin/python scripts/m3_31_engine_preflight.py --steps 32
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from temper.multipl_e.bench import load_problems  # noqa: E402
from temper.steer.emission import EMISSION_PROPERTY  # noqa: E402
from temper.steer.engine import REV2_LADDER, make_steer_sample_fn  # noqa: E402
from temper.steer.property import load_harness_freeze  # noqa: E402
from temper.steer.verdict import load_verdict_config  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]

# Two problems the P1 probe measured REACHABLE at high temperature, so the
# emission rule has a real choice to make (the P3 preflight set).
DEFAULT_PROBLEMS = (17, 11)


def read_log(path: Path) -> list:
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines()
            if ln.strip()]


def check_run(out_jsonl: Path, steps: int, envelope: int,
              n_harnesses: dict) -> dict:
    from temper.multipl_e.bench import _iter_record

    meta, rows = _iter_record(out_jsonl)
    cfg = load_verdict_config()
    checks = {}
    problems = []
    for row in rows:
        name = row["problem"]
        log_path = Path(str(out_jsonl) + ".steer") / f"{name}.jsonl"
        canvas_dir = Path(str(out_jsonl) + ".canvases")
        checks[f"{name}:steer_log_exists"] = log_path.exists()
        if not log_path.exists():
            continue
        events = read_log(log_path)
        kinds = [e["kind"] for e in events]
        emit = next((e for e in events if e["kind"] == "emit"), {})
        done = next((e for e in events if e["kind"] == "done"), {})
        meta_ev = events[0]
        r3_events = [e for e in events if e["kind"] == "r3"]
        clean = emit.get("final_clean") or []
        expected_r3 = int(n_harnesses.get(name, 0)) * len(clean)
        r3_wall = sum(float(e.get("duration_s") or 0.0) for e in r3_events)
        gen_wall = float(row.get("wall_s") or 0.0)
        c = {
            "name": name,
            "row_nfe": row["nfe"],
            "row_wall_s": gen_wall,
            "row_extract_ok": row["extract_ok"],
            "log_temperatures": meta_ev.get("temperatures"),
            "log_kill_clone_enabled": meta_ev.get("kill_clone_enabled"),
            "n_checkpoints": sum(1 for k in kinds if k == "checkpoint"),
            "checkpoint_steps": sorted({e["step"] for e in events
                                        if e["kind"] == "checkpoint"}),
            "n_kills": kinds.count("kill"),
            "n_kill_suppressed": kinds.count("kill_suppressed"),
            "emit_particle": emit.get("particle"),
            "emit_rule": emit.get("emission_rule"),
            "emit_r3_available": emit.get("r3_available"),
            "emit_r3_calls": emit.get("r3_calls"),
            "emit_property_pass_rate": emit.get("property_pass_rate"),
            "emit_ranking": emit.get("emission_ranking"),
            "screen_selected": emit.get("screen_selected"),
            "final_clean": clean,
            "n_r3_events": len(r3_events),
            "expected_r3_events": expected_r3,
            "r3_wall_s": round(r3_wall, 3),
            "r3_wall_share": (round(r3_wall / gen_wall, 5) if gen_wall else None),
            "n_abstain_events": kinds.count("r3_abstain"),
            "n_r3_error_events": kinds.count("r3_error"),
            "n_banked_canvases": len(list(canvas_dir.glob(f"{name}__p*.txt")))
                                   if canvas_dir.is_dir() else 0,
            "n_particles_spawned": kinds.count("spawn"),
            "verifier_wall_s": (done.get("meters") or {}).get(
                "verifier_wall_s", 0.0),
        }
        problems.append(c)

        checks[f"{name}:ladder_reached_the_game"] = (
            c["log_temperatures"] == list(REV2_LADDER))
        checks[f"{name}:all_eight_particles_spawned"] = (
            c["n_particles_spawned"] == len(REV2_LADDER))
        checks[f"{name}:config_sha_recorded"] = (
            meta_ev.get("verdict_config_sha256") == cfg.sha256)
        checks[f"{name}:demoted_role_recorded"] = (
            c["log_kill_clone_enabled"] is False)
        checks[f"{name}:ladder_fired_at_final_checkpoint_only"] = (
            c["checkpoint_steps"] == [steps])
        checks[f"{name}:no_kill_executed"] = c["n_kills"] == 0
        checks[f"{name}:nfe_within_envelope"] = (
            c["row_nfe"] is not None and c["row_nfe"] <= envelope)
        checks[f"{name}:r3_was_available"] = bool(c["emit_r3_available"])
        checks[f"{name}:emission_rule_is_v3"] = (
            c["emit_rule"] == EMISSION_PROPERTY)
        # THE pre-flight arithmetic (design §3)
        checks[f"{name}:r3_call_count_matches_expectation"] = (
            c["n_r3_events"] == c["expected_r3_events"])
        checks[f"{name}:r3_wall_share_le_10pct"] = (
            c["r3_wall_share"] is not None and c["r3_wall_share"] <= 0.10)
        checks[f"{name}:no_r3_verifier_crash"] = c["n_r3_error_events"] == 0
        checks[f"{name}:emitted_a_canvas"] = c["emit_particle"] is not None
        checks[f"{name}:every_particle_canvas_banked"] = (
            c["n_banked_canvases"] == c["n_particles_spawned"])
        # D2 replay with the new event kinds present
        try:
            from temper.steer.log import read_game_log, replay_game
            replay_game(read_game_log(log_path), cfg)
            checks[f"{name}:steer_log_replays"] = True
        except Exception as exc:
            checks[f"{name}:steer_log_replays"] = False
            checks[f"{name}:replay_error"] = repr(exc)

    checks["record_mode_is_steered"] = (meta or {}).get("mode") == "steered"
    checks["record_carries_config_sha"] = (
        (meta or {}).get("steer_config_sha256") == cfg.sha256)
    checks["one_row_per_problem"] = len(rows) == len(problems)
    return {"checks": checks, "problems": problems, "meta": meta,
            "rows": [{k: r.get(k) for k in
                      ("problem", "nfe", "temperature", "extract_ok",
                       "mean_commit_conf", "wall_s")} for r in rows]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--canvas", type=int, default=384)
    ap.add_argument("--problems", default=",".join(str(n) for n in DEFAULT_PROBLEMS))
    ap.add_argument("--freeze", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--json", default=None)
    ap.add_argument("--check-only", default=None)
    args = ap.parse_args(argv)

    envelope = args.steps * len(REV2_LADDER)
    frozen = load_harness_freeze(args.freeze)
    n_harnesses = {k: len(v) for k, v in frozen.items()}
    out = Path(args.out) if args.out else Path(
        tempfile.mkdtemp()) / "p-prop-engine-preflight.jsonl"

    if not args.check_only:
        from temper.multipl_e.bench import run_bench
        numbers = [int(x) for x in args.problems.split(",") if x.strip()]
        names = [td["name"] for td in load_problems("humaneval-rs",
                                                    problem_numbers=numbers)]
        missing = [n for n in names if n not in frozen]
        if missing:
            raise SystemExit(f"pre-flight problems have no frozen harness: "
                             f"{missing} (a property cell must not run without "
                             "them)")
        print(f"P-prop-engine pre-flight: {len(numbers)} problems x "
              f"{len(REV2_LADDER)} canvases x {args.steps} steps = "
              f"{envelope} NFE/problem")
        print(f"  problems: {[(n, n_harnesses[n]) for n in names]}")
        print(f"  out: {out}")
        fn = make_steer_sample_fn(
            out=out, envelope_nfe=envelope, steps=args.steps,
            canvas_len=args.canvas, base_seed=0, alg="entropy",
            harnesses=frozen, log=lambda s: print(s, flush=True))
        run_bench("humaneval-rs", n_problems=None, k=1, steps=args.steps,
                  canvas=args.canvas, out=out, sample_fn=fn,
                  problem_numbers=numbers,
                  extra_meta={"mode": "steered", "cell": "P-prop-engine-preflight",
                              "envelope_nfe": envelope,
                              "diversity_ladder": list(REV2_LADDER),
                              "steer_config_sha256": load_verdict_config().sha256,
                              "kill_clone_enabled": False,
                              "emission": EMISSION_PROPERTY},
                  log=lambda s: print(s, flush=True))

    res = check_run(Path(args.check_only) if args.check_only else out,
                    args.steps, envelope, n_harnesses)
    print("\n=== pre-flight checks ===")
    for k, v in res["checks"].items():
        print(f"  {'OK ' if v else 'NO '} {k}")
    for p in res["problems"]:
        print(f"\n  [{p['name']}] nfe={p['row_nfe']} wall={p['row_wall_s']}s "
              f"emit=p{p['emit_particle']} rule_v3={p['emit_rule'] == EMISSION_PROPERTY} "
              f"r3={p['n_r3_events']}/{p['expected_r3_events']} "
              f"share={p['r3_wall_share']} clean={p['final_clean']}")
    ok = all(v for k, v in res["checks"].items() if not k.endswith("error"))
    print(f"\nP-prop-engine PRE-FLIGHT: {'PASS' if ok else 'FAIL'}")
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=1, default=str),
                                   encoding="utf-8")
        print(f"wrote {args.json}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
