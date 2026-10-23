#!/usr/bin/env python3
"""C-full@2048 GPU PRE-FLIGHT: verify the in-loop engine end-to-end before
committing 14.9 h to it.

The C-null cell is the unsteered path (already proven by P2's bit-identity
gate). C-full is the STEERED path, and its own failure modes are not the same:
the ladder must reach every particle, the demoted role must fire the ladder at
the final checkpoint only, the emission rule must rank over the game's eight
canvases, the game's NFE must respect the envelope, and the D2 steer log must
replay. None of that has ever run on the real model — the CPU suite uses a fake
model whose canvases are nonsense.

So: two problems x the full 8-slot ladder x a short denoise schedule, through
the EXACT seam the cell uses (``temper.steer.engine.make_steer_sample_fn`` ->
``run_bench`` -> run record), then every invariant checked against the record
and the steer logs. A pre-flight failure here costs minutes; the same failure
inside the cell costs a day.

    # generation phase (Rancher CLOSED)
    temper/.venv/bin/python scripts/m3_p3_cfull_preflight.py --steps 32
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from temper.multipl_e.bench import load_problems  # noqa: E402
from temper.steer.engine import REV2_LADDER, make_steer_sample_fn  # noqa: E402
from temper.steer.verdict import load_verdict_config  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]

# Two problems the P1 probe measured REACHABLE at high temperature: the ladder
# has something to find, so the emission rule gets a real choice to make.
DEFAULT_PROBLEMS = (17, 11)


def read_log(path: Path) -> list:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def check_run(out_jsonl: Path, steps: int, envelope: int) -> dict:
    """Every invariant the 14.9 h cell depends on, read back from the record."""
    from temper.multipl_e.bench import _iter_record
    meta, rows = _iter_record(out_jsonl)
    n_ladder = len(REV2_LADDER)
    checks = {}
    problems = []
    for row in rows:
        name = row["problem"]
        log_path = Path(str(out_jsonl) + ".steer") / f"{name}.jsonl"
        checks[f"{name}:steer_log_exists"] = log_path.exists()
        if not log_path.exists():
            continue
        events = read_log(log_path)
        kinds = [e["kind"] for e in events]
        emit = next((e for e in events if e["kind"] == "emit"), None)
        checkpoints = [e for e in events if e["kind"] == "checkpoint"]
        meta_ev = events[0]
        c = {
            "name": name,
            "row_nfe": row["nfe"],
            "row_temp": row["temperature"],
            "row_extract_ok": row["extract_ok"],
            "row_has_raw": bool(row.get("raw_path")),
            "log_verdict_config_sha256": meta_ev.get("verdict_config_sha256"),
            "log_temperatures": meta_ev.get("temperatures"),
            "log_kill_clone_enabled": meta_ev.get("kill_clone_enabled"),
            "n_checkpoints": len(checkpoints),
            "checkpoint_steps": sorted({e["step"] for e in checkpoints}),
            "n_kills": kinds.count("kill"),
            "n_kill_suppressed": kinds.count("kill_suppressed"),
            "emit_particle": (emit or {}).get("particle"),
            "emit_final_clean": (emit or {}).get("final_clean"),
            "emit_n_temps": len((emit or {}).get("temperatures", {}) or {}),
            "emit_ranking": (emit or {}).get("screen_ranking"),
            "n_particles_spawned": kinds.count("spawn"),
            "done_meters": next((e.get("meters") for e in events
                                 if e["kind"] == "done"), None),
        }
        problems.append(c)

        checks[f"{name}:ladder_reached_the_game"] = (
            c["log_temperatures"] == list(REV2_LADDER))
        checks[f"{name}:all_eight_particles_spawned"] = (
            c["n_particles_spawned"] == n_ladder)
        checks[f"{name}:config_sha_recorded"] = (
            c["log_verdict_config_sha256"] == load_verdict_config().sha256)
        checks[f"{name}:demoted_role_recorded"] = (
            c["log_kill_clone_enabled"] is False)
        checks[f"{name}:ladder_fired_at_final_checkpoint_only"] = (
            c["checkpoint_steps"] == [steps])
        checks[f"{name}:no_kill_executed"] = c["n_kills"] == 0
        checks[f"{name}:nfe_within_envelope"] = (
            c["row_nfe"] is not None and c["row_nfe"] <= envelope)
        checks[f"{name}:emitted_one_canvas"] = c["emit_particle"] is not None
        checks[f"{name}:emission_saw_the_whole_ladder"] = (
            c["emit_n_temps"] == n_ladder)
        checks[f"{name}:row_carried_the_emitted_canvas"] = c["row_has_raw"]
        checks[f"{name}:verifier_metered"] = (
            c["done_meters"] is not None
            and c["done_meters"].get("nfe_verifier_calls", 0) >= n_ladder)
        # D2: the log must replay with no GPU
        try:
            from temper.steer.log import read_game_log, replay_game
            replay_game(read_game_log(log_path), load_verdict_config())
            checks[f"{name}:steer_log_replays"] = True
        except Exception as exc:
            checks[f"{name}:steer_log_replays"] = False
            checks[f"{name}:replay_error"] = repr(exc)

    checks["record_mode_is_steered"] = (meta or {}).get("mode") == "steered"
    checks["record_carries_config_sha"] = (
        (meta or {}).get("steer_config_sha256")
        == load_verdict_config().sha256)
    checks["record_carries_the_ladder"] = (
        (meta or {}).get("diversity_ladder") == list(REV2_LADDER))
    checks["one_row_per_problem"] = len(rows) == len(problems)
    return {"checks": checks, "problems": problems, "meta": meta,
            "rows": [{k: r[k] for k in
                      ("problem", "nfe", "temperature", "extract_ok",
                       "mean_commit_conf")} for r in rows]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=32,
                    help="short denoise schedule: the pre-flight checks the "
                         "CONTROL FLOW, not canvas quality")
    ap.add_argument("--canvas", type=int, default=384)
    ap.add_argument("--problems", default=",".join(str(n) for n in DEFAULT_PROBLEMS))
    ap.add_argument("--out", default=None)
    ap.add_argument("--json", default=None)
    ap.add_argument("--check-only", default=None,
                    help="skip generation; check an existing run record")
    args = ap.parse_args(argv)

    envelope = args.steps * len(REV2_LADDER)
    out = Path(args.out) if args.out else Path(
        tempfile.mkdtemp()) / "c-full-preflight.jsonl"

    if not args.check_only:
        from temper.multipl_e.bench import run_bench
        numbers = [int(x) for x in args.problems.split(",") if x.strip()]
        names = [td["name"] for td in load_problems("humaneval-rs",
                                                    problem_numbers=numbers)]
        print(f"C-full pre-flight: {len(numbers)} problems x {len(REV2_LADDER)} "
              f"canvases x {args.steps} steps = {envelope} NFE/problem")
        print(f"  problems: {names}")
        print(f"  out: {out}")
        fn = make_steer_sample_fn(
            out=out, envelope_nfe=envelope, steps=args.steps,
            canvas_len=args.canvas, base_seed=0, alg="entropy",
            log=lambda s: print(s, flush=True))
        run_bench("humaneval-rs", n_problems=None, k=1, steps=args.steps,
                  canvas=args.canvas, out=out, sample_fn=fn,
                  problem_numbers=numbers,
                  extra_meta={"mode": "steered", "cell": "C-full-preflight",
                              "envelope_nfe": envelope,
                              "diversity_ladder": list(REV2_LADDER),
                              "steer_config_sha256":
                                  load_verdict_config().sha256,
                              "kill_clone_enabled": False},
                  log=lambda s: print(s, flush=True))

    res = check_run(out, args.steps, envelope)
    print("\n=== pre-flight checks ===")
    for k, v in res["checks"].items():
        print(f"  {'OK ' if v else 'NO '} {k}")
    for p in res["problems"]:
        print(f"\n  [{p['name']}] nfe={p['row_nfe']} temp={p['row_temp']} "
              f"extract_ok={p['row_extract_ok']} emit=p{p['emit_particle']} "
              f"final_clean={p['emit_final_clean']} "
              f"ranking={p['emit_ranking']} killed={p['n_kills']} "
              f"suppressed={p['n_kill_suppressed']}")
    ok = all(v for k, v in res["checks"].items() if not k.endswith("error"))
    print(f"\nC-full PRE-FLIGHT: {'PASS' if ok else 'FAIL'}")
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=1, default=str),
                                   encoding="utf-8")
        print(f"wrote {args.json}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
