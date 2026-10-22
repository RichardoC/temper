#!/usr/bin/env python3
"""D4: ghost-closure validity on REAL mid-decode canvases (design §7 D4).

The design's §3.2 negative result came from a SIMULATION (uniform-random hole
pokes over committed canvases): the informed-class R1 rule measured 0-2/44
false kills and true-flag recall of 0-1/36 at <=85% committed, 5/36 at ~92.5%,
11/36 at ~96.4%, 20/36 at full commit. D4 replaces the simulation's assumption
with measurement on canvases captured from the REAL loop, because the real
commit order is confidence-ranked (structure-first), not uniform-random.

Method: run the steer loop with ``capture_checkpoints=True`` on a small
GPU pre-flight (2 problems x 4 canvases is the design's minimum; this script
defaults to the P1-reachable and P1-unreachable pair), save each particle's
canvas AT EVERY CHECKPOINT, then run the R0/R1/R2 ladder on those saved
canvases and report:

  * false-kill rate: informed-class evidence on a canvas whose FINAL canvas is
    container-OK (or which is one of the 4/4 all-clean particles)
  * informed recall curve vs commit fraction: how many of the particles that
    end compile-dead are flagged at >=25/50/75% committed
  * the patience rule the table supports (consecutive checkpoints needed to
    reach a given precision/recall point)

Output is the table that sets D7's patience parameters. Whatever this measures
is what gets committed — the numbers are not adjusted to match the sim.

    # generation (Rancher CLOSED)
    temper/.venv/bin/python scripts/m3_d4_validity.py --capture --json /tmp/d4.json
    # offline re-analysis of a capture (CPU)
    temper/.venv/bin/python scripts/m3_d4_validity.py --analyze /tmp/d4.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from temper.multipl_e.bench import build_prompt, load_problems  # noqa: E402
from temper.steer.view import SteerProblemView  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]

# The design's D4 minimum is 2 problems x 4 canvases. Default to one problem
# the P1 probe found REACHABLE at some temperature and one it did NOT: a
# validity table built only on reachable problems would overstate R1's usable
# recall (there is nothing correct for it to find on an unreachable one).
DEFAULT_PROBLEMS = ("HumanEval_4_mean_absolute_deviation",
                    "HumanEval_0_has_close_elements")


def capture(problems, *, k, steps, canvas, temperature, seed, out_json) -> int:
    """Generate canvases and save every checkpoint's rendered text."""
    import torch
    from temper.model.runtime import get_model
    from temper.steer.loop import _committed_mask_share, _render_canvas, run_game
    from temper.steer.verdict import load_verdict_config

    cfg = load_verdict_config()
    tds = {td["name"]: td for td in load_problems("humaneval-rs")}
    tokenizer, model = get_model()
    payload = {"config": {"k": k, "steps": steps, "canvas": canvas,
                          "temperature": temperature, "seed": seed,
                          "verdict_config_sha256": cfg.sha256},
               "problems": {}}
    for pname in problems:
        td = tds[pname]
        view = SteerProblemView.from_task(td)
        chat_prompt = build_prompt(td["prompt"])
        prompt_len = len(tokenizer.apply_chat_template(
            [{"role": "user", "content": chat_prompt}],
            tokenize=True, add_generation_prompt=True))
        print(f"[{pname}] generating K={k} steps={steps} ...", flush=True)

        # capture by stepping the loop manually in all-off mode so we can
        # snapshot at the design's checkpoint grid
        captured = []
        res = run_game(
            view, chat_prompt, tokenizer=tokenizer, model=model,
            envelope_nfe=steps * k, k=k, steps=steps, canvas_len=canvas,
            temperature=temperature, alg="entropy", base_seed=seed,
            policy_enabled=False, config=cfg,
            log=lambda s: None,
            on_checkpoint=lambda p, x, frac: captured.append({
                "particle": p.index,
                "fraction": round(frac, 4),
                "mask_share": round(_committed_mask_share(x, prompt_len), 4),
                "canvas_text": _render_canvas(tokenizer, x, prompt_len),
            }),
        )
        payload["problems"][pname] = {
            "prompt_len": prompt_len,
            "checkpoints": captured,
            "final": [{"particle": p.index, "canvas_text": p.final_raw_text}
                      for p in sorted(res.particles, key=lambda q: q.index)],
        }
        print(f"[{pname}] captured {len(captured)} checkpoint canvases, "
              f"{len(payload['problems'][pname]['final'])} finals", flush=True)

    Path(out_json).write_text(json.dumps(payload), encoding="utf-8")
    print(f"wrote {out_json}")
    return 0


def analyze(payload: dict, *, stage: str = "stack") -> dict:
    """Run the rung ladder on the captured canvases and build the D4 table."""
    from temper.steer.verdict import (
        ghost_close, informed_fatal_evidence, load_verdict_config,
        r0_shape, r1_typecheck, r2_stub_markers,
    )

    cfg = load_verdict_config()
    tds = {td["name"]: td for td in load_problems("humaneval-rs")}
    per_problem = {}
    for pname, data in payload["problems"].items():
        view = SteerProblemView.from_task(tds[pname])
        # group checkpoints per particle
        by_particle = {}
        for cp in data["checkpoints"]:
            by_particle.setdefault(cp["particle"], []).append(cp)
        rows = []
        for pid, cps in sorted(by_particle.items()):
            cps = sorted(cps, key=lambda c: c["fraction"])
            trace = []
            for cp in cps:
                canvas = cp["canvas_text"]
                r0 = r0_shape(view, canvas, mask_share=cp["mask_share"],
                              expected_mask_share=1.0 - cp["fraction"],
                              checkpoint_fraction=cp["fraction"], config=cfg)
                stubs = r2_stub_markers(canvas.replace("<|mask|>", ""), cfg)
                try:
                    verdict = r1_typecheck(
                        ghost_close(view, canvas),
                        hole_proximity=cfg.hole_proximity_lines)
                    informed = [e.code for e in
                                informed_fatal_evidence(verdict, cfg)]
                except Exception as exc:
                    informed = [f"ERROR:{type(exc).__name__}"]
                trace.append({
                    "fraction": cp["fraction"],
                    "mask_share": cp["mask_share"],
                    "r0_fatal": r0.is_fatal(cfg),
                    "r0_flags": list(r0.flags),
                    "r2_stubs": list(stubs),
                    "informed_codes": informed,
                    "any_error": bool(informed) or None,
                })
            # the FINAL canvas's escalated verdict (any-error, exact)
            final_text = next((f["canvas_text"] for f in data["final"]
                               if f["particle"] == pid), "")
            try:
                fin = r1_typecheck(ghost_close(view, final_text),
                                   hole_proximity=cfg.hole_proximity_lines)
                final_dead = (not fin.ok) or bool(fin.errors)
                final_codes = list(fin.error_codes)
            except Exception as exc:
                final_dead, final_codes = True, [f"ERROR:{type(exc).__name__}"]
            rows.append({"particle": pid, "trace": trace,
                         "final_dead": final_dead,
                         "final_codes": final_codes})
        per_problem[pname] = {"rows": rows}

    # ---- aggregate the D4 table -----------------------------------------
    table = {}
    for pname, data in per_problem.items():
        rows = data["rows"]
        dead = [r for r in rows if r["final_dead"]]
        alive = [r for r in rows if not r["final_dead"]]
        per_frac = {}
        for frac in sorted({t["fraction"] for r in rows for t in r["trace"]}):
            def flagged(r):
                return any(t["informed_codes"] for t in r["trace"]
                           if abs(t["fraction"] - frac) < 1e-9)
            def r0f(r):
                return any(t["r0_fatal"] for t in r["trace"]
                           if abs(t["fraction"] - frac) < 1e-9)
            per_frac[str(frac)] = {
                "n_dead": len(dead), "dead_flagged": sum(map(flagged, dead)),
                "n_alive": len(alive), "alive_flagged": sum(map(flagged, alive)),
                "dead_r0_fatal": sum(map(r0f, dead)),
                "alive_r0_fatal": sum(map(r0f, alive)),
            }
        table[pname] = per_frac
    return {"config": payload.get("config"), "per_problem": table,
            "raw": per_problem}


def report(result: dict, out_json: str | None = None) -> int:
    print("\n=== D4: ghost-closure validity on REAL mid-decode canvases ===")
    print("(informed-class R1 evidence per commit fraction; dead = the "
          "particle's FINAL canvas is compile-dead)")
    all_dead = all_alive = all_df = all_af = 0
    for pname, fracs in result["per_problem"].items():
        print(f"\n  [{pname}]")
        print(f"    {'committed':10s} {'dead':6s} {'flagged':8s} "
              f"{'recall':7s} {'alive':6s} {'false':6s} {'fp_rate':8s}")
        for frac, c in fracs.items():
            rec = (c["dead_flagged"] / c["n_dead"]) if c["n_dead"] else 0.0
            fp = (c["alive_flagged"] / c["n_alive"]) if c["n_alive"] else 0.0
            print(f"    {float(frac):>9.2f} {c['n_dead']:>6d} "
                  f"{c['dead_flagged']:>8d} {rec:>7.2f} {c['n_alive']:>6d} "
                  f"{c['alive_flagged']:>6d} {fp:>8.2f}")
            all_dead += c["n_dead"]; all_df += c["dead_flagged"]
            all_alive += c["n_alive"]; all_af += c["alive_flagged"]
    print(f"\n  totals: dead flagged {all_df}/{all_dead} "
          f"({(all_df / all_dead if all_dead else 0):.2f}), "
          f"alive falsely flagged {all_af}/{all_alive} "
          f"({(all_af / all_alive if all_alive else 0):.2f})")
    if out_json:
        Path(out_json).write_text(json.dumps(result, indent=1), encoding="utf-8")
        print(f"  wrote {out_json}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--capture", action="store_true",
                    help="generate canvases + save checkpoints (GPU; Rancher CLOSED)")
    ap.add_argument("--analyze", default=None,
                    help="re-analyze a capture JSON (CPU only)")
    ap.add_argument("--problems", nargs="+", default=list(DEFAULT_PROBLEMS))
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--steps", type=int, default=256)
    ap.add_argument("--canvas", type=int, default=384)
    ap.add_argument("--temperature", type=float, default=0.8,
                    help="T=0.8: P1 measured this as the lowest REACHABLE band, "
                         "so the capture contains both reachable and dead finals")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    if args.analyze:
        payload = json.loads(Path(args.analyze).read_text(encoding="utf-8"))
        return report(analyze(payload), args.json)
    if not args.capture:
        ap.print_help()
        return 2
    cap = args.json or "/tmp/m3_d4_capture.json"
    rc = capture(args.problems, k=args.k, steps=args.steps, canvas=args.canvas,
                 temperature=args.temperature, seed=args.seed, out_json=cap)
    if rc:
        return rc
    payload = json.loads(Path(cap).read_text(encoding="utf-8"))
    return report(analyze(payload), args.json and args.json.replace(".json", "-table.json"))


if __name__ == "__main__":
    raise SystemExit(main())
