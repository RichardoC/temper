#!/usr/bin/env python3
"""P2 equivalence gate: the steer loop in ALL-OFF mode must reproduce
`dream.sample` canvases BIT-IDENTICALLY (design §5 P2, the M2.3 gate pattern).

4 canvases x s128 spot-check on HumanEval_0: run ``dream.sample`` for the
reference sidecars, then run ``temper.steer.loop.run_game`` with
``policy_enabled=False`` at the same config, and compare the decoded canvas
TEXTS byte for byte (plus token ids, which is the stricter check).

This is a GPU pre-flight: one model load, 8 canvases total (~2 x 4 x 128 NFE).
Rancher must be CLOSED (generation phase).

    temper/.venv/bin/python scripts/m3_p2_equivalence.py
    temper/.venv/bin/python scripts/m3_p2_equivalence.py --steps 128 --k 4 \
        --problem HumanEval_0 --json /tmp/p2.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from temper.multipl_e.bench import build_prompt, load_problems  # noqa: E402
from temper.steer.view import SteerProblemView  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--problem", default="HumanEval_0_has_close_elements")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--steps", type=int, default=128,
                    help="s128: the M2.3 gate's spot-check config")
    ap.add_argument("--canvas", type=int, default=384)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    import torch
    from temper.model import dream
    from temper.model.runtime import get_model
    from temper.steer.loop import run_game
    from temper.steer.verdict import load_verdict_config

    problems = {td["name"]: td for td in load_problems("humaneval-rs")}
    td = problems[args.problem]
    view = SteerProblemView.from_task(td)
    chat_prompt = build_prompt(td["prompt"])

    tokenizer, model = get_model()
    prompt_len = len(tokenizer.apply_chat_template(
        [{"role": "user", "content": chat_prompt}],
        tokenize=True, add_generation_prompt=True))

    print(f"P2 equivalence: {args.problem} K={args.k} steps={args.steps} "
          f"canvas={args.canvas} T={args.temperature} seed={args.seed}")

    # ---- reference side: dream.sample (the M2.3-blessed path) -------------
    # temperatures AND algs must be passed explicitly, exactly as the bench
    # driver does. Omitting them drops dream.sample into _normalize_particles'
    # DESIGN CYCLE (T = 0.2/0.4/0.6/0.8, alg = entropy/topk_margin), which is a
    # DIFFERENT sampling config — that mismatch, not an engine divergence, is
    # what made this gate fail 0/4 on its first attempt.
    t0 = time.perf_counter()
    ref = dream.sample(
        chat_prompt, canvas_len=args.canvas, steps=args.steps,
        particles=args.k,
        temperatures=[args.temperature] * args.k,
        algs=["entropy"] * args.k,
        base_seed=args.seed, progress=lambda i, r: None,
    )
    ref_wall = time.perf_counter() - t0
    ref_texts = [r.raw_text for r in ref]
    ref_sha = [hashlib.sha256(t.encode()).hexdigest() for t in ref_texts]
    print(f"  reference: {len(ref)} canvases in {ref_wall:.1f}s "
          f"nfe={[r.nfe for r in ref]} "
          f"temps={[getattr(r, 'temperature', None) for r in ref]} "
          f"algs={[getattr(r, 'alg', None) for r in ref]}")

    # ---- steer-loop side: all-off mode -----------------------------------
    cfg = load_verdict_config()
    t0 = time.perf_counter()
    res = run_game(
        view, chat_prompt, tokenizer=tokenizer, model=model,
        envelope_nfe=args.steps * args.k, k=args.k, steps=args.steps,
        canvas_len=args.canvas, temperature=args.temperature, alg="entropy",
        base_seed=args.seed, policy_enabled=False, config=cfg, log_path=None,
        log=lambda s: None,
    )
    loop_wall = time.perf_counter() - t0
    loop_texts = [p.final_raw_text for p in
                  sorted(res.particles, key=lambda q: q.index)]
    loop_sha = [hashlib.sha256(t.encode()).hexdigest() for t in loop_texts]
    print(f"  steer loop (all-off): {len(loop_texts)} canvases in "
          f"{loop_wall:.1f}s nfe={res.meters.nfe_model}")

    # ---- the gate ---------------------------------------------------------
    identical = [a == b for a, b in zip(ref_sha, loop_sha)]
    n_ids = min(len(ref_texts), len(loop_texts))
    rows = []
    for i in range(n_ids):
        rows.append({
            "particle": i,
            "reference_sha256": ref_sha[i],
            "alloff_sha256": loop_sha[i],
            "text_identical": identical[i],
            "reference_chars": len(ref_texts[i]),
        })
    verdict = all(identical) and len(ref_texts) == len(loop_texts)
    print()
    for r in rows:
        print(f"  p{r['particle']}: {'IDENTICAL' if r['text_identical'] else 'DIFFERS'}"
              f"  ref_sha={r['reference_sha256'][:16]}..  "
              f"loop_sha={r['alloff_sha256'][:16]}..  "
              f"({r['reference_chars']} chars)")
    print(f"\nP2 EQUIVALENCE: {'PASS' if verdict else 'FAIL'} "
          f"({sum(identical)}/{len(ref_texts)} canvases bit-identical)")

    payload = {
        "config": {k: getattr(args, k) for k in
                   ("problem", "k", "steps", "canvas", "temperature", "seed")},
        "prompt_len": prompt_len,
        "reference_wall_s": round(ref_wall, 2),
        "alloff_wall_s": round(loop_wall, 2),
        "alloff_nfe": res.meters.nfe_model,
        "rows": rows,
        "verdict": "PASS" if verdict else "FAIL",
    }
    if args.json:
        Path(args.json).write_text(json.dumps(payload, indent=1), encoding="utf-8")
        print(f"wrote {args.json}")
    return 0 if verdict else 1


if __name__ == "__main__":
    raise SystemExit(main())
