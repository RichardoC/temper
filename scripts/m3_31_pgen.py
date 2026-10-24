#!/usr/bin/env python3
"""M3.1 PHASE 1 — P-gen: draft the property harness set on the diffusion LM.

Design §2.1/§3. The generator is the SAME diffusion substrate as every other
cell in the program (Dream-Coder-v0-Instruct-7B, fixed-shape sampler), drafting
**H = 2 property harnesses per problem** at T = 0.2, steps = 256, seeds 0/1.

What the model sees is the pinned instruction (``scripts/m3_31_harness.py``,
its sha recorded here and in the freeze manifest) wrapped around the problem's
**docstring + signature only**. The MultiPL-E ``tests`` string is never in the
prompt: the text is built from ``SteerProblemView.from_task`` (D5's allowlist),
and the sidecar check proves the problem region matches the banked
``<problem>__prompt.txt`` byte for byte.

Two modes, because the replacement rule is pre-registered but its INPUT is not
knowable before validation runs:

  --draft     seeds 0 and 1 for every problem (the 50 first drafts)
  --replace   seed 2 for an explicit problem:slot list (only the slots whose
              first draft failed freeze validation; §2.2/§7 Q2)

Both write into the same append-only record, so the record IS the evidence of
which draft is which. Rows are crash-safe and ``--resume`` skips completed
(problem, slot, kind) pairs.

    # generation phase (Rancher CLOSED, model resident)
    nohup temper/.venv/bin/python scripts/m3_31_pgen.py --draft \\
        --out bench/multipl-e/runs/<ts>-p-gen.jsonl > /tmp/m3_31_pgen.log 2>&1 &

    # replacement pass for the slots that failed stub-compile validation
    temper/.venv/bin/python scripts/m3_31_pgen.py --replace \\
        --slots HumanEval_7_filter_by_substring:1,... \\
        --out bench/multipl-e/runs/<ts>-p-gen.jsonl --resume
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from temper.multipl_e.bench import load_problems  # noqa: E402
from temper.steer.view import SteerProblemView  # noqa: E402
from scripts.m3_31_harness import (  # noqa: E402
    HARNESS_PROMPT_SHA256,
    build_harness_prompt,
    verify_prompt_sidecar,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

# Design §2.1/§2.2, verbatim.
DRAFT_TEMPERATURE = 0.2
DRAFT_STEPS = 256
DRAFT_SEEDS = (0, 1)
REPLACEMENT_SEED = 2
# The problem cohort: HumanEval NUMBER 0..21, 23, 24, 25 (the vendored file omits
# HumanEval_22) — the same 25 problems as every M3.0 cell.
PROBLEM_NUMBERS = tuple(range(22)) + (23, 24, 25)

DEFAULT_CANVAS = 384


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _repo_rel(p: Path) -> str:
    try:
        return str(p.relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


def _iter_record(path: Path) -> Tuple[Optional[dict], List[dict]]:
    meta, rows = None, []
    if not path.exists():
        return None, rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        o = json.loads(line)
        if o.get("type") == "run":
            meta = o
        elif o.get("type") == "candidate":
            rows.append(o)
    return meta, rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--draft", action="store_true",
                    help="draft seeds 0 and 1 for every problem (the 50 first "
                         "drafts)")
    ap.add_argument("--replace", action="store_true",
                    help="draft seed 2 for the slots named by --slots")
    ap.add_argument("--slots", default="",
                    help="problem:slot[,problem:slot...] for --replace")
    ap.add_argument("--out", required=True)
    ap.add_argument("--canvas", type=int, default=DEFAULT_CANVAS)
    ap.add_argument("--steps", type=int, default=DRAFT_STEPS)
    ap.add_argument("--temperature", type=float, default=DRAFT_TEMPERATURE)
    ap.add_argument("--alg", default="entropy")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-canvases", type=int, default=None,
                    help="stop after N new canvases (smoke runs)")
    args = ap.parse_args(argv)

    if not (args.draft or args.replace):
        print(__doc__)
        return 2

    out = Path(args.out)
    if not out.is_absolute():
        out = REPO_ROOT / out
    raw_dir = Path(str(out) + ".raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    problems = load_problems("humaneval-rs", problem_numbers=list(PROBLEM_NUMBERS))
    views = {td["name"]: SteerProblemView.from_task(td) for td in problems}
    names = [td["name"] for td in problems]

    # ---- sidecar check (input provenance) ---------------------------------
    banked = (REPO_ROOT / "bench" / "multipl-e" / "runs"
              / "20261022T220753Z-c-null.jsonl.raw")
    if banked.is_dir():
        mismatched = [n for n in names
                      if not verify_prompt_sidecar(
                          views[n], banked / f"{n}__prompt.txt")]
        print(f"prompt sidecar check vs {banked.name}: "
              f"{len(names) - len(mismatched)}/{len(names)} identical")
        if mismatched:
            print(f"  !! problem text differs from the banked sidecar for "
                  f"{mismatched} — the drafter would see different problem "
                  "text than the canvas generator saw")
            return 1
    else:
        print(f"  (banked prompt sidecar dir absent: {banked}; "
              "sidecar identity NOT checked)")

    # ---- the work list -----------------------------------------------------
    jobs: List[Tuple[str, int, str, int]] = []
    if args.draft:
        for name in names:
            for slot, seed in enumerate(DRAFT_SEEDS):
                jobs.append((name, slot, "first", seed))
    if args.replace:
        if not args.slots.strip():
            raise SystemExit("--replace needs --slots problem:slot,...")
        for item in args.slots.split(","):
            item = item.strip()
            if not item:
                continue
            name, _, slot_s = item.rpartition(":")
            if name not in views:
                raise SystemExit(f"--slots: unknown problem {name!r}")
            jobs.append((name, int(slot_s), "replacement", REPLACEMENT_SEED))

    # ---- record meta -------------------------------------------------------
    meta, done_rows = (None, []) if not args.resume else _iter_record(out)
    if out.exists() and not args.resume:
        raise SystemExit(
            f"refusing to overwrite existing record {out} (pass --resume)")
    done = {(r["problem"], int(r["i"]), r.get("kind", "first")) for r in done_rows}
    if meta is None:
        from temper.steer.verdict import load_verdict_config
        cfg = load_verdict_config()
        meta = {
            "type": "run",
            "utc": datetime.now(timezone.utc).isoformat(),
            "benchmark": "humaneval-rs",
            "cell": "P-gen",
            "milestone": "M3.1-P1",
            "model": "Dream-org/Dream-Coder-v0-Instruct-7B",
            "mode": "harness-draft",
            "problem_numbers": list(PROBLEM_NUMBERS),
            "problem_names": names,
            "n_problems": len(names),
            "steps": args.steps,
            "canvas": args.canvas,
            "temperature": args.temperature,
            "alg": args.alg,
            "draft_seeds": list(DRAFT_SEEDS),
            "replacement_seed": REPLACEMENT_SEED,
            "H": len(DRAFT_SEEDS),
            "draft_prompt_sha256": HARNESS_PROMPT_SHA256,
            "verdict_config_sha256": cfg.sha256,
            "verdict_config_version": cfg.version,
            "input": "problem docstring+signature only (D5 allowlist); the "
                     "MultiPL-E tests string is never in the prompt",
            "note": "Design §2.1/§3 P-gen. Rows: kind=first (slot 0/1, seeds "
                    "0/1) then kind=replacement (seed 2) for slots that failed "
                    "freeze stub-compile validation.",
        }
        with open(out, "w", encoding="utf-8") as f:
            f.write(json.dumps(meta) + "\n")
        print(f"meta written: {out}")
    else:
        for key in ("draft_prompt_sha256", "steps", "canvas", "temperature",
                    "alg", "problem_names"):
            got = meta.get(key)
            want = {"draft_prompt_sha256": HARNESS_PROMPT_SHA256,
                    "steps": args.steps, "canvas": args.canvas,
                    "temperature": args.temperature, "alg": args.alg,
                    "problem_names": names}[key]
            if got != want:
                raise SystemExit(
                    f"{out}: resume mismatch in {key} ({got!r} != {want!r})")

    # ---- generate ----------------------------------------------------------
    from temper.model.dream import sample as dream_sample

    nfe_total_planned = len(jobs) * args.steps
    print(f"P-gen: {len(jobs)} canvases x {args.steps} steps "
          f"= {nfe_total_planned} NFE")
    print(f"  mode={'draft' if args.draft else ''}"
          f"{'+replace' if args.draft and args.replace else ''}"
          f"{'replace' if args.replace and not args.draft else ''}")
    print(f"  draft prompt sha256: {HARNESS_PROMPT_SHA256}")
    print(f"  out: {out}")

    t_all = time.perf_counter()
    n_new = 0
    for name, slot, kind, seed in jobs:
        if (name, slot, kind) in done:
            print(f"  [skip] {name} slot {slot} {kind} (already recorded)")
            continue
        if args.max_canvases is not None and n_new >= args.max_canvases:
            print(f"  --max-canvases {args.max_canvases} reached; stopping")
            break
        view = views[name]
        prompt = build_harness_prompt(view)
        print(f"[{n_new + 1}/{len(jobs)}] {name} slot={slot} kind={kind} "
              f"seed={seed} T={args.temperature} steps={args.steps}",
              flush=True)
        t0 = time.perf_counter()
        results = dream_sample(
            prompt, canvas_len=args.canvas, steps=args.steps,
            temperature=args.temperature, alg=args.alg, particles=1,
            base_seed=seed,
        )
        p = results[0]
        wall = time.perf_counter() - t0
        tag = "first" if kind == "first" else "replacement"
        sidecar = raw_dir / f"{name}__p{slot}__{tag}.txt"
        sidecar.write_text(p.raw_text, encoding="utf-8")
        row = {
            "type": "candidate",
            "problem": name,
            "i": slot,
            "kind": kind,
            "seed": seed,
            "temperature": p.temperature,
            "alg": p.alg,
            "steps": args.steps,
            "canvas": args.canvas,
            "nfe": p.nfe,
            "gen_wall_s": round(p.wall_s, 2),
            "driver_wall_s": round(wall, 2),
            "committed_tokens_total": p.committed_tokens_total,
            "mean_commit_conf": p.mean_commit_conf,
            "truncated": p.truncated,
            "gen_error": p.error,
            "prompt_sha256": _sha(prompt),
            "draft_prompt_sha256": HARNESS_PROMPT_SHA256,
            "problem_prompt_sha256": _sha(view.prompt),
            "raw_path": _repo_rel(sidecar),
            "raw_sha256": _sha(p.raw_text),
            "raw_chars": len(p.raw_text),
        }
        with open(out, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        done.add((name, slot, kind))
        n_new += 1
        print(f"    wall={row['gen_wall_s']}s nfe={p.nfe} "
              f"truncated={p.truncated} chars={len(p.raw_text)} "
              f"err={p.error}", flush=True)

    # ---- summary -----------------------------------------------------------
    meta2, rows = _iter_record(out)
    summary = {
        "type": "done",
        "utc": datetime.now(timezone.utc).isoformat(),
        "n_rows": len(rows),
        "n_first": sum(1 for r in rows if r.get("kind") == "first"),
        "n_replacement": sum(1 for r in rows if r.get("kind") == "replacement"),
        "nfe_total": sum(r.get("nfe") or 0 for r in rows),
        "gen_wall_s_total": round(sum(r.get("gen_wall_s") or 0.0
                                      for r in rows), 2),
        "driver_wall_s": round(time.perf_counter() - t_all, 2),
        "truncated_rows": sum(1 for r in rows if r.get("truncated")),
        "gen_error_rows": sum(1 for r in rows if r.get("gen_error")),
        "draft_prompt_sha256": HARNESS_PROMPT_SHA256,
        "note": "harness NFE is infrastructure, reported on its own line and "
                "never inside a selection cell's 2048 NFE/problem envelope "
                "(design §5)",
    }
    with open(out, "a", encoding="utf-8") as f:
        f.write(json.dumps(summary) + "\n")
    print(f"\nP-gen done: {summary['n_rows']} rows ({n_new} new), "
          f"nfe_total={summary['nfe_total']}, "
          f"wall={summary['gen_wall_s_total']}s "
          f"({summary['gen_wall_s_total'] / 3600:.2f} h)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
