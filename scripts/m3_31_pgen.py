#!/usr/bin/env python3
"""M3.1 PHASE 1 — P-gen: draft the property harness set.

Design §2.1/§3 AS AMENDED by §7 C5: the harness drafter is the AR twin
(Qwen2.5-Coder-7B-Instruct, the M2.4b B3 instrument), drafting **H = 6 property
harnesses per problem** at T = 0.2 into a 256-token canvas. The dLLM could not
author a compiling proptest file at all (measured 0/12 — kept on record in
bench/multipl-e/m3-31/p-gen-dream-negative.json); CANVASES remain
diffusion-drawn in every cell, so the program's substrate claim about STEERING
is untouched.

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
    HARNESS_INSTRUCTION_VERSION as INSTRUCTION_VERSION,
    HARNESS_PROMPT_SHA256,
    build_harness_prompt,
    verify_prompt_sidecar,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

# Design §2.1/§2.2 as amended by §7 C5 (operator ruling 2026-10-24):
#   * generator: the AR twin (Qwen2.5-Coder-7B, the M2.4b B3 instrument), because
#     the dLLM cannot author compiling proptest (measured 0/12 — see
#     bench/multipl-e/m3-31/p-gen-dream-negative.json);
#   * H 2 -> 6, because H=2 is structurally starved at ANY plausible compile
#     rate, and AR drafts cost ~20 s each;
#   * canvas 256, because the AR decode loop emits non-finite logits at 384+ on
#     this prompt (11/25 NaN) and not at 256 (2/25);
#   * one seed-2 replacement round per slot, unchanged.
DRAFT_TEMPERATURE = 0.2
DRAFT_STEPS = 0        # AR has no denoise steps; 0 is the honest record value
DRAFT_SEEDS = (0, 1, 2, 3, 4, 5)
REPLACEMENT_SEED = 2
# The problem cohort: HumanEval NUMBER 0..21, 23, 24, 25 (the vendored file omits
# HumanEval_22) — the same 25 problems as every M3.0 cell.
PROBLEM_NUMBERS = tuple(range(22)) + (23, 24, 25)

DEFAULT_CANVAS = 256


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
    ap.add_argument("--alg", default="ar")
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
            # `model` names the model that AUTHORED the artifacts in this
            # record. On a harness-draft run that is the AR twin, NOT the
            # program's diffusion canvas model — the two must never be
            # ambiguous on the same record.
            "model": "Qwen/Qwen2.5-Coder-7B-Instruct",
            "authored_by": "Qwen/Qwen2.5-Coder-7B-Instruct",
            "canvas_model_note": ("this cell drafts HARNESSES; no canvases are "
                                  "drawn here. Diffusion canvases belong to the "
                                  "selection cells (P-prop-screen / "
                                  "P-prop-engine)."),
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
            "instruction_version": INSTRUCTION_VERSION,
            "harness_generator": "Qwen/Qwen2.5-Coder-7B-Instruct",
            "harness_generator_note": (
                "§7 C5 amendment: harnesses are drafted by the AR twin; CANVASES "
                "remain diffusion-drawn in every cell"),
            "verdict_config_sha256": cfg.sha256,
            "verdict_config_version": cfg.version,
            "input": "problem docstring+signature only (D5 allowlist); the "
                     "MultiPL-E tests string is never in the prompt",
            "note": (f"Design §2.1/§3 P-gen as amended by §7 C5. Rows: "
                     f"kind=first for slots 0..{len(DRAFT_SEEDS)-1} (seeds "
                     f"{list(DRAFT_SEEDS)}), then kind=replacement (seed "
                     f"{REPLACEMENT_SEED}) for slots that failed freeze "
                     f"stub-compile validation. The harness drafter is the AR "
                     f"twin; CANVASES remain diffusion-drawn in every cell."),
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
    from temper.model.qwen import sample as ar_sample

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
        results = ar_sample(
            prompt, canvas_len=args.canvas, steps=args.steps,
            temperature=args.temperature, particles=1, base_seed=seed,
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
            "instruction_version": INSTRUCTION_VERSION,
            "harness_generator": "Qwen/Qwen2.5-Coder-7B-Instruct",
            "harness_generator_note": (
                "§7 C5 amendment: harnesses are drafted by the AR twin; CANVASES "
                "remain diffusion-drawn in every cell"),
            "problem_prompt_sha256": _sha(view.prompt),
            "raw_path": _repo_rel(sidecar),
            "raw_sha256": _sha(p.raw_text),
            "raw_chars": len(p.raw_text),
            # M3.1 C5 decode guards: a retried or fence-cut attempt is visible
            # on the row, so a retry can never look like a clean first attempt.
            "n_attempts": getattr(p, "n_attempts", 1),
            "attempts": getattr(p, "attempts", []),
            "retried": getattr(p, "retried", False),
            "retry_succeeded": getattr(p, "retry_succeeded", False),
            "fence_closed": getattr(p, "fence_closed", False),
            "fence_truncated_chars": getattr(p, "fence_truncated_chars", 0),
            "harness_generator": "Qwen/Qwen2.5-Coder-7B-Instruct",
            "instruction_version": meta["instruction_version"],
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
