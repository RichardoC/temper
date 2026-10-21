#!/usr/bin/env python3
"""D1: verifier-rung latency bench (M3.0, docs/m3-design.md §7 D1).

CPU-only, committed artifacts only (no GPU, no container). Measures, on the
raw sidecars of the four corrected-basis dLLM cells (s96/s128/s256/s512):

  R0   shape/fence python pass                     -- per checkpoint
  R1   ghost-closure + stable-rustc lib type-check -- per checkpoint, per
       particle (measured at the FINAL checkpoint, i.e. zero holes, and at
       simulated mid-canvas hole densities so the cost is shown to be
       spawn-dominated rather than error-class dominated)
  R2   committed stub-marker scan                  -- free
  tk   tokenizer decode per checkpoint render
  R3   M0 harness cycle on a completed fragment    -- sampled (expensive)

Then it prints the amortization the design's kill-criterion 4 depends on:
R1 wall as a fraction of the canvas wall at the 1024 and 2048 envelopes
(1.03 s/NFE, M2.4a/b least-squares fit).

    temper/.venv/bin/python scripts/m3_verifier_latency.py            # all cells
    temper/.venv/bin/python scripts/m3_verifier_latency.py --cells s256
    temper/.venv/bin/python scripts/m3_verifier_latency.py --json out.json

The R3 sample is deliberately small (default 4 rows) and runs the temper
bench task's own approved properties through the M0 harness — the design's
"R3 harness cycle on completed fragments (236 ms workload)".

Committed latency table: docs/m3.0-instruments.md (this script's output).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from temper.multipl_e.bench import load_problems  # noqa: E402
from temper.steer.verdict import (  # noqa: E402
    final_checkpoint_clean,
    ghost_close,
    informed_fatal_evidence,
    load_verdict_config,
    r0_shape,
    r1_typecheck,
    r2_stub_markers,
)
from temper.steer.view import SteerProblemView  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS = REPO_ROOT / "bench" / "multipl-e" / "runs"

# Corrected-basis dLLM cells: label -> (run record, corrected norepair results
# JSON). Sidecars live at <run>.raw/; the results JSON supplies the container
# ground truth for the instrument-validity cross-check at the end of each cell.
CELLS = {
    "s96": ("20261019T103020Z-b2pilot.jsonl", "20261021T144931Z.json"),
    "s128": ("20261019T183015Z-s128.jsonl", "20261021T145038Z.json"),
    "s256": ("20261020T100251Z-s256.jsonl", "20261021T144401Z.json"),
    "s512": ("20261020T225623Z-s512.jsonl", "20261021T145224Z.json"),
}

# Canvas wall per NFE (M2.4a/b least-squares fit on this host).
S_PER_NFE = 1.035
ENVELOPES = (1024, 2048)
K_MAX_EFF = 8
CHECKPOINTS = 4

_WORD_RE = re.compile(r"\S+")


def simulate_partial(text: str, committed_fraction: float, *, seed: int,
                     hole_pool: float = 0.30) -> str:
    """A deterministic mid-canvas rendering: poke `<|mask|>` over word units.

    The design's §3.2 pilot masked a fixed 30% word-unit pool and revealed
    nested subsets as the canvas fills; we reproduce the SHAPE (hole pool size,
    committed fraction) because only the R1 cost is being measured here — the
    validity curve on real canvases is D4.
    """
    words = _WORD_RE.findall(text)
    if not words:
        return text
    rng = random.Random(seed)
    pool = max(1, int(len(words) * hole_pool))
    targets = rng.sample(range(len(words)), min(pool, len(words)))
    n_holes = int(round(pool * (1.0 - committed_fraction)))
    holes = set(targets[:n_holes])
    out, idx = [], 0
    for m in _WORD_RE.finditer(text):
        out.append(text[idx:m.start()])
        out.append("<|mask|>" if len(out) // 2 in holes else m.group(0))
        idx = m.end()
    out.append(text[idx:])
    return "".join(out)


def pct(vals, q):
    if not vals:
        return None
    return round(statistics.quantiles(vals, n=100)[int(q * 100) - 1], 3) \
        if len(vals) >= 2 else round(vals[0], 3)


def summarize(vals):
    s = sorted(vals)
    return {
        "n": len(vals),
        "p50": round(statistics.median(vals), 3) if vals else None,
        "p95": round(s[min(len(s) - 1, int(round(0.95 * (len(s) - 1))))], 3)
        if s else None,
        "max": round(s[-1], 3) if s else None,
        "mean": round(sum(vals) / len(vals), 3) if vals else None,
    }


def load_sidecars(run_name: str):
    """(problem_name, raw text) for every *.raw/*__p<i>.txt of one cell."""
    raw_dir = RUNS / (run_name + ".raw")
    out = []
    for p in sorted(raw_dir.glob("*__p[0-9].txt")):
        problem = p.name.split("__p")[0]
        out.append((problem, p.read_text(encoding="utf-8")))
    return out


def container_statuses(results_json: str) -> dict:
    """{problem: [container statuses]} from one committed results JSON."""
    d = json.loads((REPO_ROOT / "bench" / "multipl-e" / "results" /
                    results_json).read_text(encoding="utf-8"))
    return {p["name"]: p["statuses_official"] for p in d["per_problem"]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cells", nargs="+", default=sorted(CELLS),
                    choices=sorted(CELLS))
    ap.add_argument("--json", default=None, help="also write raw stats here")
    ap.add_argument("--r3-sample", type=int, default=4,
                    help="rows sampled for the R3 (M0 harness) cycle; 0 disables")
    ap.add_argument("--skip-tokenizer", action="store_true",
                    help="skip the tokenizer decode timing (still CPU-only)")
    args = ap.parse_args(argv)

    cfg = load_verdict_config()
    problems = {td["name"]: td for td in load_problems("humaneval-rs")}
    views = {n: SteerProblemView.from_task(td) for n, td in problems.items()}

    report = {"verdict_config_sha256": cfg.sha256,
              "verdict_config_version": cfg.version,
              "s_per_nfe": S_PER_NFE, "cells": {}}

    for label in args.cells:
        run_name, results_name = CELLS[label]
        rows = load_sidecars(run_name)
        print(f"\n=== cell {label} ({run_name}): {len(rows)} raw sidecars ===")
        r0_wall, r1_wall, r1_holes, r1_codes = [], [], [], []
        r1_verdicts = []
        r1_exact = []
        stub_hits = 0
        for i, (pname, raw) in enumerate(rows):
            view = views.get(pname)
            if view is None:
                continue
            # ---- R0 (final checkpoint: no residual mask share) -------------
            t0 = time.perf_counter()
            r0_shape(view, raw, mask_share=0.0, expected_mask_share=0.0,
                     checkpoint_fraction=1.0, config=cfg)
            r0_wall.append((time.perf_counter() - t0) * 1000.0)
            # ---- R2 ---------------------------------------------------------
            if r2_stub_markers(raw, cfg):
                stub_hits += 1
            # ---- R1 at the FINAL checkpoint (zero holes) -------------------
            ghost = ghost_close(view, raw)
            v = r1_typecheck(ghost, hole_proximity=cfg.hole_proximity_lines)
            r1_verdicts.append(v)
            r1_wall.append(v.wall_s * 1000.0)
            r1_codes.extend(v.error_codes)
            r1_exact.append(ghost.fragment_exact)
            # ---- R1 mid-canvas (simulated 30% hole pool, 50% committed) ----
            if i % 2 == 0:  # every other row keeps the sweep cheap
                partial = simulate_partial(raw, 0.5, seed=i)
                g2 = ghost_close(view, partial)
                v2 = r1_typecheck(g2, hole_proximity=cfg.hole_proximity_lines)
                r1_holes.append(v2.wall_s * 1000.0)

        # ---- instrument validity: R1-final vs container ground truth ------
        # Design §3.2: at the FINAL checkpoint (zero holes) R1 is EXACT on the
        # measured corpus. Report the confusion per CONTAINER STATUS CLASS so
        # the claim is checkable, not aggregate: the any-error rule must catch
        # every SyntaxError (compile-dead) row and flag none of the OK rows;
        # Exception rows compile and panic, so a type-check correctly CLEARS
        # them (they are R3/held-out territory, not R1's).
        statuses = container_statuses(results_name)
        confusion = {}
        for (pname, _raw), v in zip(rows, r1_verdicts):
            st = statuses.get(pname) or []
            if not st:
                cls = "excluded"
            elif all(s == "OK" for s in st):
                cls = "OK"
            elif any(s == "SyntaxError" for s in st):
                cls = "SyntaxError"
            elif any(s == "Exception" for s in st):
                cls = "Exception"
            else:
                cls = "+".join(sorted(set(st)))
            bucket = confusion.setdefault(cls, {
                "n": 0, "any_error_flagged": 0, "informed_flagged": 0})
            bucket["n"] += 1
            bucket["any_error_flagged"] += (not final_checkpoint_clean(v))
            bucket["informed_flagged"] += bool(informed_fatal_evidence(v, cfg))

        cell = {
            "n_rows": len(r0_wall),
            "r0_ms": summarize(r0_wall),
            "r1_final_ms": summarize(r1_wall),
            "r1_midcanvas_ms": summarize(r1_holes),
            "r2_stub_hits": stub_hits,
            "r1_error_code_histogram": {c: r1_codes.count(c)
                                        for c in sorted(set(r1_codes))},
            "r1_fragment_exact_rows": sum(1 for e in r1_exact if e),
            "r1_final_confusion_by_status": confusion,
        }
        print(f"R0  shape        : {cell['r0_ms']}")
        print(f"R1  final (0 holes): {cell['r1_final_ms']}")
        print(f"R1  mid (30% holes): {cell['r1_midcanvas_ms']}")
        print(f"R2  stub hits      : {stub_hits}/{len(r0_wall)}")
        print(f"R1  exact fragments: {cell['r1_fragment_exact_rows']}/{len(r0_wall)}")
        for cls in sorted(confusion):
            b = confusion[cls]
            print(f"R1  final x {cls:12s}: n={b['n']:3d} "
                  f"any_error_flagged={b['any_error_flagged']:3d} "
                  f"informed_flagged={b['informed_flagged']:3d}")
        print(f"R1  error codes    : {cell['r1_error_code_histogram']}")
        report["cells"][label] = cell

    # ---- tokenizer decode cost per checkpoint render ----------------------
    if not args.skip_tokenizer:
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(
                "Dream-org/Dream-Coder-v0-Instruct-7B", trust_remote_code=True)
            prompt_toks = tok("x" * 1000).input_ids[:254]
            canvas_toks = prompt_toks + [151666] * 384
            walls = []
            for _ in range(20):
                t0 = time.perf_counter()
                tok.decode(canvas_toks, skip_special_tokens=True)
                walls.append((time.perf_counter() - t0) * 1000.0)
            report["tokenizer_decode_ms"] = summarize(walls)
            print(f"\ntokenizer decode (prompt+384 canvas): "
                  f"{report['tokenizer_decode_ms']}")
        except Exception as exc:  # loud but non-fatal: it is a measurement
            report["tokenizer_decode_ms"] = {"error": repr(exc)}
            print(f"\ntokenizer decode: SKIPPED ({exc})")

    # ---- R3 sample: M0 harness cycle on a completed fragment --------------
    if args.r3_sample > 0:
        try:
            from temper.harness.runner import run_candidate
            task_dir = REPO_ROOT / "bench" / "sort"
            props = (task_dir / "properties.rs").read_text(encoding="utf-8")
            cand = (task_dir / "candidate.example.rs").read_text(encoding="utf-8")
            walls = []
            for _ in range(args.r3_sample):
                out = run_candidate(cand, props, cases=256, timeout_ms=2000)
                walls.append(out.compile.duration_s * 1000.0
                             + (out.test.duration_s * 1000.0 if out.test else 0.0))
            report["r3_cycle_ms"] = summarize(walls)
            print(f"R3  M0 harness cycle (n={args.r3_sample}): "
                  f"{report['r3_cycle_ms']}")
        except Exception as exc:
            report["r3_cycle_ms"] = {"error": repr(exc)}
            print(f"R3: SKIPPED ({exc})")

    # ---- amortization vs canvas wall (kill criterion 4) -------------------
    r1_p50 = max((report["cells"][c]["r1_final_ms"]["p50"] or 0.0)
                 for c in report["cells"]) if report["cells"] else 0.0
    amort = {}
    for env in ENVELOPES:
        canvas_wall_s = env * S_PER_NFE
        r1_total_s = CHECKPOINTS * K_MAX_EFF * r1_p50 / 1000.0
        amort[env] = {
            "canvas_wall_s": round(canvas_wall_s, 1),
            "r1_total_s_worst_case": round(r1_total_s, 3),
            "r1_fraction": round(r1_total_s / canvas_wall_s, 6),
            "r1_fraction_pct": round(100.0 * r1_total_s / canvas_wall_s, 4),
        }
    report["amortization"] = amort
    print("\n=== R1 amortization (K_max_eff 8 x 4 checkpoints vs canvas wall) ===")
    for env, a in amort.items():
        print(f"  envelope {env:5d}: R1 total {a['r1_total_s_worst_case']:7.3f}s "
              f"of {a['canvas_wall_s']:8.1f}s canvas wall = "
              f"{a['r1_fraction_pct']:.4f}%  (design cap 10%, D1 acceptance 1%)")

    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=1), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
