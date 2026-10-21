"""temper bench-multipl-e: MultiPL-E generation driver (M2.2, baseline row B2).

Unsteered best-of-K generation on vendored MultiPL-E problems, through the
SAME sampling wrapper (temper.model.dream.sample) and the SAME record format
(schema v1.2) as temper L0 runs — so the M2.0 export/score instrument joins
these records directly:

    temper bench-multipl-e --benchmark humaneval-rs --problems 25 -K 4 \\
        --steps 96 --canvas 384 --out bench/multipl-e/runs/<ts>.jsonl
    temper export-multipl-e --benchmark humaneval-rs --runs <that.jsonl> --out DIR
    temper score-multipl-e --input DIR

What "unsteered" means here (decision doc row B2): NO temper steering —
no verifier rerank, no seed corpus, no L1. The sampler is the model
publisher's own recommended config (model-card example): temperature
samples, remasking alg fixed per run (default "entropy", the card's choice),
top_p 0.95 (dream.py), K sequential canvases, one NFE per executed denoise
step. Sample i of every problem uses seed = base_seed + i.

MultiPL-E completion contract (pinned by tests): the scorer compiles
``prompt + completion + '\\n' + tests``; the vendored prompts end mid-function
(``... -> T {\\n``) and the tests string begins with the function's closing
``}``. The driver therefore instructs the model to emit the COMPLETE function
(signature + body) in ONE rust fence; extraction uses the existing pipeline
(``extract_rust``) and the M2.0 export reduces it to the prompted function's
body fragment (``to_body_fragment``) — never re-stating the signature, never
emitting the fn-closing brace (the tests supply it).

Record format: one JSONL, meta line first, then one candidate row per
(problem, sample) APPENDED AS SAMPLED (crash-safe: finished rows survive a
killed run and ``--resume`` skips them), then a final ``done`` summary line.
Raw model text lives in run-adjacent sidecars ``<out>.raw/<problem>__p<i>.txt``
(sha256-referenced from the row), the per-problem chat prompt in
``<out>.raw/<problem>__prompt.txt``.

No scoring happens here (no rustc, no harness): scoring is the official
container's job, via export-multipl-e + score-multipl-e.
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional

from temper.model.extract import extract_rust
from temper.multipl_e.export import (
    REPO_ROOT,
    _fn_header_line,
    load_benchmark,
)

__all__ = ["INSTRUCTION_TEMPLATE", "build_prompt", "load_problems", "run_bench"]

# The chat instruction wrapping each problem's prompt. Format-only: it says
# nothing about HOW to solve the task (no algorithm hints) — this is the
# unsteered baseline; the exact text is pinned by tests and recorded in the
# run meta (sha) for reproducibility.
INSTRUCTION_TEMPLATE = """\
Continue the following Rust function. You are given the function's \
documentation comments and its exact signature line; the body is missing.

```rust
{prompt}```

Write the COMPLETE function: repeat the signature line exactly as given above \
(same function name, same parameter names and types, same return type), \
followed by an implementation body. Output exactly ONE ```rust fenced code \
block containing only this function and nothing else: no explanations, no \
tests, no `fn main`, no example usage, no additional top-level items.

Rules:
- Use only the Rust standard library; no external crates.
- If you need a helper function, define it INSIDE the function body.
- Do not print anything.
"""


def build_prompt(problem_prompt: str) -> str:
    """Chat instruction for one problem: format rules + the verbatim prompt
    (doc comments + signature ending in ``{``)."""
    return INSTRUCTION_TEMPLATE.format(prompt=problem_prompt)


def load_problems(benchmark: str, n: Optional[int] = None) -> List[dict]:
    """The first n problems of the vendored benchmark, in vendored file order
    (stable across runs — the datasets-server row order)."""
    tasks = load_benchmark(benchmark)
    problems = list(tasks.values())  # dicts keep insertion (file) order
    if n is None:
        return problems
    if n > len(problems):
        raise ValueError(
            f"--problems {n} exceeds {benchmark} size ({len(problems)})")
    return problems[:n]


def _repo_rel(p: Path) -> str:
    try:
        return str(p.relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _iter_record(path: Path):
    """(meta, candidate rows) of a possibly partial bench record."""
    meta, rows = None, []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if obj.get("type") == "run":
            meta = obj
        elif obj.get("type") == "candidate":
            rows.append(obj)
    return meta, rows


def _validate_problem(td: dict) -> str:
    """The prompted function name; raises if the problem shape is not one the
    instrument can export (no fn header in prompt)."""
    m = _fn_header_line(td["prompt"])
    if m is None:
        raise ValueError(
            f"problem {td['name']!r}: no fn header in prompt — the export "
            "instrument cannot join this shape; refusing")
    return m.group(1)


DREAM_MODEL = "Dream-org/Dream-Coder-v0-Instruct-7B"
DREAM_MILESTONE = "M2.2-B2"


def run_bench(
    benchmark: str,
    *,
    n_problems: Optional[int],
    k: int,
    steps: int,
    canvas: int,
    base_seed: int = 0,
    temperature: float = 0.2,
    alg: str = "entropy",
    out: Path | str,
    resume: bool = False,
    sample_fn: Optional[Callable] = None,
    model: Optional[str] = None,
    milestone: Optional[str] = None,
    log: Callable[[str], None] = lambda s: print(s, flush=True),
) -> dict:
    """Unsteered best-of-K generation over the first `n_problems` problems
    (None = all). Writes the run record incrementally; returns a summary.

    resume=True: if `out` exists and its meta matches this invocation's
    config (benchmark, problem list, K, steps, canvas, temperature, alg,
    seed, model, instruction template), already-sampled (problem, sample)
    pairs are skipped and generation continues from the first missing one.

    model/milestone (M2.4b): meta-line labels for alternative engines via
    the sample_fn injection point (e.g. the Qwen AR twin, row B3) — they
    default to the Dream dLLM values. The engine itself is whatever the
    injected sample_fn implements.
    """
    out = Path(out)
    problems = load_problems(benchmark, n_problems)
    # validate every problem shape up front (cheap, before any GPU time)
    fn_names = {}
    for td in problems:
        fn_names[td["name"]] = _validate_problem(td)

    prompts = {td["name"]: build_prompt(td["prompt"]) for td in problems}
    template_sha = _sha256(INSTRUCTION_TEMPLATE)

    done: Dict[tuple, dict] = {}
    if out.exists():
        if not resume:
            raise SystemExit(
                f"refusing to overwrite existing run record {out} "
                "(pass --resume to continue it, or choose a new --out)")
        old_meta, old_rows = _iter_record(out)
        if old_meta is None:
            raise SystemExit(f"{out}: no run meta line — cannot resume")
        expect = {
            "benchmark": benchmark, "K": k, "steps": steps, "canvas": canvas,
            "temperature": temperature, "alg": alg, "base_seed": base_seed,
            "model": model or DREAM_MODEL,
            "instruction_template_sha256": template_sha,
            "n_problems": len(problems),
            "problem_names": [td["name"] for td in problems],
        }
        mismatched = [key for key, val in expect.items() if old_meta.get(key) != val]
        if mismatched:
            raise SystemExit(
                f"{out}: resume config mismatch in {mismatched} — "
                "record was created with different settings")
        for r in old_rows:
            done[(r["problem"], r["i"])] = r
        log(f"resume: {len(done)}/{len(problems) * k} samples already present")
    else:
        if resume:
            raise SystemExit(f"{out}: --resume given but record does not exist")
        out.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "type": "run",
            "schema_version": "1.2",
            "milestone": milestone or DREAM_MILESTONE,
            "mode": "unsteered",
            "benchmark": benchmark,
            "K": k,
            "steps": steps,
            "canvas": canvas,
            "temperature": temperature,
            "alg": alg,
            "base_seed": base_seed,
            "top_p": 0.95,
            "n_problems": len(problems),
            "problem_names": [td["name"] for td in problems],
            "model": model or DREAM_MODEL,
            "instruction_template_sha256": template_sha,
            "utc": datetime.now(timezone.utc).isoformat(),
        }
        with open(out, "w", encoding="utf-8") as f:
            f.write(json.dumps(meta) + "\n")

    raw_dir = Path(str(out) + ".raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    if sample_fn is None:
        # lazy import: keeps this module (and its tests) torch-free
        from temper.model.dream import sample as sample_fn

    t_all = time.perf_counter()
    n_new = 0
    for pidx, td in enumerate(problems):
        name = td["name"]
        if all((name, i) in done for i in range(k)):
            log(f"[{pidx + 1}/{len(problems)}] {name}: already complete, skip")
            continue
        chat_prompt = prompts[name]
        (raw_dir / f"{name}__prompt.txt").write_text(chat_prompt, encoding="utf-8")
        log(f"[{pidx + 1}/{len(problems)}] {name}: K={k} T={temperature} "
            f"alg={alg} steps={steps}")

        def _progress(i, r):
            prefill = getattr(r, "nfe_prefill", None)
            log(f"    sample {i}: wall={r.wall_s}s nfe={r.nfe}"
                + (f" (prefill {prefill})" if prefill is not None else "")
                + f" truncated={r.truncated} err={r.error or 'none'}")

        t0 = time.perf_counter()
        particles = sample_fn(
            chat_prompt,
            canvas_len=canvas,
            steps=steps,
            particles=k,
            temperatures=[temperature] * k,
            algs=[alg] * k,
            base_seed=base_seed,
            progress=_progress,
        )
        gen_wall = time.perf_counter() - t0

        for p in particles:
            if (name, p.index) in done:
                continue
            sidecar = raw_dir / f"{name}__p{p.index}.txt"
            sidecar.write_text(p.raw_text, encoding="utf-8")
            # M2.4b (additive, schema v1.2): engines that render the chat
            # prompt internally (Qwen twin) carry it on the particle — record
            # the rendered prompt verbatim + its sha so the model-visible
            # prompt is pinned in the record, not just the user-turn text.
            rendered = getattr(p, "rendered_prompt", None)
            rendered_sha = _sha256(rendered) if rendered else None
            if rendered:
                (raw_dir / f"{name}__prompt.rendered.txt").write_text(
                    rendered, encoding="utf-8")
            row = {
                "type": "candidate",
                "problem": name,
                "problem_index": pidx,
                "i": p.index,
                "seed": base_seed + p.index,
                "temperature": p.temperature,
                "alg": p.alg,
                "steps": steps,
                "canvas": canvas,
                "nfe": p.nfe,
                "gen_wall_s": p.wall_s,
                "s_per_step": (round(sum(p.per_step_s) / len(p.per_step_s), 3)
                               if p.per_step_s else None),
                "committed_tokens_total": p.committed_tokens_total,
                "mean_commit_conf": p.mean_commit_conf,
                "truncated": p.truncated,
                "gen_error": p.error,
                "prompt_sha256": _sha256(td["prompt"]),
                "raw_path": _repo_rel(sidecar),
                "raw_sha256": _sha256(p.raw_text),
                "repair_attempts": [],
                "repaired": False,
                "prompted_fn": fn_names[name],
            }
            nfe_prefill = getattr(p, "nfe_prefill", None)
            if nfe_prefill is not None:
                row["nfe_prefill"] = nfe_prefill
            n_fwd = getattr(p, "n_forward_total", None)
            if n_fwd is not None:
                row["n_forward_total"] = n_fwd
            if rendered_sha is not None:
                row["rendered_prompt_sha256"] = rendered_sha
            if p.error:
                row["extract_ok"] = False
                row["extract_reason"] = f"skipped (generation error: {p.error})"
                row["code_chars"] = 0
            else:
                ex = extract_rust(p.raw_text, must_define=[fn_names[name]])
                row["extract_ok"] = ex.ok
                row["extract_reason"] = ex.reason
                row["code_chars"] = len(ex.code) if ex.ok else 0
            with open(out, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            done[(name, p.index)] = row
            n_new += 1
        ok = sum(1 for i in range(k)
                 if (name, i) in done and done[(name, i)]['extract_ok'])
        log(f"    problem wall={gen_wall:.1f}s extract_ok={ok}/{k}")

    # ---- final summary line (last one wins when parsing) -------------------
    rows = [done[(td["name"], i)] for td in problems for i in range(k)
            if (td["name"], i) in done]
    total_wall = time.perf_counter() - t_all
    summary = {
        "type": "done",
        "utc": datetime.now(timezone.utc).isoformat(),
        "n_rows": len(rows),
        "nfe_total": sum(r.get("nfe") or 0 for r in rows),
        "nfe_prefill_total": sum(r.get("nfe_prefill") or 0 for r in rows),
        "n_forward_total": sum(r.get("n_forward_total") or 0 for r in rows),
        "gen_wall_s_total": round(sum(r.get("gen_wall_s") or 0.0 for r in rows), 2),
        "driver_wall_s": round(total_wall, 2),
        "extract_ok_rows": sum(1 for r in rows if r.get("extract_ok")),
        "gen_error_rows": sum(1 for r in rows if r.get("gen_error")),
        "truncated_rows": sum(1 for r in rows if r.get("truncated")),
    }
    with open(out, "a", encoding="utf-8") as f:
        f.write(json.dumps(summary) + "\n")

    log(f"bench-multipl-e done: {summary['n_rows']} rows "
        f"({n_new} new), nfe_total={summary['nfe_total']}, "
        f"gen_wall={summary['gen_wall_s_total']}s, "
        f"extract_ok={summary['extract_ok_rows']}/{len(rows)}")
    return {"out": str(out), "summary": summary}
