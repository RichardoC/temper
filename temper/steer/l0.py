"""L0 steering: best-of-K sampling + verifier reranking (design §1/§5/§6).

Per task:
  1. load task.toml + approved properties.rs,
  2. K sequential Dream-Coder canvases (model-side wall time each),
  3. extract Rust (temper.model.extract),
  4. M0 harness run per candidate (seatbelt default; harness-side wall time),
  5. score = 100*1[compiles] + 1000*1[tests pass] - 5*reject_fraction,
     tie-break: higher mean commit confidence, then lower index,
  6. JSONL run record `bench/<task>/runs/<ts>.jsonl` (one row per candidate),
  7. failing seeds persisted into `bench/<task>/corpus/<strategy_hash>/`.

M0 harness API untouched — this module is a pure consumer of
``temper.harness.runner.run_candidate``.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from temper.harness.runner import run_candidate
from temper.model.dream import sample
from temper.model.extract import extract_rust

__all__ = ["CandidateRow", "load_task", "run_l0", "eval_runs"]

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_toml(path: Path) -> dict:
    try:
        import tomllib  # py3.11+
    except ImportError:
        import tomli as tomllib  # venv dep, py3.9
    with open(path, "rb") as f:
        return tomllib.load(f)


@dataclass
class Task:
    name: str
    dir: Path
    signature: str
    doc: str
    prompt_template: str
    must_define: List[str]
    properties_approved: bool
    properties_source: str
    strategy_hash: str

    def build_prompt(self) -> str:
        return self.prompt_template.format(signature=self.signature, doc=self.doc)


def load_task(task_dir: Path | str) -> Task:
    task_dir = Path(task_dir)
    if not task_dir.is_absolute():
        task_dir = Path.cwd() / task_dir
    meta = _load_toml(task_dir / "task.toml")["task"]
    props_path = task_dir / "properties.rs"
    if not props_path.exists():
        raise FileNotFoundError(f"{props_path} missing — run 'temper draft' first")
    props = props_path.read_text(encoding="utf-8")
    approved = bool(meta.get("properties_approved", False)) or bool(
        _load_toml(task_dir / "task.toml").get("properties", {}).get("approved", False)
    )
    if not approved:
        raise PermissionError(
            f"{task_dir}: properties are NOT approved "
            "(task.toml properties.approved != true) — human sign-off required"
        )
    import hashlib

    return Task(
        name=meta["name"],
        dir=task_dir,
        signature=meta["signature"],
        doc=meta.get("doc", ""),
        prompt_template=meta["prompt_template"],
        must_define=list(meta.get("must_define", [])),
        properties_approved=True,
        properties_source=props,
        strategy_hash=hashlib.sha256(props.encode("utf-8")).hexdigest(),
    )


def score_of(compiles: bool, tests_pass: bool, report: Optional[dict]) -> float:
    s = 0.0
    if compiles:
        s += 100.0
    if tests_pass:
        s += 1000.0
    if report:
        succ = int(report.get("successes", 0))
        rej = int(report.get("local_rejects", 0)) + int(report.get("global_rejects", 0))
        denom = max(1, succ + rej)
        s -= 5.0 * (rej / denom)
    return round(s, 2)


@dataclass
class CandidateRow:
    i: int
    temperature: float
    alg: str
    seed: int
    gen_wall_s: float
    s_per_step: Optional[float]
    committed_tokens_total: int
    mean_commit_conf: Optional[float]
    truncated: bool
    gen_error: Optional[str]
    extract_ok: bool
    extract_reason: str
    code_chars: int
    compile_status: str
    diag_codes: List[str] = field(default_factory=list)
    diag_spans: List[dict] = field(default_factory=list)
    test_status: str = "not_attempted"
    minimal_input: Optional[str] = None
    failing_seed: Optional[str] = None
    successes: Optional[int] = None
    rejects: Optional[int] = None
    reason: Optional[str] = None
    harness_wall_s: Optional[float] = None
    score: float = 0.0
    selected: bool = False


def _diag_summary(outcome) -> (List[str], List[dict]):
    codes, spans = [], []
    for d in outcome.compile.diagnostics or []:
        if d.code and d.code not in codes:
            codes.append(d.code)
        for sp in getattr(d, "spans", []) or []:
            if getattr(sp, "is_primary", False):
                spans.append({
                    "file": getattr(sp, "file", None) or "src/candidate.rs",
                    "line_start": getattr(sp, "line_start", None),
                    "line_end": getattr(sp, "line_end", None),
                    "is_primary": True,
                })
    return codes, spans[:4]


def _persist_failing_seed(task: Task, row: CandidateRow, run_ts: str) -> Optional[str]:
    if row.failing_seed is None:
        return None
    corpus = task.dir / "corpus" / task.strategy_hash
    corpus.mkdir(parents=True, exist_ok=True)
    fname = f"p{row.i}-{run_ts}.json"
    payload = {
        "seed": row.failing_seed,
        "minimal_input": row.minimal_input,
        "reason": row.reason,
        "task": task.name,
        "strategy_hash": task.strategy_hash,
        "particle": row.i,
        "run": run_ts,
    }
    (corpus / fname).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str((corpus / fname).relative_to(REPO_ROOT))


def run_l0(
    task_dir: Path | str,
    *,
    k: int,
    steps: int,
    canvas: int,
    base_seed: int = 0,
    sandbox: Optional[str] = None,
    cases: int = 256,
    timeout_ms: int = 2000,
    temperatures: Optional[List[float]] = None,
    algs: Optional[List[str]] = None,
) -> dict:
    """Full L0 cycle for one task; returns the run summary dict."""
    task = load_task(task_dir)
    prompt = task.build_prompt()
    run_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    print(f"L0 gen: task={task.name} K={k} steps={steps} canvas={canvas} "
          f"strategy_hash={task.strategy_hash[:12]}", flush=True)

    t0 = time.perf_counter()
    particles = sample(
        prompt,
        canvas_len=canvas,
        steps=steps,
        particles=k,
        temperatures=temperatures,
        algs=algs,
        base_seed=base_seed,
        progress=lambda i, r: print(
            f"  particle {i}: temp={r.temperature} alg={r.alg} "
            f"wall={r.wall_s}s conf={r.mean_commit_conf} truncated={r.truncated} "
            f"err={r.error or 'none'}",
            flush=True,
        ),
    )
    model_wall_s = time.perf_counter() - t0

    rows: List[CandidateRow] = []
    for p in particles:
        row = CandidateRow(
            i=p.index,
            temperature=p.temperature,
            alg=p.alg,
            seed=p.seed,
            gen_wall_s=p.wall_s,
            s_per_step=(round(sum(p.per_step_s) / len(p.per_step_s), 3)
                        if p.per_step_s else None),
            committed_tokens_total=p.committed_tokens_total,
            mean_commit_conf=p.mean_commit_conf,
            truncated=p.truncated,
            gen_error=p.error,
            extract_ok=False, extract_reason="not attempted",
            code_chars=0, compile_status="not_attempted",
        )
        if p.error:
            row.extract_reason = f"skipped (generation error: {p.error})"
            rows.append(row)
            continue

        ex = extract_rust(p.raw_text, must_define=task.must_define)
        row.extract_ok = ex.ok
        row.extract_reason = ex.reason
        if not ex.ok:
            rows.append(row)
            continue
        row.code_chars = len(ex.code)

        t_h = time.perf_counter()
        outcome = run_candidate(
            ex.code, task.properties_source,
            cases=cases, timeout_ms=timeout_ms, sandbox=sandbox,
        )
        row.harness_wall_s = round(time.perf_counter() - t_h, 3)
        row.compile_status = outcome.compile.status
        row.diag_codes, row.diag_spans = _diag_summary(outcome)
        if outcome.test is not None:
            rep = outcome.test.report or {}
            row.test_status = outcome.test.status
            row.reason = outcome.test.reason or None
            row.successes = rep.get("successes")
            if rep:
                row.rejects = int(rep.get("local_rejects", 0)) + int(
                    rep.get("global_rejects", 0))
            if outcome.test.status == "fail":
                row.minimal_input = outcome.test.minimal_input
                row.failing_seed = outcome.test.seed or None
        row.score = score_of(
            row.compile_status == "ok",
            row.test_status == "pass",
            outcome.test.report if outcome.test else None,
        )
        rows.append(row)
        print(f"  candidate {row.i}: compile={row.compile_status} "
              f"test={row.test_status} score={row.score} "
              f"harness={row.harness_wall_s}s", flush=True)

    # ---- selection: highest score; tie-break mean conf, then first ---------
    def key(r: CandidateRow):
        conf = r.mean_commit_conf if r.mean_commit_conf is not None else -1.0
        return (r.score, conf, -r.i)

    best = max(rows, key=key) if rows else None
    if best is not None:
        best.selected = True

    # ---- persist failing seeds into the corpus -----------------------------
    seed_files = []
    for r in rows:
        path = _persist_failing_seed(task, r, run_ts)
        if path:
            seed_files.append(path)

    # ---- JSONL run record ---------------------------------------------------
    runs_dir = task.dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    rec_path = runs_dir / f"{run_ts}.jsonl"
    from temper.model.runtime import mps_memory_snapshot

    meta = {
        "type": "run",
        "milestone": "M1-L0",
        "task": task.name,
        "task_dir": str(task.dir),
        "strategy_hash": task.strategy_hash,
        "K": k,
        "steps": steps,
        "canvas": canvas,
        "base_seed": base_seed,
        "model": "Dream-org/Dream-Coder-v0-Instruct-7B",
        "model_wall_s": round(model_wall_s, 2),
        "sandbox": sandbox or "default(seatbelt)",
        "cases": cases,
        "timeout_ms": timeout_ms,
        "utc": datetime.now(timezone.utc).isoformat(),
        "memory": mps_memory_snapshot(),
    }
    with open(rec_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(meta) + "\n")
        for r in rows:
            f.write(json.dumps({"type": "candidate", **asdict(r)}) + "\n")
        f.write(json.dumps({
            "type": "best",
            "i": best.i if best else None,
            "score": best.score if best else None,
            "test_status": best.test_status if best else None,
            "failing_seed_files": seed_files,
        }) + "\n")

    summary = {
        "run_record": str(rec_path),
        "task": task.name,
        "K": k,
        "best": asdict(best) if best else None,
        "n_pass": sum(1 for r in rows if r.test_status == "pass"),
        "n_compile_ok": sum(1 for r in rows if r.compile_status == "ok"),
        "model_wall_s": round(model_wall_s, 2),
        "harness_wall_s": round(sum(r.harness_wall_s or 0 for r in rows), 2),
        "seed_files": seed_files,
    }
    print(f"L0 done: {summary['n_pass']}/{k} pass, "
          f"{summary['n_compile_ok']}/{k} compile ok; "
          f"best=p{best.i if best else '-'} score={best.score if best else '-'}; "
          f"record={rec_path}", flush=True)
    return summary


# --------------------------------------------------------------------------
# temper eval: pass@k aggregation across run records
# --------------------------------------------------------------------------

def eval_runs(task_dir: Path | str, k: int, repeat: int) -> dict:
    """Unbiased pass@k over the last `repeat` run records with K==k.

    Per run with n candidates, c passing: pass@j = 1 - C(n-c, j)/C(n, j)
    (Chen et al. 2021, product form for stability), averaged over runs.
    """
    task_dir = Path(task_dir)
    if not task_dir.is_absolute():
        task_dir = Path.cwd() / task_dir
    recs = sorted((task_dir / "runs").glob("*.jsonl"))
    runs = []
    for path in recs:
        cands, meta = [], None
        for line in path.read_text(encoding="utf-8").splitlines():
            obj = json.loads(line)
            if obj.get("type") == "run":
                meta = obj
            elif obj.get("type") == "candidate":
                cands.append(obj)
        if meta and meta.get("K") == k and len(cands) == k:
            runs.append({"meta": meta, "candidates": cands, "file": path.name})
    if not runs:
        raise SystemExit(f"no run records with K={k} in {task_dir/'runs'}")
    runs = runs[-repeat:] if repeat else runs

    ks = sorted({1, 2, 4, 8} & set(range(1, k + 1)))
    table = {}
    for kk in ks:
        vals = []
        for run in runs:
            n = len(run["candidates"])
            c = sum(1 for c2 in run["candidates"] if c2["test_status"] == "pass")
            if kk > n:
                continue
            # 1 - C(n-c, kk)/C(n, kk) in product form
            prod = 1.0
            for i in range(kk):
                prod *= (n - c - i) / (n - i)
            vals.append(1.0 - prod)
        table[kk] = round(sum(vals) / len(vals), 3) if vals else None

    per_run = [{
        "file": run["file"],
        "utc": run["meta"].get("utc"),
        "steps": run["meta"].get("steps"),
        "n_pass": sum(1 for c in run["candidates"] if c["test_status"] == "pass"),
        "best_score": max((c["score"] for c in run["candidates"]), default=None),
        "best_selected_test": next(
            (c["test_status"] for c in run["candidates"] if c.get("selected")), None),
    } for run in runs]

    n_pass_runs = sum(1 for r in per_run if r["n_pass"] > 0)
    return {
        "task": task_dir.name,
        "K": k,
        "runs_used": len(per_run),
        "runs_with_at_least_one_pass": n_pass_runs,
        "pass_at_k": table,
        "per_run": per_run,
    }
