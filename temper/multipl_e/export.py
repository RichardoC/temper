"""Export temper run records to MultiPL-E completion files (M2.0 item 2).

Official completion-file format (consumed by the MultiPL-E container's
``main.py``): one ``<name>.json`` per problem::

    {"name": ..., "language": "rs", "prompt": ..., "tests": ...,
     "completions": [...]}

program compiled by the scorer = prompt + completions[i] + '\\n' + tests.

Two run-record flavors:

1. **MultiPL-E runs** (meta carries ``"benchmark"``): candidate rows name a
   problem from the vendored dataset (``bench/multipl-e/data/<benchmark>.json``);
   prompt/tests/stop_tokens join from the task definition, and the extracted
   whole-module code is reduced to the prompted function's body fragment (the
   official completion shape: the prompt ends mid-function and the tests
   string begins with the function's closing brace).

2. **temper bench runs** (task not in the benchmark — the M2.0 smoke-test
   flavor): each temper task becomes a *synthetic* problem: prompt = ""
   (extracted code is a complete module) and tests = the hand-authored
   deterministic Rust main in ``bench/<task>/multipl_e.tests.rs`` (a faithful
   port of the task's approved property contract to fixed cases; properties.rs
   itself stays the held-in verifier and needs proptest, which the offline
   container does not carry).

GOTCHA 1 (verified, decision doc §container): the scorer does NOT enforce
stop_tokens — a retained stop sequence duplicates the closing brace and
becomes a spurious SyntaxError. Every completion is therefore stripped at
export time: cut at the first stop-token occurrence, drop fence residue.

Honest counting (design): only extract_ok rows are exported, but EVERY
excluded row is recorded in ``manifest.json`` with its reason, so pass@k can
be reconstructed with extraction failures counted as scored failures. A
problem whose rows are all excluded is still written (completions: []) —
the container skips it, and the results stage reports it as all-fail.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from temper.model.extract import extract_rust, repair_code

__all__ = ["export_runs", "load_run_record", "load_benchmark"]

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "bench" / "multipl-e" / "data"

# official MultiPL-E whole-file -> body-fragment conversion needs the prompted
# function's signature line; it is the last `fn` header in the prompt.
_PROMPTED_FN_RE = re.compile(r"^\s*(?:pub\s+)?fn\s+([A-Za-z_][A-Za-z0-9_]*).*$", re.M)


def load_benchmark(benchmark: str, data_dir: Path = DATA_DIR) -> Dict[str, dict]:
    """Vendored task definitions keyed by problem name."""
    path = data_dir / f"{benchmark}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"vendored benchmark {benchmark!r} not found at {path} — "
            "run scripts/fetch_multipl_e.py first"
        )
    tasks = json.loads(path.read_text(encoding="utf-8"))
    return {t["name"]: t for t in tasks}


def load_run_record(path: Path) -> Tuple[dict, List[dict]]:
    """Parse one temper run record (meta + candidate rows; `best` ignored)."""
    meta, cands = None, []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if obj.get("type") == "run":
            meta = obj
        elif obj.get("type") == "candidate":
            cands.append(obj)
    if meta is None:
        raise ValueError(f"{path}: no run meta line")
    return meta, cands


def strip_stop_residue(code: str, stop_tokens=()) -> str:
    """GOTCHA 1: the scorer does not enforce stop_tokens; retained stop
    sequences / fence residue duplicate closing braces -> spurious
    SyntaxError. Cut at the first stop-token occurrence, drop fence-marker
    lines and trailing blank lines."""
    cut = len(code)
    for tok in stop_tokens or ():
        if not tok:
            continue
        i = code.find(tok)
        if i != -1:
            cut = min(cut, i)
    code = code[:cut]
    lines = [ln for ln in code.splitlines()
             if not ln.strip().startswith("```")]
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


def _fn_header_line(prompt: str) -> Optional[str]:
    """The (unindented) `fn ...` header line the MultiPL-E prompt ends with."""
    m = None
    for m in _PROMPTED_FN_RE.finditer(prompt):
        pass
    return m


def to_body_fragment(module_code: str, prompt: str) -> Optional[str]:
    """Reduce a whole-module extraction to the official MultiPL-E completion
    shape (a body fragment of the prompted function).

    Returns None when the extraction is not safely convertible (prompted
    function not found at module level, or its signature line does not match
    the prompt's) — the caller then EXCLUDES the row with a manifest reason
    instead of exporting a completion that cannot compile against the prompt.
    """
    m = _fn_header_line(prompt)
    if m is None:
        return None
    fn_name = m.group(1)
    lines = module_code.splitlines()
    start = None
    for i, ln in enumerate(lines):
        if re.match(r"^\s*(?:pub\s+)?fn\s+" + re.escape(fn_name) + r"\b", ln):
            start = i
            break
    if start is None:
        return None  # prompted fn not defined at module level in extraction
    # the signature must match the prompt's header up to whitespace, else the
    # model renamed/reshaped the API -> not exportable
    sig_chars = "".join(lines[start].split())
    prompt_chars = "".join(m.group(0).split())
    if not prompt_chars.endswith("{"):
        prompt_chars += "{"
    if sig_chars.rstrip() != prompt_chars.rstrip():
        return None
    # line-based brace scan to the matching close of the prompted fn
    depth, end = 0, None
    for j in range(start, len(lines)):
        depth += lines[j].count("{") - lines[j].count("}")
        if depth <= 0 and j > start:
            end = j
            break
        if depth == 0 and j == start and "{" in lines[j] and "}" in lines[j]:
            end = j  # one-line fn
            break
    if end is None:
        return None  # unbalanced -> extraction was truncated, not exportable
    body = lines[start + 1 : end]
    while body and not body[0].strip():
        body.pop(0)
    while body and not body[-1].strip():
        body.pop()
    if not body:
        return None
    # dedent by the body's common indent: a model continuation would start at
    # column 0; interior relative indentation is preserved.
    indents = [len(ln) - len(ln.lstrip()) for ln in body if ln.strip()]
    cut = min(indents) if indents else 0
    body = [ln[cut:] if ln.strip() else "" for ln in body]
    return "\n".join(body)


def _raw_text_of(row: dict, run_dir: Path) -> Optional[str]:
    rp = row.get("raw_path")
    if not rp:
        return None
    p = Path(rp)
    if not p.is_absolute():
        p = REPO_ROOT / rp
    if not p.exists():
        return None
    return p.read_text(encoding="utf-8")


def _temper_task_dir(meta: dict) -> Optional[Path]:
    td = meta.get("task_dir")
    if not td:
        return None
    p = Path(td)
    return p if p.is_absolute() else REPO_ROOT / p


def _task_must_define(task_dir: Optional[Path]) -> List[str]:
    """must_define straight from the task's task.toml (run records did not
    carry it before schema v1.2)."""
    if task_dir is None:
        return []
    try:
        import tomllib  # py3.11+
    except ImportError:
        import tomli as tomllib  # venv dep
    with open(task_dir / "task.toml", "rb") as f:
        meta = tomllib.load(f)["task"]
    return list(meta.get("must_define", []))


def export_runs(
    runs: List[Path],
    out_dir: Path,
    benchmark: Optional[str] = None,
    also_unsteered_guise: bool = False,
    repair: bool = True,
) -> dict:
    """Export candidate rows of the given run records into MultiPL-E
    completion files under ``out_dir``; returns the manifest dict (also
    written to ``out_dir/manifest.json``).

    repair=True (default, M2.1 behavior): rows whose extraction failed but is
    repairable — or whose record verdict says compile_error — get one
    deterministic L0.5 repair round before being counted as extraction
    failures. repair=False (M2.2 ablation column) skips the repair pass
    entirely: the unsteered model's raw extraction quality, no salvage.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    bench = load_benchmark(benchmark) if benchmark else {}

    # group rows per problem; temper tasks merge across runs of the same task
    problems: Dict[str, dict] = {}
    excluded: List[dict] = []

    for run_path in runs:
        run_path = Path(run_path)
        meta, cands = load_run_record(run_path)
        run_id = run_path.stem
        task_name = meta.get("task", "unknown")

        if meta.get("benchmark") or any(c.get("problem") for c in cands):
            flavor = "multipl-e"
        else:
            flavor = "temper"

        if flavor == "multipl-e":
            # rows must name their problem; prompt/tests join from the dataset
            for c in cands:
                pname = c.get("problem") or meta.get("benchmark_problem")
                if pname not in bench:
                    excluded.append({
                        "run": run_path.name, "i": c.get("i"),
                        "problem": pname,
                        "reason": f"problem {pname!r} not in vendored benchmark {benchmark!r}",
                    })
                    continue
                td = bench[pname]
                problems.setdefault(pname, {
                    "name": pname, "language": "rs",
                    "prompt": td["prompt"], "tests": td["tests"],
                    "stop_tokens": td.get("stop_tokens", []),
                    "flavor": "multipl-e", "completions": [],
                    "source_rows": [],
                })
        else:
            # temper synthetic problem: prompt="", tests = hand-authored
            # deterministic port of the approved property contract
            task_dir = _temper_task_dir(meta)
            tests_path = (task_dir / "multipl_e.tests.rs") if task_dir else None
            if tests_path is None or not tests_path.exists():
                raise FileNotFoundError(
                    f"{run_path}: temper task {task_name!r} has no "
                    f"multipl_e.tests.rs (hand-authored deterministic port of "
                    "the approved property) — refusing to invent scoring tests"
                )
            problems.setdefault(f"temper-{task_name}", {
                "name": f"temper-{task_name}",
                "language": "rs",
                "prompt": "",
                "tests": tests_path.read_text(encoding="utf-8"),
                "stop_tokens": [],
                "flavor": "temper",
                "completions": [],
                "source_rows": [],
            })

        # ---- per-candidate export --------------------------------------
        pname = (f"temper-{task_name}" if flavor == "temper"
                 else None)  # multipl-e rows carry their own problem name
        for c in cands:
            if flavor == "multipl-e":
                pname_c = c.get("problem") or meta.get("benchmark_problem")
                if pname_c not in bench:
                    continue  # already excluded above
                prob = problems[pname_c]
                td = bench[pname_c]
                stop_tokens = td.get("stop_tokens", [])
                must_define = []
                mfn = _fn_header_line(td["prompt"])
                if mfn:
                    must_define = [mfn.group(1)]
            else:
                prob = problems[pname]
                stop_tokens = ()
                must_define = _task_must_define(task_dir)

            base = {"run": run_path.name, "i": c.get("i"),
                    "problem": prob["name"]}
            if c.get("gen_error"):
                excluded.append({**base,
                    "reason": f"generation error: {c['gen_error']}"})
                continue
            raw = _raw_text_of(c, run_path.parent)
            if raw is None:
                excluded.append({**base,
                    "reason": "raw text unavailable (pre-M1.5 record: no raw_path sidecar)"})
                continue
            ex = extract_rust(raw, must_define=must_define)
            repaired_at_export = False
            if repair and not ex.ok and ex.code:
                # L0.5 salvage, same pipeline as l0: a repairable span is
                # retried through the deterministic repair pass before being
                # counted as an extraction failure
                rp = repair_code(ex.code, must_define=must_define)
                if rp.valid:
                    ex = type(ex)(ok=True, code=rp.code,
                                  reason="ok [L0.5 repair applied]")
                    repaired_at_export = True
            elif repair and ex.ok and c.get("compile_status") == "compile_error":
                # the record's harness verdict says this code fails rustc
                # (e.g. run-B p0's stray block) — same L0.5 repair round
                rp = repair_code(ex.code, must_define=must_define)
                if rp.repaired and rp.valid:
                    ex = type(ex)(ok=True, code=rp.code,
                                  reason="ok [L0.5 repair applied post-compile]")
                    repaired_at_export = True
            if not ex.ok:
                excluded.append({**base,
                    "reason": f"extract rejected at export: {ex.reason}",
                    "raw_path": c.get("raw_path")})
                continue
            if prob["flavor"] == "multipl-e":
                body = to_body_fragment(ex.code, prob["prompt"])
                if body is None:
                    excluded.append({**base,
                        "reason": ("whole-module extraction not convertible to the "
                                   "prompted-fn body fragment (renamed/reshaped API "
                                   "or unsupported shape)"),
                        "raw_path": c.get("raw_path")})
                    continue
                code = body
            else:
                code = ex.code
            # GOTCHA 1 stop-strip (M2.4b correction): the `\n}` stop-token cut
            # is for WHOLE-module/continuation completions, where a col-0 `}`
            # is the fn's own close. A multipl-e BODY FRAGMENT excludes the
            # fn's close by construction (to_body_fragment), but after the
            # min-indent dedent a TOP-LEVEL block in the fn body (for/if/
            # match) legitimately lands its close at col 0 — cutting there
            # amputates a syntactically complete body into a guaranteed
            # SyntaxError (measured: 64-76 of each committed cell's
            # SyntaxErrors, incl. all 76 in the s512 cell — see
            # scripts/m2_4b_strip_artifact.py and docs/m2.4b-report.md).
            # The fragment path therefore keeps only the fence-line/blank-line
            # trimming; the temper-flavor path (whole modules vs self-contained
            # tests) keeps the full stop-strip unchanged.
            frag = prob["flavor"] == "multipl-e"
            code = strip_stop_residue(code, stop_tokens if not frag else ())
            if not code.strip():
                excluded.append({**base,
                    "reason": "completion empty after stop-token/fence stripping",
                    "raw_path": c.get("raw_path")})
                continue
            prob["completions"].append(code)
            prob["source_rows"].append({
                "run": run_path.name, "i": c.get("i"),
                "raw_path": c.get("raw_path"),
                "record_extract_ok": c.get("extract_ok"),
                "repaired_at_export": repaired_at_export,
                "chars": len(code),
            })

    # ---- write problem files -------------------------------------------
    written = []
    for prob in problems.values():
        payload = {
            "name": prob["name"], "language": "rs",
            "prompt": prob["prompt"], "tests": prob["tests"],
            "completions": prob["completions"],
        }
        f = out_dir / f"{prob['name']}.json"
        f.write_text(json.dumps(payload, indent=1), encoding="utf-8")
        written.append({"file": f.name, "name": prob["name"],
                        "flavor": prob["flavor"],
                        "completions": len(prob["completions"]),
                        "source_rows": prob["source_rows"]})
        if also_unsteered_guise and prob["completions"]:
            # Qwen-twin format demo: the SAME completions re-emitted under a
            # plain "unsteered model" guise — scoring must be identical,
            # proving the format (and the adapter) is model-blind.
            gp = dict(payload)
            gp["name"] = prob["name"] + "-unsteered-guise"
            gf = out_dir / f"{gp['name']}.json"
            gf.write_text(json.dumps(gp, indent=1), encoding="utf-8")
            written.append({"file": gf.name, "name": gp["name"],
                            "flavor": prob["flavor"] + "+unsteered-guise",
                            "completions": len(gp["completions"])})

    n_exported = sum(w["completions"] for w in written
                     if "guise" not in w["flavor"])
    manifest = {
        "type": "multipl-e-export-manifest",
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "benchmark_arg": benchmark,
        "repair": repair,
        "runs": [str(r) for r in runs],
        "problems": written,
        "excluded": excluded,
        "counts": {
            "candidate_rows": sum(1 for _ in excluded) + n_exported,
            "exported_completions": n_exported,
            "excluded_rows": len(excluded),
        },
        "note": ("excluded rows are scored failures under honest counting "
                 "(pass@k is recomputable with n=exported+excluded)"),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
