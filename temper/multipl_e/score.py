"""Score MultiPL-E completion files through the official container (M2.0 item 3).

Mechanics (all live-verified against this machine, see decision doc
§container + m2 design):
  * image `ghcr.io/nuprl/multipl-e-evaluation` (amd64-only; runs under
    Rosetta on Apple Silicon -> expect slowness),
  * macOS Docker bind mounts come up empty on this host -> input/results are
    staged with `docker cp` (the container /tmp dir is used as the staging
    parent because docker cp does not create intermediate directories),
  * `official`  : entrypoint `python3 main.py --dir ... --output-dir ...`,
    rustc invoked with NO --edition -> Rust 2015 (GOTCHA 2),
  * `normalized`: same image, eval_rust.py patched inside the container so
    rustc gets `--edition=2021`; both columns are reported and their delta is
    itself a finding.

pass@k: faithful port of the MultiPL-E repo's pass_k.py (unbiased estimator,
numpy is in the venv), aggregated per problem and over problems. Honest
counting: the export manifest's excluded rows are folded back in as failures
(`honest` column) in addition to the scored-only column.

Results land in `bench/multipl-e/results/<ts>.json` with per-problem rows.
Stdlib + numpy only.
"""

from __future__ import annotations

import gzip
import json
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

__all__ = ["pass_at_k", "score_dir", "run_container"]

IMAGE = "ghcr.io/nuprl/multipl-e-evaluation"
PLATFORM = "linux/amd64"
DOCKER_TIMEOUT_S = 60.0          # per docker plumbing call
RUN_TIMEOUT_S = 3 * 3600.0       # hard ceiling on one container scoring run
CONTAINER_RUSTC = "1.75.0"       # pinned in the 2024-07-12 image build
# rustc invocation in /code/eval_rust.py: ["rustc", path, "-o", basename]
_EDITION_SED = (
    "sed -i 's/\"rustc\", path, \"-o\", basename\\]/"
    "\"rustc\", path, \"-o\", basename, \"--edition=2021\"\\]/' /code/eval_rust.py"
)
_EDITION_GREP = "grep -q '\"--edition=2021\"' /code/eval_rust.py"


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator (Chen et al. 2021; MultiPL-E pass_k.py).

    1 - C(n-c, k) / C(n, k), computed in the product form used upstream.
    Returns 1.0 when n - c < k (more successes than the window needs).
    """
    if n - c < k:
        return 1.0
    return 1.0 - float(np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))


def _docker(*args: str, timeout: float = DOCKER_TIMEOUT_S) -> str:
    p = subprocess.run(["docker", *args], capture_output=True, text=True,
                       timeout=timeout)
    if p.returncode != 0:
        raise RuntimeError(f"docker {' '.join(args[:3])}... failed: "
                           f"{p.stderr.strip()[:500]}")
    return p.stdout


def run_container(
    input_dir: Path,
    out_dir: Path,
    *,
    version: str,
    max_workers: int = 2,
    image: str = IMAGE,
) -> dict:
    """Stage `input_dir` into a fresh container, run the official scorer,
    copy results back into `out_dir`. Returns {problem: [statuses]}."""
    input_dir = Path(input_dir).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    cid = f"temper-me-{ts}-{version}"

    if version == "official":
        inner = (f"mkdir -p /tmp/output && exec python3 main.py "
                 f"--dir /tmp/input --output-dir /tmp/output "
                 f"--max-workers {max_workers}")
    elif version == "normalized":
        inner = (f"mkdir -p /tmp/output && {_EDITION_SED} && {_EDITION_GREP} "
                 f"&& exec python3 main.py --dir /tmp/input "
                 f"--output-dir /tmp/output --max-workers {max_workers}")
    else:
        raise ValueError(f"unknown version {version!r} (official|normalized)")

    _docker("create", "--platform", PLATFORM, "--name", cid,
            "--entrypoint", "/bin/sh", image, "-c", inner)
    try:
        # GOTCHA 3: docker cp, not bind mounts. docker cp does not create
        # intermediate directories and cannot write into a not-yet-started
        # container's missing paths, so we stage a manifest-free copy of the
        # input dir locally and copy it in one shot (docker cp creates
        # /tmp/input from the copied directory). manifest.json itself is not
        # a problem file and would crash main.py.
        stage = Path(out_dir) / ".staged-input"
        if stage.exists():
            shutil.rmtree(stage)
        stage.mkdir(parents=True)
        staged = []
        for f in sorted(input_dir.glob("*.json")):
            if f.name == "manifest.json":
                continue
            shutil.copy2(f, stage / f.name)
            staged.append(f.name)
        _docker("cp", str(stage), f"{cid}:/tmp/input", timeout=300.0)
        shutil.rmtree(stage)
        t0 = time.monotonic()
        proc = subprocess.run(
            ["docker", "start", "-a", cid],
            capture_output=True, text=True, timeout=RUN_TIMEOUT_S,
        )
        wall = round(time.monotonic() - t0, 1)
        if proc.returncode != 0:
            raise RuntimeError(
                f"container {cid} exited {proc.returncode}: "
                f"{proc.stderr.strip()[-800:]}")
        _docker("cp", f"{cid}:/tmp/output/.", str(out_dir) + "/", timeout=300.0)
    finally:
        try:
            _docker("rm", "-f", cid)
        except Exception:
            pass

    statuses = {}
    for res in sorted(out_dir.glob("*.results.json")) + \
               sorted(out_dir.glob("*.results.json.gz")):
        opener = gzip.open if res.suffix == ".gz" else open
        with opener(res, "rt") as f:
            data = json.load(f)
        statuses[data["name"]] = [r.get("status") for r in data.get("results", [])]
    return {"statuses": statuses, "staged": staged, "wall_s": wall,
            "container": cid, "image": image, "platform": PLATFORM,
            "rustc": CONTAINER_RUSTC,
            "edition": "2015" if version == "official" else "2021"}


def _parse_input_dir(input_dir: Path) -> dict:
    """Problem metadata + per-problem export counts (from problem files and
    the manifest's excluded rows)."""
    manifest = {}
    mpath = input_dir / "manifest.json"
    if mpath.exists():
        manifest = json.loads(mpath.read_text(encoding="utf-8"))
    problems = {}
    for f in sorted(input_dir.glob("*.json")):
        if f.name == "manifest.json":
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        problems[d["name"]] = {
            "completions_file": f.name,
            "n_scored": len(d.get("completions", [])),
        }
    excluded_by_problem: dict = {}
    for e in manifest.get("excluded", []):
        p = e.get("problem")
        if p:
            excluded_by_problem[p] = excluded_by_problem.get(p, 0) + 1
    return {"problems": problems, "manifest": manifest,
            "excluded": manifest.get("excluded", []),
            "excluded_by_problem": excluded_by_problem}


def _read_results_dir(res_dir: Path) -> dict:
    out = {}
    for res in sorted(res_dir.glob("*.results.json")) + \
               sorted(res_dir.glob("*.results.json.gz")):
        opener = gzip.open if res.suffix == ".gz" else open
        with opener(res, "rt") as f:
            data = json.load(f)
        out[data["name"]] = [r.get("status") for r in data.get("results", [])]
    return out


_KS = (1, 10, 100)


def score_dir(
    input_dir: Path,
    *,
    version: str = "both",
    max_workers: int = 2,
    results_dir: Path | None = None,
) -> dict:
    """Score every problem file in `input_dir` under the requested container
    version(s) and write the merged results JSON."""
    input_dir = Path(input_dir).resolve()
    if results_dir is None:
        results_dir = Path(__file__).resolve().parents[2] / "bench" / \
            "multipl-e" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    info = _parse_input_dir(input_dir)
    problems_meta = info["problems"]
    if not problems_meta:
        raise RuntimeError(f"no problem files to score in {input_dir}")

    versions = (["official", "normalized"] if version == "both" else [version])
    runs = {}
    for v in versions:
        tmp_out = results_dir / f".run-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{v}"
        runs[v] = run_container(input_dir, tmp_out, version=v,
                                max_workers=max_workers)
        statuses = _read_results_dir(tmp_out)
        runs[v]["statuses"] = statuses

    # ---- per-problem pass@k ------------------------------------------------
    per_problem = []
    agg = {v: {f"pass@{k}": [] for k in _KS} for v in versions}
    agg_honest = {v: {f"pass@{k}": [] for k in _KS} for v in versions}
    n_excluded_total = len(info["excluded"])
    for name, meta in problems_meta.items():
        n_excl = info["excluded_by_problem"].get(name, 0)
        row = {"name": name, "n_scored": meta["n_scored"],
               "n_excluded": n_excl}
        for v in versions:
            sts = runs[v]["statuses"].get(name, [])
            n = len(sts)
            c = sum(1 for s in sts if s == "OK")
            row[f"statuses_{v}"] = sts
            row[f"pass_at_k_{v}"] = {
                f"pass@{k}": (pass_at_k(n, c, k) if n >= k else None)
                for k in _KS if n > 0
            }
            for k in _KS:
                if n >= k:
                    agg[v][f"pass@{k}"].append(pass_at_k(n, c, k))
            # honest counting: this problem's excluded rows count as failures
            # (extraction failures are scored failures per the design)
            n_h = n + n_excl
            c_h = c
            row[f"pass_at_k_honest_{v}"] = {
                f"pass@{k}": (pass_at_k(n_h, c_h, k) if n_h >= k else None)
                for k in _KS if n_h > 0
            }
            for k in _KS:
                if n_h >= k:
                    agg_honest[v][f"pass@{k}"].append(pass_at_k(n_h, c_h, k))
        per_problem.append(row)

    def mean(vals):
        return round(sum(vals) / len(vals), 4) if vals else None

    aggregate = {}
    for v in versions:
        aggregate[v] = {
            f"pass@{k}": mean(agg[v][f"pass@{k}"]) for k in _KS
        }
        aggregate[v]["n_problems_with_n_ge_k"] = {
            f"pass@{k}": len(agg[v][f"pass@{k}"]) for k in _KS
        }
        aggregate[f"honest-{v}"] = {
            f"pass@{k}": mean(agg_honest[v][f"pass@{k}"]) for k in _KS
        }
        aggregate[f"honest-{v}"]["n_problems_with_n_ge_k"] = {
            f"pass@{k}": len(agg_honest[v][f"pass@{k}"]) for k in _KS
        }
        aggregate[f"honest-{v}"]["n_excluded_rows"] = n_excluded_total
        aggregate[f"honest-{v}"]["n_excluded_by_problem"] = info["excluded_by_problem"]

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    results = {
        "type": "multipl-e-results",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_dir": str(input_dir),
        "versions": versions,
        "container": {v: {k: runs[v][k] for k in
                          ("image", "platform", "rustc", "edition", "wall_s")}
                      for v in versions},
        "per_problem": per_problem,
        "aggregate": aggregate,
    }
    out_path = results_dir / f"{ts}.json"
    out_path.write_text(json.dumps(results, indent=1), encoding="utf-8")
    return {
        "input": str(input_dir),
        "versions": "+".join(versions),
        "results_path": str(out_path),
        "aggregate": aggregate,
        **results,
    }
