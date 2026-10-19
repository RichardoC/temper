#!/usr/bin/env python3
"""One-time fetch of the MultiPL-E Rust task definitions (M2.0 item 1).

Downloads the `humaneval-rs` (156 tasks) and `mbpp-rs` (354 tasks) configs of
HF dataset `nuprl/MultiPL-E` via the datasets-server rows API (plain REST, no
pyarrow needed) and writes them verbatim as JSON arrays:

    bench/multipl-e/data/humaneval-rs.json
    bench/multipl-e/data/mbpp-rs.json

After this one-time fetch, temper is offline with respect to the benchmark:
the vendored files are the join source for `temper export-multipl-e`.
Dataset revision (git sha) is recorded in bench/multipl-e/data/README.md.

Usage: temper/.venv/bin/python scripts/fetch_multipl_e.py [--out DIR]
Stdlib only (urllib).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

DATASET = "nuprl/MultiPL-E"
CONFIGS = ("humaneval-rs", "mbpp-rs")
SPLIT = "test"
PAGE = 100
ROWS_URL = "https://datasets-server.huggingface.co/rows"
REPO_API_URL = "https://huggingface.co/api/datasets/" + DATASET
EXPECTED_COUNTS = {"humaneval-rs": 156, "mbpp-rs": 354}

ATTRIBUTION = """\
# Vendored MultiPL-E Rust benchmark data (M2.0)

Source: Hugging Face dataset `{dataset}` (configs: {configs}, split: `{split}`).
Fetched: {fetched} via the datasets-server rows API (one-time fetch, offline after).
Dataset revision (git sha) at fetch time: `{sha}`.

Counts: {counts}.

Upstream license: **BSD 3-Clause License with Machine Learning Restriction**
(Copyright (c) 2022, Northeastern University, Oberlin College, Roblox Inc,
Stevens Institute of Technology, University of Massachusetts Amherst, and
Wellesley College). See https://github.com/nuprl/MultiPL-E/blob/master/LICENSE.
The files in this directory are verbatim task definitions ({fields}) and are
redistributed here under the same license, used solely as evaluation benchmark
material.

Reference: Cassano et al., "MultiPL-E: A Benchmark and Evaluation Harness for
Cross-Programming-Language Code Generation" (COLING 2023).
"""


def _get_json(url: str, retries: int = 4) -> dict:
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "temper-m2/1.0"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:  # transient network flake -> backoff and retry
            last = exc
            time.sleep(2.0 * (attempt + 1))
    raise SystemExit(f"GET failed after {retries} attempts: {url}: {last}")


def fetch_config(config: str) -> list:
    """Page through the rows API; returns the raw task dicts in order."""
    tasks: list = []
    offset = 0
    while True:
        qs = urllib.parse.urlencode({
            "dataset": DATASET, "config": config,
            "split": SPLIT, "offset": offset, "length": PAGE,
        })
        url = f"{ROWS_URL}?{qs}"
        d = _get_json(url)
        if "rows" not in d:
            raise SystemExit(f"unexpected rows API response for {config}: {str(d)[:300]}")
        for r in d["rows"]:
            tasks.append(r["row"])
        total = d.get("num_rows_total")
        offset += len(d["rows"])
        if total is not None and offset >= total:
            break
        if not d["rows"]:
            break
    return tasks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "bench" / "multipl-e" / "data"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    repo = _get_json(REPO_API_URL)
    sha = repo.get("sha", "unknown")

    counts = {}
    for config in CONFIGS:
        tasks = fetch_config(config)
        expected = EXPECTED_COUNTS[config]
        if len(tasks) != expected:
            print(f"WARNING: {config}: fetched {len(tasks)} rows, expected {expected}",
                  file=sys.stderr)
        names = [t.get("name") for t in tasks]
        if len(set(names)) != len(names):
            raise SystemExit(f"{config}: duplicate task names in dataset")
        path = out / f"{config}.json"
        path.write_text(json.dumps(tasks, indent=1), encoding="utf-8")
        counts[config] = len(tasks)
        print(f"{config}: {len(tasks)} tasks -> {path}")

    readme = out / "README.md"
    readme.write_text(ATTRIBUTION.format(
        dataset=DATASET,
        configs=", ".join(CONFIGS),
        split=SPLIT,
        fetched=datetime.now(timezone.utc).isoformat(),
        sha=sha,
        counts=", ".join(f"{k}={v}" for k, v in counts.items()),
        fields="name, language, prompt, doctests, original, prompt_terminology, tests, stop_tokens",
    ), encoding="utf-8")
    print(f"revision sha: {sha}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
