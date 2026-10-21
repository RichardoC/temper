#!/usr/bin/env python3
"""M2.4b: quantify the stop-strip truncation artifact in committed exports.

FOUND DURING M2.4b (B3 Qwen twin): export_runs applies strip_stop_residue
(GOTCHA 1: the scorer does not enforce MultiPL-E stop_tokens, so a retained
stop token duplicates the closing brace) to the BODY FRAGMENT produced by
to_body_fragment. But a body fragment whose prompted function contains a
TOP-LEVEL BLOCK (for/if/match at fn-body depth) legitimately contains a
column-0 `}` after the min-indent dedent — and the `\\n}` stop cut amputates
the fragment there, turning a syntactically complete body into a guaranteed
SyntaxError. The strip was designed for whole-module completions (where a
col-0 `}` IS the fn's own close); in the fragment path to_body_fragment
already excludes the fn's close, so an interior `\\n}` cut is ALWAYS wrong.

This script measures, per committed cell (run record + export dir + results
JSON), how many SCORED SyntaxError rows are attributable to that cut:
for every exported completion it recomputes the UNstripped fragment from the
raw sidecar with the committed extraction code and classifies the row as
  strip-cut      committed fragment is a strict prefix of the unstripped
                 body, cut exactly at a `\\n}` boundary, and the unstripped
                 body is brace-balanced (the amputation, not the model,
                 broke it);
  identical      export == unstripped body (no cut);
  other          anything else (model-shaped differences).

Every input is a committed artifact; no GPU, no container.

    temper/.venv/bin/python scripts/m2_4b_strip_artifact.py \
        bench/multipl-e/runs/20261021T141115Z-b3qwen.jsonl \
        bench/multipl-e/exports/20261021T142705Z-b3qwen-norepair \
        bench/multipl-e/results/20261021T142800Z.json [...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from temper.model.extract import extract_rust  # noqa: E402
from temper.multipl_e.export import (  # noqa: E402
    load_benchmark,
    load_run_record,
    to_body_fragment,
    _fn_header_line,
)

REPO = Path(__file__).resolve().parents[1]


def quantify(run_path: str, export_dir: str, results_path: str) -> dict:
    bench = load_benchmark("humaneval-rs")
    meta, cands = load_run_record(Path(run_path))
    res = json.loads(Path(results_path).read_text())
    statuses = {p["name"]: p["statuses_official"] for p in res["per_problem"]}

    # completions in the export are appended in record order per problem
    by_problem = {}
    for c in cands:
        by_problem.setdefault(c["problem"], []).append(c)

    counts = {"strip-cut": 0, "identical": 0, "other": 0, "not-scored": 0}
    per_problem_cuts = {}
    for pname, rows in by_problem.items():
        td = bench[pname]
        mfn = _fn_header_line(td["prompt"])
        must = [mfn.group(1)] if mfn else []
        probs = statuses.get(pname, [])
        for i, c in enumerate(rows):
            st = probs[i] if i < len(probs) else None
            raw_p = REPO / c["raw_path"]
            if not raw_p.exists() or st is None:
                counts["not-scored"] += 1
                continue
            ex = extract_rust(raw_p.read_text(encoding="utf-8"),
                              must_define=must)
            body = to_body_fragment(ex.code, td["prompt"]) if ex.ok else None
            committed = json.loads(
                (Path(export_dir) / f"{pname}.json").read_text()
            )["completions"][i]
            if body is None:
                counts["other"] += 1
                continue
            if committed == body:
                counts["identical"] += 1
                continue
            # strip-cut signature: strict prefix, cut at a `\n}` boundary,
            # unstripped body balanced
            if (body.startswith(committed)
                    and body[len(committed):len(committed) + 1] == "\n"
                    and body[len(committed) + 1:len(committed) + 2] == "}"
                    and body.count("{") == body.count("}")):
                counts["strip-cut"] += 1
                per_problem_cuts.setdefault(pname, []).append(i)
                if st == "SyntaxError":
                    counts.setdefault("strip-cut-and-SyntaxError", 0)
                    counts["strip-cut-and-SyntaxError"] += 1
            else:
                counts["other"] += 1
    return {"counts": counts, "cut_problems": per_problem_cuts,
            "statuses": {n: s for n, s in statuses.items()
                         if n in per_problem_cuts}}


def main(argv):
    if len(argv) < 4 or (len(argv) - 1) % 3 != 0:
        print(__doc__)
        return 2
    print("cell\trows\tidentical\tstrip-cut\tcut&SyntaxErr\tother\tnot-scored")
    i = 1
    while i < len(argv):
        tag = Path(argv[i]).stem.split("Z-")[-1]
        q = quantify(argv[i], argv[i + 1], argv[i + 2])
        c = q["counts"]
        print(f"{tag}\t{sum(v for k, v in c.items() if not isinstance(v, list))}"
              f"\t{c.get('identical', 0)}\t{c.get('strip-cut', 0)}"
              f"\t{c.get('strip-cut-and-SyntaxError', 0)}"
              f"\t{c.get('other', 0)}\t{c.get('not-scored', 0)}")
        if q["cut_problems"]:
            print(f"   cut problems: "
                  f"{ {k: v for k, v in q['cut_problems'].items()} }")
        i += 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
