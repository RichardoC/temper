#!/usr/bin/env python3
"""M2.4a per-problem migration analysis: steps=256 cell vs steps=512 cell.

Reads two (run-record, norepair-results-JSON) pairs and prints, per problem,
the status-class state at 256 and at 512 plus the transition class, then a
migration summary (newly-pass / newly-fail / newly-exclude / recovered /
held). Status classes per problem (K=4):

    PASSn  n of 4 canvases passed            EXCL4  all 4 rows excluded
    FAIL(details)  mixed: S=syntax, W=wrong, T=timeout, O=other status

All inputs are committed artifacts, so every number in the M2.4a report is
reproducible:

    temper/.venv/bin/python scripts/m2_4a_migration.py \
        bench/multipl-e/runs/20261020T100251Z-s256.jsonl \
        bench/multipl-e/results/20261020T221305Z.json \
        bench/multipl-e/runs/<ts>-s512.jsonl \
        bench/multipl-e/results/<ts>.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def load_results(path: str) -> dict:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    per = {p["name"]: p for p in d["per_problem"]}
    return {"per": per, "agg": d["aggregate"], "input_dir": d["input_dir"]}


def state(p: dict) -> str:
    n_ex = p["n_excluded"]
    st = p["statuses_official"]
    n_sc = p["n_scored"]
    if n_ex == 4:
        return "EXCL4"
    npass = sum(1 for s in st if s == "OK")
    if npass == 4:
        return "PASS4"
    # mixed: compact letter counts over scored rows
    letters = {"SyntaxError": "S", "WrongAnswer": "W", "TimeoutError": "T",
               "OK": "P"}
    counts = {}
    for s in st:
        c = letters.get(s, "O")
        counts[c] = counts.get(c, 0) + 1
    detail = "".join(f"{c}{n}" for c, n in sorted(counts.items()))
    return f"FAIL({detail},scored={n_sc},excl={n_ex})"


def klass(p: dict) -> str:
    if p["n_excluded"] == 4:
        return "EXCL"
    return "PASS" if sum(1 for s in p["statuses_official"] if s == "OK") == 4 \
        else "FAIL"


TRANSITIONS = {
    ("FAIL", "PASS"): "NEWLY-PASS",
    ("EXCL", "PASS"): "RECOVERED-EXCL->PASS",
    ("EXCL", "FAIL"): "UNEXCLUDED->FAIL",
    ("PASS", "PASS"): "held-pass",
    ("FAIL", "FAIL"): "held-fail",
    ("EXCL", "EXCL"): "held-excluded",
    ("PASS", "FAIL"): "REGRESSED-PASS->FAIL",
    ("PASS", "EXCL"): "REGRESSED-PASS->EXCL",
    ("FAIL", "EXCL"): "NEWLY-EXCLUDED",
}


def main(argv):
    if len(argv) != 5:
        print(__doc__)
        return 2
    a = load_results(argv[2])   # 256 results JSON
    b = load_results(argv[4])   # 512 results JSON
    names = [n for n in a["per"] if n in b["per"]]
    summary = {}
    print(f"{'problem':44s} {'256':28s} {'512':28s} transition")
    for n in sorted(names):
        pa, pb = a["per"][n], b["per"][n]
        ka, kb = klass(pa), klass(pb)
        tr = TRANSITIONS.get((ka, kb), f"{ka}->{kb}")
        summary[tr] = summary.get(tr, 0) + 1
        print(f"{n:44s} {state(pa):28s} {state(pb):28s} {tr}")
    print("\n[migration summary] 256->512 (25 problems)")
    for t, c in sorted(summary.items(), key=lambda kv: -kv[1]):
        print(f"  {t:28s} {c}")
    for tag, r in (("256", a), ("512", b)):
        ho = r["agg"]["honest-official"]
        print(f"[{tag}] hon p@1={ho['pass@1']} "
              f"excl_rows={ho.get('n_excluded_rows')} "
              f"excl_by_problem={ho.get('n_excluded_by_problem')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
