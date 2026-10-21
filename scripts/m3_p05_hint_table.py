#!/usr/bin/env python3
"""C-hint quartet table (M3.0 P0.5): hint cell vs the corrected s256 anchor.

Prints, per problem of the type-trap quartet, the container status class of the
hint cell's 4 canvases beside the corrected s256 anchor's class, plus the
comparison the design's §5 P0.5 decision rule is read from (passes >= 3/4 ⇒
information-fixable; passes <= 1/4 ⇒ mode-lock).

    temper/.venv/bin/python scripts/m3_p05_hint_table.py \
        --hint-run bench/multipl-e/runs/20261021T160000Z-c-hint.jsonl \
        --hint-results bench/multipl-e/results/20261021T201746Z.json \
        --anchor-results bench/multipl-e/results/20261021T144401Z.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REPO_ROOT = Path(__file__).resolve().parents[1]


def load(path: str) -> dict:
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    d = json.loads(p.read_text(encoding="utf-8"))
    return {row["name"]: row for row in d["per_problem"]}


def classify(row: dict | None) -> str:
    """Compact per-problem class: PASS / SE / EXC / EXCL / mixed letters."""
    if row is None:
        return "NOT-RUN"
    st = row["statuses_official"]
    if row.get("n_excluded") and not st:
        return "EXCL4"
    if not st:
        return "EMPTY"
    n_ok = sum(1 for s in st if s == "OK")
    if n_ok == len(st) and not row.get("n_excluded"):
        return "PASS4"
    letters = {"OK": "P", "SyntaxError": "S", "WrongAnswer": "W",
               "Exception": "E", "TimeoutError": "T"}
    counts = {}
    for s in st:
        c = letters.get(s, "?")
        counts[c] = counts.get(c, 0) + 1
    out = "".join(f"{c}{n}" for c, n in sorted(counts.items()))
    if row.get("n_excluded"):
        out += f"+x{row['n_excluded']}"
    return out


def passes(row: dict | None) -> bool:
    if row is None:
        return False
    return any(s == "OK" for s in row["statuses_official"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hint-run", required=True)
    ap.add_argument("--hint-results", required=True)
    ap.add_argument("--anchor-results", required=True,
                    help="corrected s256 norepair results JSON")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    hint = load(args.hint_results)
    anchor = load(args.anchor_results)
    run_path = Path(args.hint_run)
    if not run_path.is_absolute():
        run_path = REPO_ROOT / run_path
    meta = json.loads(run_path.read_text(encoding="utf-8").splitlines()[0])
    names = meta.get("problem_names") or sorted(hint)

    print(f"C-hint cell: {run_path.name}")
    print(f"  hint_sha256: {meta.get('hint_sha256')}")
    print(f"  hint_text  : {meta.get('hint_text')}")
    print(f"  config     : K={meta.get('K')} steps={meta.get('steps')} "
          f"T={meta.get('temperature')} canvas={meta.get('canvas')} "
          f"seed={meta.get('base_seed')}")
    print()
    hdr = f"{'problem':44s} {'hint cell':14s} {'corrected s256':16s} flip"
    print(hdr)
    print("-" * len(hdr))
    n_hint = 0
    rows = []
    for name in names:
        h = hint.get(name)
        a = anchor.get(name)
        hp, ap = passes(h), passes(a)
        n_hint += hp
        flip = "PASS->pass" if (ap and hp) else \
               ("FAIL->PASS" if hp else ("PASS->fail" if ap else "held-fail"))
        print(f"{name:44s} {classify(h):14s} {classify(a):16s} {flip}")
        rows.append({"problem": name, "hint": classify(h),
                     "anchor": classify(a), "hint_pass": hp,
                     "anchor_pass": ap, "flip": flip})
    print("-" * len(hdr))
    print(f"\nquartet hint passes: {n_hint}/{len(names)}")
    if n_hint >= 3:
        verdict = ("INFORMATION-FIXABLE (>=3/4): the quartet's flips are "
                   "attributable to knowledge delivery; the mechanism claim "
                   "must live in the semantic/exclusion cohorts")
    elif n_hint <= 1:
        verdict = ("MODE-LOCK (<=1/4): the traps are not a knowledge gap — "
                   "steer-reachability becomes the more interesting claim")
    else:
        verdict = ("AMBIGUOUS (2/4): neither branch of the design's rule "
                   "fires; report as pre-registered partial")
    print(f"design §5 P0.5 rule: {verdict}")
    if args.json:
        Path(args.json).write_text(json.dumps(
            {"meta": {k: meta.get(k) for k in
                      ("hint_sha256", "hint_text", "K", "steps",
                       "temperature", "canvas", "base_seed")},
             "rows": rows, "hint_passes": n_hint, "verdict": verdict},
            indent=1), encoding="utf-8")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
