#!/usr/bin/env python3
"""M2.3 steps-ablation table builder (report phase 5 tooling).

Reads (run record, norepair results JSON, repair results JSON) triples — one
per steps cell (96 from M2.2, 128, 256 from M2.3) — and prints the per-steps
report table plus the exclusion accounting. All inputs are committed
artifacts, so every number in docs/m2.3-report.md is reproducible:

    temper/.venv/bin/python scripts/m2_3_cells.py \
        bench/multipl-e/runs/20261019T103020Z-b2pilot.jsonl \
        bench/multipl-e/results/20261019T165244Z.json \
        bench/multipl-e/results/20261019T165450Z.json \
        [...repeat per cell...]

Passing only a run record (no results JSONs) prints generation-side stats
only (NFE/problem, gen wall, rows) — used for resume-safe partials before
scoring has run.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def load_run(path: str) -> dict:
    meta, rows, done = None, [], None
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        o = json.loads(line)
        if o.get("type") == "run":
            meta = o
        elif o.get("type") == "candidate":
            rows.append(o)
        elif o.get("type") == "done":
            done = o
    return {"meta": meta, "rows": rows, "done": done}


def gen_stats(run: dict) -> dict:
    rows = run["rows"]
    meta = run["meta"] or {}
    nfe = [r.get("nfe") or 0 for r in rows]
    walls = [r.get("gen_wall_s") or 0.0 for r in rows]
    return {
        "steps": meta.get("steps"),
        "n_rows": len(rows),
        "n_problems_started": len({r["problem"] for r in rows}),
        "problems_planned": meta.get("n_problems"),
        "nfe_per_problem": (sum(nfe) / max(1, len(rows))
                            * (meta.get("K") or 1) / max(1, (meta.get("K") or 1))),
        "nfe_per_candidate": sum(nfe) / max(1, len(nfe)),
        "nfe_total": sum(nfe),
        "gen_wall_s_total": (run["done"] or {}).get("gen_wall_s_total")
                            or round(sum(walls), 2),
        "extract_ok_rows": sum(1 for r in rows if r.get("extract_ok")),
    }


def score_stats(results_path: str) -> dict:
    d = json.loads(Path(results_path).read_text(encoding="utf-8"))
    agg = d["aggregate"]
    per = d["per_problem"]
    scored = sum(p["n_scored"] for p in per)
    syn = sum(1 for p in per for s in p["statuses_official"]
              if s == "SyntaxError")
    exc = agg.get("honest-official", {}).get("n_excluded_rows")
    return {
        "scored_rows": scored,
        "syntaxerror_share_of_scored": round(syn / scored, 4) if scored else None,
        "official_p1": agg["official"]["pass@1"],
        "official_p4": agg["official"]["pass@4"],
        "honest_p1": agg["honest-official"]["pass@1"],
        "honest_p4": agg["honest-official"]["pass@4"],
        "excluded_rows": exc,
        "n_problems_scored": len(per),
        "input_dir": Path(d["input_dir"]).name,
    }


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    i = 1
    cells = []
    while i < len(argv):
        run = load_run(argv[i])
        g = gen_stats(run)
        s = score_stats(argv[i + 1]) if i + 1 < len(argv) \
            and argv[i + 1].endswith(".json") else None
        r = score_stats(argv[i + 2]) if s is not None and i + 2 < len(argv) \
            and argv[i + 2].endswith(".json") else None
        cells.append((g, s, r))
        i += 3 if (s is not None and r is not None) else \
            (2 if s is not None else 1)

    hdr = ("steps", "rows/scored", "SyntaxErr share", "off p@1", "off p@4",
           "hon p@1", "hon p@4", "repair delta (hon p@1)", "NFE/prob",
           "gen_wall_h", "excluded")
    print("\t".join(hdr))
    for g, s, r in cells:
        nfe_prob = g["nfe_per_candidate"] * 4  # K=4 canvases per problem
        row = [
            str(g["steps"]),
            f"{g['n_rows']}/{s['scored_rows'] if s else '-'}",
            f"{s['syntaxerror_share_of_scored']:.2f}" if s and
            s["syntaxerror_share_of_scored"] is not None else "-",
            f"{s['official_p1']:.3f}" if s else "-",
            f"{s['official_p4']:.3f}" if s else "-",
            f"{s['honest_p1']:.3f}" if s else "-",
            f"{s['honest_p4']:.3f}" if s else "-",
            (f"{r['honest_p1'] - s['honest_p1']:+.3f}"
             if s and r else "-"),
            f"{nfe_prob:.0f}",
            f"{g['gen_wall_s_total'] / 3600:.2f}",
            str(s["excluded_rows"]) if s else
            f"{g['n_rows'] - g['extract_ok_rows']} (pre-score)",
        ]
        print("\t".join(row))
    for g, s, r in cells:
        tag = f"steps={g['steps']}"
        print(f"\n[{tag}] rows={g['n_rows']} "
              f"problems_started={g['n_problems_started']}"
              f"/{g['problems_planned']} "
              f"extract_ok={g['extract_ok_rows']} "
              f"nfe_total={g['nfe_total']} "
              f"gen_wall={g['gen_wall_s_total']}s")
        if s:
            print(f"[{tag}] scored={s['scored_rows']} "
                  f"problems_scored={s['n_problems_scored']} "
                  f"export={s['input_dir']}")
        if s and r:
            print(f"[{tag}] repair-edition honest p@1={r['honest_p1']} "
                  f"(norepair {s['honest_p1']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
