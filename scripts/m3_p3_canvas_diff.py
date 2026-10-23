#!/usr/bin/env python3
"""C-full vs C-screen: the emitted canvas PAIR table (engine forensics).

Design rev2 V3 re-roled C-full@2048 as the in-loop engine row and declared that
a divergence between the in-loop engine and the offline C-screen counterfactual
is **engine forensics, not mechanism evidence**. "Diff every problem's emitted
canvas pair table" (the phase brief) is this script.

For every problem it puts side by side:

  * C-screen's chosen canvas (offline: best R1-clean by confidence over the
    C-null ensemble) — index, temperature, confidence, container status;
  * C-full's emitted canvas (in-loop: the same rule applied inside the game,
    where the ensemble is the ladder the loop actually ran) — same columns,
    read from the game's D2 steer log ``emit`` event.

Three outcomes per problem, and they mean different things:

  SAME_EVIDENCE  both chose the same (index, temperature, confidence) — the
                 engine and the counterfactual agree; any outcome difference
                 is container noise, which is impossible for a deterministic
                 scorer on identical text, so this class should always agree
                 on status too.
  SAME_OUTCOME   different canvas, same container status. Expected and healthy:
                 the loop's canvases are different draws.
  DIVERGED       different container status. THE forensics cell: either the
                 loop's ladder produced a canvas the counterfactual never had
                 (mechanism-adjacent), or the emission rule fired differently
                 (rule implementation bug — must be explained, not reported).

    temper/.venv/bin/python scripts/m3_p3_canvas_diff.py \\
        --screen-detail /tmp/m3_p3/emission-detail.json \\
        --cfull-steer-dir bench/multipl-e/runs/<ts>-c-full.jsonl.steer \\
        --cfull-results /tmp/m3_p3/c-full-emission-results.json \\
        --json /tmp/m3_p3/canvas-pair-diff.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def load_cscreen(path: str) -> dict:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return {row["problem"]: row for row in d["rows"]}


def load_cfull_emits(steer_dir: Path) -> dict:
    """{problem: emit event} from the D2 steer logs (one log per problem)."""
    out = {}
    for log_path in sorted(steer_dir.glob("*.jsonl")):
        emit = None
        for line in log_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            ev = json.loads(line)
            if ev.get("kind") == "emit":
                emit = ev
        if emit is not None:
            out[log_path.stem] = emit
    return out


def load_statuses(path: str) -> dict:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    out = {}
    for row in d["per_problem"]:
        st = row.get("statuses_official") or []
        out[row["name"]] = {
            "status": st[0] if st else None,
            "excluded": bool(row.get("n_excluded")),
            "emitted": row.get("emitted"),
            "emitted_temp": row.get("emitted_temp"),
        }
    return out


def classify(screen: dict, cfull_status: dict, emit: dict, n_ladder: int) -> dict:
    """Classify one problem's pair."""
    s_emitted = screen.get("screen_emitted")
    s_temp = screen.get("screen_temp")
    s_status = screen.get("screen_status")
    c_emitted = cfull_status.get("emitted")
    c_temp = emit.get("temperatures", {}).get(str(c_emitted)) if emit else None
    c_status = cfull_status.get("status")
    c_confs = (emit or {}).get("mean_commit_conf", {}) or {}
    s_conf = None
    if s_emitted is not None and emit:
        s_conf = c_confs.get(str(s_emitted))

    same_slot = (s_emitted == c_emitted
                 and s_temp is not None and c_temp is not None
                 and abs(float(s_temp) - float(c_temp)) < 1e-9)
    # Outcome comparison treats an export exclusion as its own (failing) class
    s_outcome = "EXCLUDED" if screen.get("screen_status") is None else s_status
    c_outcome = "EXCLUDED" if cfull_status.get("excluded") else c_status
    # A side with no emitted canvas at all is MISSING data, not a divergence —
    # reporting it as DIVERGED would manufacture forensics work out of an
    # absent log.
    if c_emitted is None or not emit:
        verdict = "MISSING"
    elif same_slot:
        verdict = "SAME_EVIDENCE"
    elif s_outcome == c_outcome:
        verdict = "SAME_OUTCOME"
    else:
        verdict = "DIVERGED"
    return {
        "screen_emitted": s_emitted, "screen_temp": s_temp,
        "screen_status": s_outcome, "screen_conf": s_conf,
        "cfull_emitted": c_emitted, "cfull_temp": c_temp,
        "cfull_status": c_outcome, "cfull_conf": (
            c_confs.get(str(c_emitted)) if emit else None),
        "cfull_final_clean": (emit or {}).get("final_clean"),
        "cfull_n_emittable": len((emit or {}).get("mean_commit_conf", {}) or {}),
        "verdict": verdict,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--screen-detail", required=True,
                    help="emission-detail.json from scripts/m3_p3_emissions.py")
    ap.add_argument("--cfull-steer-dir", required=True,
                    help="<c-full run>.steer/ (one D2 log per problem)")
    ap.add_argument("--cfull-results", required=True,
                    help="C-full emission results JSON (the emitted outcomes)")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    screen = load_cscreen(args.screen_detail)
    emits = load_cfull_emits(Path(args.cfull_steer_dir))
    statuses = load_statuses(args.cfull_results)

    print(f"{'problem':44s} {'C-screen':>22s} {'C-full':>22s}  verdict")
    print("-" * 100)
    rows, tally = [], {}
    for problem in sorted(statuses):
        s = screen.get(problem, {})
        st = statuses[problem]
        emit = emits.get(problem)
        rec = classify(s, st, emit, 0)
        rec["problem"] = problem
        rows.append(rec)
        tally[rec["verdict"]] = tally.get(rec["verdict"], 0) + 1

        def cell(em, tp, stat):
            if em is None:
                return "-"
            return f"p{em}@{tp} {stat}"
        print(f"{problem:44s} "
              f"{cell(rec['screen_emitted'], rec['screen_temp'], rec['screen_status']):>22s} "
              f"{cell(rec['cfull_emitted'], rec['cfull_temp'], rec['cfull_status']):>22s}  "
              f"{rec['verdict']}")

    print("\n=== tally ===")
    for k in ("SAME_EVIDENCE", "SAME_OUTCOME", "DIVERGED", "MISSING"):
        if tally.get(k, 0) or k != "MISSING":
            print(f"  {k:14s} {tally.get(k, 0)}")
    diverged = [r["problem"] for r in rows if r["verdict"] == "DIVERGED"]
    if diverged:
        print(f"\nDIVERGED problems (forensics required, NOT mechanism): "
              f"{', '.join(diverged)}")
        print("  For each: is the divergence a LADDER difference (the loop ran "
              "canvases the counterfactual never had) or a RULE difference "
              "(both saw the same canvas and emitted differently)? A rule "
              "difference is an engine bug and must be fixed, not reported.")
    missing_logs = [p for p in statuses if p not in emits]
    if missing_logs:
        print(f"\n!! no steer log for: {missing_logs}")

    payload = {"rows": rows, "tally": tally, "diverged": diverged,
               "screen_detail": args.screen_detail,
               "cfull_steer_dir": args.cfull_steer_dir,
               "cfull_results": args.cfull_results}
    if args.json:
        Path(args.json).write_text(json.dumps(payload, indent=1), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
