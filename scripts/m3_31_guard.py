#!/usr/bin/env python3
"""M3.1 OPS guard: the launch-time checks and the scoring-window discipline.

Every GPU phase in M3.1 uses this, unchanged from the P3 run (the guardrails are
operator-fixed and this lane adds nothing to them):

  * **generation**: Rancher/container CLOSED (0 procs, no ``/var/run/docker.sock``),
    no resident model process already running, memory free >= 50 %, and the
    frozen config sha logged.
  * **scoring window**: only AFTER the model process is ps-verified exited;
    the window opens Rancher, runs the container score, then quits and verifies
    the socket is gone before anything else runs.

The guard is a CHECKER, not a doer: it never starts or stops Rancher itself.
That is deliberate — a helper that silently opened a scoring window would make
the "no scoring during generation" claim unauditable, and the whole point of
the guard log is that the operator can read what was true at launch.

    # before a generation phase (exit 1 = do not launch)
    temper/.venv/bin/python scripts/m3_31_guard.py generation \\
        --log /tmp/m3_31_pgen_guard.log --phase P-gen

    # after the model is verified gone, before opening Rancher
    temper/.venv/bin/python scripts/m3_31_guard.py scoring \\
        --log /tmp/m3_31_pgen_score_guard.log --phase P-gen-score
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DOCKER_SOCKET = Path("/var/run/docker.sock")
RANCHER_PAT = re.compile(r"rancher|docker|vpnkit|com\.apple\.Virtualization",
                         re.IGNORECASE)
MODEL_PAT = re.compile(r"temper.*(dream|bench)|Dream-Coder|m3_31_|m3_p3_",
                       re.IGNORECASE)
# The guard must not match ITSELF (its own command line carries the pattern).
SELF_PAT = re.compile(r"m3_31_guard\.py")


def _ps() -> list[str]:
    out = subprocess.run(["ps", "-Ao", "pid=,command="],
                         capture_output=True, text=True, timeout=30)
    return [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]


def _matching(pattern: re.Pattern) -> list[str]:
    hits = []
    for line in _ps():
        if SELF_PAT.search(line):
            continue
        if pattern.search(line):
            hits.append(line[:200])
    return hits


def memory_free_pct() -> int | None:
    """System-wide memory free percentage as macOS reports it."""
    try:
        out = subprocess.run(["memory_pressure"], capture_output=True,
                             text=True, timeout=60).stdout
    except Exception:
        return None
    m = re.search(r"System-wide memory free percentage:\s*(\d+)%", out)
    return int(m.group(1)) if m else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("generation", "scoring", "status"))
    ap.add_argument("--log", default=None)
    ap.add_argument("--phase", default="")
    ap.add_argument("--min-free-pct", type=int, default=50)
    args = ap.parse_args(argv)

    from temper.steer.verdict import load_verdict_config
    cfg = load_verdict_config()

    rancher = _matching(RANCHER_PAT)
    models = _matching(MODEL_PAT)
    socket_present = DOCKER_SOCKET.exists()
    mem = memory_free_pct()

    lines = [f"=== M3.1 guard [{args.mode}] phase={args.phase or '-'} "
             f"{datetime.now(timezone.utc).strftime('%a %d %b %Y %H:%M:%S UTC')} ==="]
    lines.append(f"memory: System-wide memory free percentage: "
                 f"{mem if mem is not None else 'UNKNOWN'}%")
    lines.append(f"rancher/docker procs: {len(rancher)}")
    for hit in rancher[:5]:
        lines.append(f"  $ {hit}")
    lines.append(f"docker socket: "
                 f"{'PRESENT' if socket_present else 'absent'} ({DOCKER_SOCKET})")
    lines.append(f"model procs (temper/dream): {len(models)}")
    for hit in models[:8]:
        lines.append(f"  $ {hit}")
    lines.append(f"verdict_config sha256: {cfg.sha256} (version {cfg.version})")
    lines.append(f"harness_freeze: {_freeze_sha()}")

    ok = True
    if args.mode == "generation":
        if rancher:
            ok = False
            lines.append("REFUSE: Rancher/container processes are running — a "
                         "generation phase must run with the container CLOSED")
        if socket_present:
            ok = False
            lines.append("REFUSE: /var/run/docker.sock exists — container not "
                         "fully closed")
        if models:
            ok = False
            lines.append("REFUSE: a model/bench process is already running")
        if mem is None:
            ok = False
            lines.append("REFUSE: memory free percentage unreadable")
        elif mem < args.min_free_pct:
            ok = False
            lines.append(f"REFUSE: memory free {mem}% < {args.min_free_pct}%")
    elif args.mode == "scoring":
        if models:
            ok = False
            lines.append("REFUSE: a model/bench process is still alive — the "
                         "scoring window opens only after ps-verified exit")
        if rancher:
            lines.append("NOTE: Rancher already running; reuse it rather than "
                         "starting a second instance")
    lines.append(f"RESULT: {'OK' if ok else 'REFUSED'}")

    text = "\n".join(lines) + "\n"
    sys.stdout.write(text)
    if args.log:
        p = Path(args.log)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8") as f:
            f.write(text)
    return 0 if ok else 1


def _freeze_sha() -> str:
    from temper.steer.property import HARNESS_FREEZE_PATH
    if not HARNESS_FREEZE_PATH.exists():
        return "not frozen yet"
    try:
        raw = json.loads(HARNESS_FREEZE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return "unreadable"
    return str(raw.get("sha256") or "unstamped")


if __name__ == "__main__":
    raise SystemExit(main())
