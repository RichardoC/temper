#!/usr/bin/env python3
"""Freeze / verify the steer verdict config (M3.0 D7, docs/m3-design.md §7).

The informed-class list + patience-rule parameters are a pre-commitment
against in-sample reward shaping: they are frozen with a sha BEFORE any C-cell
launches, and that sha is recorded in every C-cell run record. This tool is
the scaffold that makes freezing a mechanical, auditable act:

    # stamp (or re-stamp) the canonical sha the file carries
    temper/.venv/bin/python scripts/m3_freeze_verdict_config.py --write

    # verify the committed file is untouched (exit 1 on tamper)
    temper/.venv/bin/python scripts/m3_freeze_verdict_config.py --check

Re-pinning is a LABELED decision, not a silent edit: D4 (ghost-closure
validity on real canvases) sets the patience parameters at the P0->P2
boundary, bumps ``version``, and re-stamps — both shas go in the report.

The digest is :func:`temper.steer.verdict.compute_config_sha256` — canonical
JSON (sorted keys, no whitespace) of the object WITHOUT its ``sha256`` field —
so formatting/indentation churn never changes the frozen identity, but any
semantic edit does.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from temper.steer.verdict import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    compute_config_sha256,
    load_verdict_config,
)


def main(argv: list[str]) -> int:
    args = argv[1:]
    if not args or args[0] not in ("--write", "--check"):
        print(__doc__)
        return 2
    path = Path(args[1]) if len(args) > 1 else DEFAULT_CONFIG_PATH
    if not path.exists():
        print(f"{path}: not found", file=sys.stderr)
        return 2
    raw = json.loads(path.read_text(encoding="utf-8"))

    if args[0] == "--write":
        payload = dict(raw, sha256=compute_config_sha256(raw))
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"froze {path}: version={payload.get('version')} "
              f"sha256={payload['sha256']}")
        return 0

    try:
        cfg = load_verdict_config(path)
    except Exception as exc:  # VerdictConfigError or JSON errors
        print(f"TAMPER/INVALID: {exc}", file=sys.stderr)
        return 1
    print(f"ok {path}: version={cfg.version} sha256={cfg.sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
