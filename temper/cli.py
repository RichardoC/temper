"""temper CLI (M1): gen / eval / draft.

Run with the project venv:  temper/.venv/bin/python -m temper.cli <cmd> ...
(or the bin/temper wrapper).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def cmd_gen(args) -> int:
    from temper.steer.l0 import run_l0

    summary = run_l0(
        args.task,
        k=args.K,
        steps=args.steps,
        canvas=args.canvas,
        base_seed=args.seed,
        sandbox=args.sandbox,
        cases=args.cases,
        timeout_ms=args.timeout_ms,
    )
    best = summary["best"] or {}
    print("\n=== L0 BEST ===")
    print(f"particle : {best.get('i')}")
    print(f"score    : {best.get('score')}")
    print(f"compile  : {best.get('compile_status')}")
    print(f"tests    : {best.get('test_status')}")
    if best.get("failing_seed"):
        print(f"seed     : {best['failing_seed']}")
    return 0 if best.get("test_status") == "pass" else 1


def cmd_eval(args) -> int:
    from temper.steer.l0 import eval_runs

    res = eval_runs(args.task, k=args.K, repeat=args.repeat)
    print(f"pass@k  task={res['task']} K={res['K']} "
          f"runs={res['runs_used']} "
          f"(runs with >=1 pass: {res['runs_with_at_least_one_pass']})")
    for k, v in sorted(res["pass_at_k"].items()):
        print(f"  pass@{k}: {v}")
    for r in res["per_run"]:
        print(f"  run {r['file']} steps={r['steps']} n_pass={r['n_pass']} "
              f"best_score={r['best_score']} selected={r['best_selected_test']}")
    return 0


def cmd_draft(args) -> int:
    from temper.authoring.draft import draft_properties

    return draft_properties(
        Path(args.task), approve=args.approve, model=args.model,
        write_draft_only=not args.approve,
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="temper",
        description="Diffusion-LM -> sandboxed proptest loop (M1: L0).",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gen", help="sample K candidates, verify, select best")
    g.add_argument("--task", required=True, help="bench task dir, e.g. bench/sort")
    g.add_argument("-K", type=int, default=8, help="number of sequential particles")
    g.add_argument("--steps", type=int, default=128, help="denoise steps per canvas")
    g.add_argument("--canvas", type=int, default=384, help="canvas length (new tokens)")
    g.add_argument("--seed", type=int, default=0, help="base seed (particle i: seed+i)")
    g.add_argument("--sandbox", default=None,
                   help="harness sandbox backend (default: seatbelt on macOS)")
    g.add_argument("--cases", type=int, default=256)
    g.add_argument("--timeout-ms", type=int, default=2000)
    g.set_defaults(func=cmd_gen)

    e = sub.add_parser("eval", help="pass@k table from run records")
    e.add_argument("--task", required=True)
    e.add_argument("-K", type=int, default=8)
    e.add_argument("--repeat", type=int, default=3,
                   help="use the last N run records with matching K")
    e.set_defaults(func=cmd_eval)

    d = sub.add_parser("draft", help="GLM-5.3 drafts properties.rs via Synthetic")
    d.add_argument("--task", required=True)
    d.add_argument("--approve", action="store_true",
                   help="after drafting, interactive y/N installs properties.rs "
                        "and stamps task.toml properties.approved=true")
    d.add_argument("--model", default="hf:zai-org/GLM-5.3")
    d.set_defaults(func=cmd_draft)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except PermissionError as exc:
        print(f"temper: REFUSED: {exc}", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"temper: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
