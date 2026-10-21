"""temper CLI (M2): gen / eval / draft / export-multipl-e / score-multipl-e.

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
    fc = res.get("failure_classes") or {}
    if fc.get("task"):
        agg = " ".join(f"{k}={v}" for k, v in sorted(fc["task"].items()))
        print(f"failure classes (task, last {res['runs_used']} runs): {agg}")
        for fname, h in fc.get("per_run", {}).items():
            print(f"  run {fname}: " + " ".join(f"{k}={v}" for k, v in sorted(h.items())))
    return 0


def cmd_export_multipl_e(args) -> int:
    from temper.multipl_e.export import export_runs

    manifest = export_runs(
        [Path(p) for p in args.runs],
        Path(args.out),
        benchmark=args.benchmark,
        also_unsteered_guise=args.also_unsteered_guise,
        repair=args.repair,
    )
    c = manifest["counts"]
    print(f"export (repair={'on' if manifest.get('repair') else 'off'}): "
          f"{c['exported_completions']} completions, "
          f"{c['excluded_rows']} excluded rows (honest counting, see manifest)")
    for p in manifest["problems"]:
        print(f"  {p['file']}: {p['completions']} completions ({p['flavor']})")
    for e in manifest["excluded"]:
        print(f"  EXCLUDED run={e['run']} i={e['i']}: {e['reason'][:110]}")
    return 0


def cmd_score_multipl_e(args) -> int:
    from temper.multipl_e.score import score_dir

    res = score_dir(
        Path(args.input),
        version=args.version,
        max_workers=args.max_workers,
        results_dir=Path(args.results_dir) if args.results_dir else None,
    )
    print(f"scored {res['input']} version={res['versions']} "
          f"-> {res['results_path']}")
    return 0


def cmd_bench_multipl_e(args) -> int:
    from temper.multipl_e.bench import run_bench

    n = None if args.problems == "all" else int(args.problems)
    sample_fn = None
    model = milestone = None
    if args.engine == "qwen":
        # M2.4b row B3: the AR twin through the SAME driver + record format.
        if args.steps != 0:
            raise SystemExit(
                "--steps must be 0 for the qwen engine (AR has no denoise "
                "steps; --canvas is the max_new_tokens budget)")
        from temper.model.qwen import MODEL_ID as QWEN_MODEL_ID, sample as qwen_sample
        sample_fn = qwen_sample
        model = QWEN_MODEL_ID
        milestone = "M2.4b-B3"
    problem_indices = None
    if args.problem_indices:
        problem_indices = [int(x) for x in args.problem_indices.split(",") if x.strip()]
    temps = None
    if args.temperatures:
        temps = [float(x) for x in args.temperatures.split(",") if x.strip()]
    hint_line = None
    if args.hint_file:
        hint_line = Path(args.hint_file).read_text(encoding="utf-8").rstrip("\n")
    res = run_bench(
        args.benchmark,
        n_problems=n,
        k=args.K,
        steps=args.steps,
        canvas=args.canvas,
        base_seed=args.seed,
        temperature=args.temperature,
        alg=args.alg,
        out=Path(args.out),
        resume=args.resume,
        sample_fn=sample_fn,
        model=model,
        milestone=milestone,
        problem_indices=problem_indices,
        temperatures=temps,
        hint_line=hint_line,
    )
    s = res["summary"]
    return 0 if s["n_rows"] else 1


def cmd_steer_replay(args) -> int:
    """Re-derive every steer decision from a game log, with NO GPU (D2)."""
    from temper.steer.log import ReplayMismatch, read_game_log, replay_game
    from temper.steer.verdict import VerdictConfigError, load_verdict_config

    try:
        cfg = load_verdict_config(args.config)
    except VerdictConfigError as exc:
        print(f"temper steer-replay: verdict config refused: {exc}",
              file=sys.stderr)
        return 2
    paths = [Path(p) for p in args.logs]
    if args.dir:
        paths.extend(sorted(Path(args.dir).glob("*.jsonl")))
    if not paths:
        print("temper steer-replay: no logs given (pass paths or --dir)",
              file=sys.stderr)
        return 2
    rc = 0
    for path in paths:
        events = read_game_log(path)
        meta = next((e for e in events if e.get("kind") == "meta"), {})
        recorded = meta.get("verdict_config_sha256")
        if recorded and recorded != cfg.sha256:
            print(f"  {path.name}: REFUSED — log was produced under verdict "
                  f"config sha {recorded}, current is {cfg.sha256} "
                  "(replay under the same freeze or fetch the versioned one)")
            rc = 1
            continue
        try:
            decisions = replay_game(events, cfg)
        except ReplayMismatch as exc:
            print(f"  {path.name}: REPLAY MISMATCH — {exc}")
            rc = 1
            continue
        counts = {}
        for d in decisions:
            counts[d.action] = counts.get(d.action, 0) + 1
        summary = " ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        print(f"  {path.name}: {len(decisions)} decisions replay green "
              f"({summary or 'none'})")
    return rc


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

    r = sub.add_parser(
        "steer-replay",
        help="re-derive steer decisions from a schema-1.3 game log (no GPU, D2)")
    r.add_argument("logs", nargs="*", help="<out>.steer/<problem>.jsonl files")
    r.add_argument("--dir", default=None,
                   help="also replay every *.jsonl in this steer-log dir")
    r.add_argument("--config", default=None,
                   help="verdict config to replay under (default: the frozen "
                        "repo-root steer_verdict_config.json)")
    r.set_defaults(func=cmd_steer_replay)

    d = sub.add_parser("draft", help="GLM-5.3 drafts properties.rs via Synthetic")
    d.add_argument("--task", required=True)
    d.add_argument("--approve", action="store_true",
                   help="after drafting, interactive y/N installs properties.rs "
                        "and stamps task.toml properties.approved=true")
    d.add_argument("--model", default="hf:zai-org/GLM-5.3")
    d.set_defaults(func=cmd_draft)

    x = sub.add_parser(
        "export-multipl-e",
        help="map temper run records to MultiPL-E completion files",
    )
    x.add_argument("--benchmark", default=None,
                   help="vendored benchmark to join against (humaneval-rs | mbpp-rs); "
                        "temper bench tasks are exported as synthetic problems instead")
    x.add_argument("--runs", nargs="+", required=True, help="run record .jsonl files")
    x.add_argument("--out", required=True, help="output dir for completion files + manifest.json")
    x.add_argument("--also-unsteered-guise", action="store_true",
                   help="also emit the same completions under an 'unsteered model' "
                        "guise name (format-blindness demo)")
    x.add_argument("--no-repair", dest="repair", action="store_false",
                   help="disable the L0.5 repair round at export (M2.2 ablation "
                        "column: repair=off vs the default repair=on)")
    x.set_defaults(func=cmd_export_multipl_e)

    s = sub.add_parser(
        "score-multipl-e",
        help="score a completion-file dir through the official MultiPL-E container",
    )
    s.add_argument("--input", required=True, help="dir of completion files (export-multipl-e out)")
    s.add_argument("--version", default="both", choices=["official", "normalized", "both"],
                   help="official (Rust 2015 default) and/or normalized (--edition=2021)")
    s.add_argument("--max-workers", type=int, default=2)
    s.add_argument("--results-dir", default=None,
                   help="results dir (default bench/multipl-e/results)")
    s.set_defaults(func=cmd_score_multipl_e)

    b = sub.add_parser(
        "bench-multipl-e",
        help="MultiPL-E generation driver (unsteered baseline, M2.2 row B2)",
    )
    b.add_argument("--benchmark", required=True,
                   help="vendored benchmark (humaneval-rs | mbpp-rs)")
    b.add_argument("--problems", default="all",
                   help="first N problems in vendored (stable) order, or 'all'")
    b.add_argument("--problem-indices", default=None,
                   help="explicit cohort: comma-separated 0-based vendored "
                        "indices, e.g. '0,4,20,21' (M3 cohort cells). "
                        "Mutually exclusive with a numeric --problems")
    b.add_argument("--temperatures", default=None,
                   help="per-canvas temperature list, comma-separated, e.g. "
                        "'0.4,0.8,1.2' — gives each problem len(list) canvases "
                        "and records each row's temperature (M3 P1 probe)")
    b.add_argument("--hint-file", default=None,
                   help="file whose first line-block is appended to the chat "
                        "prompt as ONE extra line; text+sha256 land in the run "
                        "meta (M3 P0.5 C-hint knowledge-injection null)")
    b.add_argument("-K", type=int, default=4, help="samples per problem")
    b.add_argument("--steps", type=int, default=96, help="denoise steps per canvas "
                   "(Dream engine; must be 0 for the qwen AR engine)")
    b.add_argument("--engine", default="dream", choices=["dream", "qwen"],
                   help="sampling engine: dream (dLLM diffusion canvas, B2) or "
                        "qwen (Qwen2.5-Coder-7B-Instruct AR decode, B3 twin) — "
                        "same driver, prompt template, seeds and record schema")
    b.add_argument("--canvas", type=int, default=384, help="canvas length (new tokens)")
    b.add_argument("--seed", type=int, default=0,
                   help="base seed; sample i of every problem uses seed+i")
    b.add_argument("--temperature", type=float, default=0.2,
                   help="sampling temperature for ALL samples (MultiPL-E pass@1 "
                        "convention: T=0.2; the pass@10/100 convention T=0.8 is "
                        "for larger-K runs)")
    b.add_argument("--alg", default="entropy",
                   help="remasking algorithm (model card default: entropy)")
    b.add_argument("--out", required=True,
                   help="run record .jsonl path (raw sidecars at <out>.raw/)")
    b.add_argument("--resume", action="store_true",
                   help="continue an existing record: skip already-sampled "
                        "(problem, sample) pairs; config must match")
    b.set_defaults(func=cmd_bench_multipl_e)
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
