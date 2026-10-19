# M2 design: evaluation instrument (M2.0 adapter) + salvage layer (M2.1)

Owner of the why: `docs/benchmarks-decision.md` (instrument before engine).
Owner of the failure data: `docs/m1.5-report.md` (repair-pass evidence, taxonomy).

## M2.0 — MultiPL-E evaluation adapter

**Goal**: temper can score model generations against the official MultiPL-E Rust
benchmarks (156 humaneval-rs + 354 mbpp-rs tasks) through the official container,
without our steering code touching scoring internals.

Scope:
1. **Vendored dataset**: fetch `humaneval-rs` and `mbpp-rs` task defs from the HF
   datasets-server rows API (`huggingface_hub` is in the venv; no pyarrow) and
   commit as `bench/multipl-e/data/{humaneval-rs,mbpp-rs}.json` with BSD-3
   attribution note + dataset revision recorded. One-time fetch, offline after.
2. **Export**: `temper export-multipl-e --benchmark humaneval-rs --runs <run.jsonl>... --out DIR`
   maps our candidate rows to MultiPL-E completion files
   `{name, prompt, tests, completions, language:"rs"}`.
   - **GOTCHA (verified)**: strip all stop markers / fence residue from completions;
     the scorer does not enforce stop_tokens and duplicated closing braces become
     spurious SyntaxErrors. Reuse the extract pipeline; export only extract_ok rows,
     but ALSO record excluded rows in a sidecar manifest (honest counting — an
     extraction failure is a scored failure via zero completions).
3. **Runner**: `temper score-multipl-e --input DIR --version official|normalized`
   - `official`: the pulled `ghcr.io/nuprl/multipl-e-evaluation` image (amd64 via
     Rosetta; use `docker cp` not bind mounts — verified macOS file-sharing artifact).
   - `normalized`: same image, entrypoint patched to add `--edition=2021` to rustc
     invocations. Both scored on every benchmark run; report both columns.
     Rationale (decision doc §container): official default is Rust 2015; our model
     emits 2018+ idioms (`dyn Trait`). The delta is itself a reportable finding.
   - pass@k via the repo's `pass_k.py` port (numpy in venv) into
   `bench/multipl-e/results/<ts>.json` with per-problem and unbiased pass@1/10/100.
4. **NFE accounting** (run-record schema v1.2, backward compatible):
   candidate rows already have s/step × steps → add explicit `nfe` (forward passes
   actually executed per candidate) and run-row `nfe_total`, `verifier_wall_s_total`
   (already have harness_wall_s per candidate — sum it). New fields nullable for
   old records. This is the x-axis for every curve going forward.

## M2.1 — L0.5 salvage + failure taxonomy

**Goal**: stop losing salvageable candidates (M1.5: p0 was one stray block from
passing) and make failure classes machine-readable.

1. **Repair pass** in `temper/model/extract.py` (deterministic, conservative,
   unit-tested to exhaustion): given extracted code that fails compile/must_define,
   try in order, each attempt recorded: (a) drop unreferenced degenerate items
   (brace-scan: any item whose identifier is never referenced and whose removal
   keeps the balance — e.g. stray `pub RingBuf { fn new(..) }` blocks), (b) strip
   trailing prose after final code block. Record per row:
   `repair_attempts: [...]`, `repaired: true|false`. Never re-prompt the model.
2. **Failure taxonomy**: classify each non-scoring row from raw sidecar + extract
   reason: `impl_method | renamed_or_self | truncated_canvas | prose_only | echo |
   compile_error | test_fail`. Pure functions in extract or a new
   `temper/model/classify.py`; no LLM.
3. **eval**: `temper eval` gains a failure-class histogram per run and per task.
4. **Diag persistence**: candidate rows gain `diag_msg` (primary rustc error string,
   truncated 500 chars) so E-code-less errors stop being invisible.

## Constraints (binding, same as M1/M1.5)
- Python stdlib-only tool; venv has torch/transformers/huggingface_hub/numpy — no new pip packages.
- M0 harness contract untouched; `python3 scripts/demo_m0.py` must pass green at the end.
- One-writer: this lane owns `~/temper` exclusively until report commit.
- Commit after each numbered item; `docs/m2.0-report.md` (covers both) at the end.
- Cargo prefix: `export PATH="/opt/homebrew/opt/rustup/bin:$PATH"`.
- Do NOT touch `/Users/richardf/temper-spike-m1`.
- Generation runs for verification only if needed (GPU time is fine); prefer scoring
  pre-existing bench run records for the adapter smoke test — and record one
  Qwen-style re-export of an L0 run as the format demo.

## Acceptance checklist (for parent verification)
- [ ] Vendored dataset files with counts 156/354 verifiable (wc/len)
- [ ] Export/score round trip: score one existing `bench/sort` L0 run through the
      official container; results JSON exists; numbers sane
- [ ] Both `official` and `normalized` edition columns produced
- [ ] NFE fields present on a fresh small generation run (K=1, steps=8)
- [ ] Repair pass converts the stored run-B p0 raw text to (at least) compile-ok
- [ ] Taxonomy histogram matches M1.5's hand-counts on run B (4 rows)
- [ ] demo_m0 all-pass, all unit tests green
- [ ] Report committed
