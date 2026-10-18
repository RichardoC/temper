# temper — M1 report

Milestone: **M1 — L0: best-of-K sampling + verifier reranking** (design:
`docs/m1-design.md`, incl. §0 spike corrections). Machine: MacBook Pro M3 Max
(16 cores), 48 GB unified, macOS arm64, python3 3.9.6, venv
`temper/.venv` (torch 2.8.0, transformers 4.57.6, accelerate 1.10.1,
huggingface_hub 0.36.2, tomli). Model from the shared HF cache (no re-download,
no cache mutation, no monkeypatching).

## Verdict (3 lines)

1. **M1 works end-to-end**: `temper gen --task bench/sort -K 4 --steps 96`
   produced a 4-candidate scored JSONL run record; **2/4 candidates compiled
   and FAILED proptest (wrong-direction sort bugs, shrunk to minimal
   counterexamples), 2/4 passed** — the verifier reranked real, fixable model
   mistakes and the selected winner passes tests its siblings fail
   (acceptance #1 and #2 on the same run).
2. **Steps sweep (sort, K=2)**: steps 64/96/128 → pass rate 1/2, 0/2 (+2/4 at
   96 from the K=4 run), 2/2 — quality rises with steps as the spike
   predicted; **M1 default fixed at steps=128** (canvas 384), with 96 as the
   budget option. Measured cost: 0.92–0.94 s/step (entropy, fresh process),
   **1.97 s/step for topk_margin** and ~1.9–2.1 s/step for later particles in
   long sequential runs (sustained-load effect) — see deviations.
3. **GLM-5.3 drafting proven live** on Synthetic (auth+roundtrip 1.5–1.7 s on
   a probe; full draft = 4655 chars streamed in ~14 min): the drafted
   wrap-sum property compiles, **passes the correct candidate and fails the
   buggy one** (boundary-weighted `prop_oneof!` over u64::MAX edges +
   `catch_unwind` panic detection).

## What was built (commits)

- `b7850e2` **M1: model runtime (Dream-Coder MPS, K=1 sequential)** —
  `temper/model/runtime.py` (watermark env before torch import; mps+bf16
  required, loud CPU debug fallback; resident loader;
  `model.config.use_cache = False` right after load; memory probes),
  `temper/model/dream.py` (K sequential canvases over the model's own
  `diffusion_generate(output_history=True, return_dict_in_generate=True)`;
  tokens hook = per-step `torch.mps.empty_cache()` + committed-token/memory
  stats; logits hook = fp32 chunked top-1 confidence stats, never cloning
  logits; per-particle temp {0.2,0.4,0.6,0.8} + entropy/topk_margin
  alternation), `temper/model/extract.py` (fenced-block → bare-span
  heuristics, must_define symbol check, mask-token rejection; 12 unit tests),
  venv + `.gitignore`.
- `3f4da3e` **M1: L0 steering loop + CLI** — `temper/steer/l0.py`
  (K sequential → extract → M0 `run_candidate` → score 100·1[compiles] +
  1000·1[tests] − 5·reject_fraction, tie-break mean commit confidence →
  JSONL run record with per-candidate particle config, compile status,
  diagnostic codes+spans, test status, minimal failing input + persisted
  seed, model/harness wall times; failing seeds persisted to
  `bench/<task>/corpus/<strategy_hash>/`; `temper eval` = unbiased pass@k
  (Chen et al. 2021 product form) over run records), `temper/cli.py` +
  `bin/temper` (gen/eval/draft).
- `0943a07` **M1: GLM-5.3 property drafting** — `temper/authoring/draft.py`
  (stdlib urllib+json only; SSE streaming; env-then-models.json key;
  `properties.draft.rs` written, `--approve` gated behind interactive y/N +
  `task.toml properties.approved = true`; never auto-approves),
  `docs/bench-tasks.md` (procedure for adding tasks).
- This commit — bench suite (sort / ringbuf / wrap-sum), acceptance + sweep
  runs, this report.

## Acceptance criteria (design §7)

1. **End-to-end gen on one task with K scored candidates + best** — DONE:
   `bench/sort/runs/20261018T203659Z.jsonl` (K=4, steps=96, all fields:
   particle config, compile status, diag codes, test status, minimal input,
   persisted seed, wall times both sides, memory snapshot; `best` row).
2. **Verifier did real work (winner beats failing siblings)** — DONE on the
   FIRST acceptance run: sort K=4: p0/p2 compile ok but proptest FAIL
   (`got [279524015, 0], want [0, 279524015]` / `got [0, -1], want [-1, 0]`
   — descending-sort bugs shrunk to 2-element counterexamples), p1/p3 PASS;
   best = p3. Failing seeds persisted to
   `bench/sort/corpus/<strategy_hash>/` (2 seed files, `cc` persisted-seed
   format, replayable per M0).
3. **Bench suite: 3 tasks with human-approved properties** — DONE:
   `bench/sort` (multiset-preserving ascending sort), `bench/ringbuf`
   (capacity-bounded FIFO ring buffer, push-overwrites-oldest semantics —
   the empty/full confusion class), `bench/wrap-sum` (u64 wrapping sum that
   must not panic in debug). Third task chosen as design §7's own
   "int-add-with-overflow-wrap" suggestion instead of a leaky counter:
   it has an exact oracle (u64 wrapping arithmetic), and the canonical bug
   (plain `+`/`.sum()` panicking under the debug profile) is exactly the
   verifier-has-tears shape we want. All 3 verified: example candidates PASS
   (seatbelt), deliberately buggy candidates FAIL with shrunk minimal inputs
   (e.g. ringbuf drop-newest bug → `(2, [0, 0, -1])`; wrap-sum plain sum →
   `attempt to add with overflow` on `[12085889504598553676, 6360854569110997940]`).
   Procedure for more tasks: `docs/bench-tasks.md`.
4. **This report** — pass@k table, timing shares, memory peak, spike check,
   deviations, M2 issues — below.

## pass@k (temper eval; unbiased estimator, averaged over run records)

sort, K=2, repeat=3 (runs at steps 64/96/128 — see sweep caveat):

| task | K | runs | pass@1 | pass@2 |
|---|---|---|---|---|
| sort | 2 | 3 | 0.500 | 0.667 |

sort, K=4, steps=96 (1 run so far; a second K=4 run did not fit the time
budget):

| task | K | runs | pass@1 | pass@2 | pass@4 |
|---|---|---|---|---|---|
| sort | 4 | 1 | 0.500 | 0.833 | 1.000 |

(The K=2 table mixes steps 64/96/128 — that is the sweep material, not a
controlled pass@k; treat pass@2=0.667 as "best-of-2 across settings".)

## Steps sweep (bench/sort, canvas 384, seeds 21/22/23; K=4 run at 96 = seed 11)

| steps | tok/step | K | pass | compile ok | extract ok | gen wall/candidate | notes |
|---|---|---|---|---|---|---|---|
| 64  | 6.0 | 2 | 1/2 | 0(+1 n/a) | 1/2 | 58.8 s (0.92 s/step) / 126.0 s (1.97) | p0: full canvas, no code — 6 tok/step parallel decode → prose, extract rejects (spike's "degraded" regime) |
| 96  | 4.0 | 2+4 | 2/6 | 4/6 | 5/6 | 88–197 s | canonical failure: descending sort; 1 compile_error; 1 impl-methods-instead-of-free-fns rejected by must_define |
| 128 | 3.0 | 2 | 2/2 | 2/2 | 2/2 | 118.2 s (0.92) / 252.8 s (1.97) | cleanest run |

On the harder ringbuf task the same sweep direction held (steps=96: 0/4
usable; steps=128: 1/2 extracted, still compile_error) — more steps help
usability but task difficulty dominates.

**M1 default: steps=128, canvas=384.** 96 is acceptable when quality ~50% is
enough; 64 degrades (parallel decode → non-code). This matches the spike
prediction (32 steps at canvas 256 degraded; ≥96 needed for clean code).

## Timing: model-side vs harness-side shares

From the four sort run records (K=4@96: 668.6 s model vs 16.9 s harness;
K=2@64: 193.1 vs 5.1; K=2@96: 288.7 vs 7.5; K=2@128: 379.2 vs 9.7):

| stage | share of cycle |
|---|---|
| model generation (sequential K canvases) | **96.4–97.4%** |
| harness (emit+cache+check+build+run, seatbelt, 256 cases) | **2.6–3.6%** (~1.7–5.9 s/candidate) |
| extraction + scoring | <0.1% (ms) |

Per-candidate harness wall (compile+run, includes ~0.3 s sandbox-wrapped
build) ≈ 1.75–5.9 s — consistent with M0's 236 ms warm-build measurement plus
first-cycle cache priming and a 256-case proptest run. The M0 "cheap-reject"
path (compile_error → skip run) cost only 1.75 s in the K=2@96 run.

Memory peak (per run record, process exit values): torch MPS **allocated
stable at 14.5 GB** (model + activations) across every run; **driver**
(purgable virtual) 88–160 GB accumulating over a process's sequential
canvases; python RSS peak 0.8–1.3 GB; **no swap, no OOM** with the
watermark+per-step-empty_cache fix. The spike's 83–88% system-memory peaks
were not exceeded; driver growth over a long K-sequence is the quantity to
watch (M2: sample `driver_allocated_memory` growth per canvas, consider
periodic process recycling).

## Draft-API proof (Group C)

- Key resolution: env `SYNTHETIC_API_KEY` → `~/.pi/agent/models.json`
  `providers.synthetic.apiKey` (both paths implemented; models.json used
  here).
- Probe call (`Reply with exactly: OK`): **1.5–1.7 s roundtrip**, auth OK,
  `hf:zai-org/GLM-5.3` responded (accepts GLM's `thinking` flag too).
- Full drafting call (wrap-sum instructions): non-streaming TIMED OUT at
  120 s and again at 420 s (reasoning model: silent socket for >10 min);
  **fixed with stdlib SSE streaming** — 4655 chars streamed in ~14 min.
- Quality: drafted `properties.draft.rs` compiles, **passes the correct
  example candidate and fails the buggy candidate**
  (`wrapping_sum panicked on [9223372036854775808, 1, 18446744073709551615]`)
  via a boundary-weighted `prop_oneof!` strategy + `catch_unwind`. The
  draft was NOT installed: bench properties stay parent-authored;
  `--approve` requires interactive y/N and stamps `task.toml`.

## Spike-prediction check (design §0 vs measured)

| spike prediction | measured in M1 | verdict |
|---|---|---|
| 0.53–0.59 s/step, linear in steps (slope 0.566) | entropy fresh-process 0.92–0.94 s/step, linear; topk_margin 1.97 s/step | PARTIAL: +0.36 s/step is our instrumentation (per-step empty_cache + fp32 chunked confidence + sync); the rest matches |
| K=1 sequential mandatory; batched K>1 corrupted | K=1 used everywhere; no NaNs, no padded rows | honored |
| use_cache=False + watermark env + per-step empty_cache mandatory | baked into runtime.py/dream.py; no OOM, no swap, no degenerate text in 14 canvases | confirmed necessary + sufficient |
| steps=128 ≈ 75 s/candidate | 118 s (entropy, fresh) | quality trend confirmed; wall higher due to instrumentation + canvas 384 vs 256 |
| hooks natively expose tokens/logits + output_history | used for ALL instrumentation; no fork, no monkeypatch, HF cache untouched | confirmed |
| 48 GB sufficient | peak allocated 14.5 GB; driver purgable ≤160 GB; no swap | confirmed |

## Deviations from spec / caveats

1. **`temper gen` needs ~2× the spike's s/step at equal settings.** Two
   measured components: (a) instrumentation ≈ +0.36 s/step (confidence stat
   forces one GPU sync per step; the spike had none); (b) **topk_margin costs
   2.1× per step vs entropy** (1.97 vs 0.92 s/step, consistent across all
   processes) — the spike never measured alg-conditional cost; (c) later
   particles in a long sequential run run ~1.9–2.1 s/step even on entropy
   (K=4 run: p2 entropy 1.925) — sustained GPU-load effect (thermal or driver
   state), not investigated further.
2. **Python 3.9 has no tomllib** → venv also pins `tomli` (pure-python,
   stdlib-adjacent); all other temper code is stdlib.
3. **eval pass@k estimator** is the unbiased combinatorial one averaged over
   run records; with only 1–3 records per (task, K) the CIs are wide. The
   K=2 table mixes sweep settings (documented above).
4. **ringbuf is hard for Dream-Coder-7B**: steps=96 K=4 → 0/4 (3×
   impl-methods-instead-of-free-fns rejected by `must_define`, 1 compile
   error); steps=128 K=2 rerun → 0/2 (E0425 undefined identifier; 1×
   must_define rejection). Multi-item APIs (struct + 3 free fns with
   precise semantics) are measurably harder than single-function tasks
   (sort/wrap-sum compile and often pass). The `must_define` extract guard
   is doing real work rejecting wrong-API-shape answers. Raw model text is
   not stored in run records (M2: add it — needed to debug such
   rejections).
5. **Candidate rows keep diagnostics as codes+primary spans only** (full
   rendered diagnostics would bloat records); `dead_code` warnings appear in
   diag_codes but only errors gate compilation.
6. The spike's suggested "alternating remasking algorithms" is implemented
   but topk_margin's 2.1× cost makes entropy-only the pragmatic default for
   M2 experiments (config knob exists: `--algs`).

## M2 open issues

1. **Instrumentation overhead**: sample confidence every S-th step (S=8)
   instead of every step → recovers ~0.3 s/step; keep empty_cache per step
   (mandatory).
2. **topk_margin cost**: profile `sample_tokens` margin path on MPS; either
   drop it from the diversity cycle or implement a cheaper low-confidence
   remasking in a custom loop (`_sample` is ~90 lines, spike §6).
3. **Sustained sequential slowdown** (0.92 → 1.93 s/step within one process):
   thermals vs Metal driver state; consider canvas-batched process pool or
   `taskset`-style affinity; measure GPU clocks.
4. **Driver-memory growth over long K-sequences** (88 → 160 GB purgable
   virtual in one K=2@128 process): fine today, but monitor; worst case
   recycle the process every N candidates.
5. **Store raw model output per candidate** in the run record (extract
   rejections are currently opaque; ringbuf impl-methods case is the example).
6. **Prompt calibration for multi-item APIs**: state "free functions, not
   methods/inherent impl" explicitly in the template; measure ringbuf K=4
   again before L1.
7. **L1 needs true per-token entropy of committed tokens** (design §3
   pseudo-likelihood): currently we record top-1 confidence only; extend the
   logits hook to accumulate fp32 entropy stats per committed position.
8. **Drafting latency**: GLM-5.3 thinking phase >10 min per property; add a
   streaming "reasoning off" flag test (`thinking: {"type":"disabled"}` was
   accepted by the probe) or a smaller drafting model for iteration, keeping
   GLM-5.3 for the final draft.
9. **corpus/ growth policy**: dedupe seeds across runs per strategy_hash;
   M3 will want per-strategy metadata (property, candidate lineage).

## Repro

```sh
./bin/temper gen --task bench/sort -K 4 --steps 96 --canvas 384   # acceptance
./bin/temper eval --task bench/sort -K 4 --repeat 3               # pass@k
./bin/temper draft --task bench/wrap-sum                          # drafting demo
temper/.venv/bin/python -m unittest tests.test_extract            # unit tests
```

Run records: `bench/*/runs/*.jsonl`; failing-seed corpus:
`bench/*/corpus/<strategy_hash>/*.json`.
