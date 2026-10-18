# M1 design — L0: Best-of-K sampling + verifier reranking

Status: DRAFT (parent-authored, 2026-10-18). Gate: M1.0 spike go/no-go.
Supersedes nothing; M0 (`m0-report.md`) is complete and verified.

## 1. Objective

Close the generation→verification loop end-to-end with the simplest possible
steering strategy, so later tiers (SMC, adversarial seeds) have an honest
baseline:

1. Human provides a Rust task: signature + docstring (the "spec").
2. GLM-5.3 (Synthetic API, human-approved) drafts proptest properties.
3. Dream-Coder-7B (local, PyTorch+MPS) samples K candidate implementations.
4. Each candidate is emitted into the M0 harness, sandboxed, and scored.
5. Report best candidate by score; record all runs for analysis.

This is inference-time search tier **L0 = best-of-K + verifier reranking**.
No mid-denoise intervention yet (that is M2/L1).

## 2. Architecture

```
            ┌──────────────────────┐
 spec.rs.md │  temper authoring    │  (human + GLM-5.3 assist)
  signature └─────────┬────────────┘
                      ▼
            bench/<task>/task.toml  — signature, doc, prompt template
            bench/<task>/properties.rs — APPROVED proptest property source
                      │
                      ▼
        ┌─────────────────────────────┐
        │ temper.model.dream (LOCAL)  │  Dream-Coder-7B, PyTorch MPS bf16
        │  - load once, keep resident │
        │  - sample K canvases:       │  per-step temperature jitter across
        │    diversity via temp/alg   │  particles (see §4)
        └──────────────┬──────────────┘
                       │ K × candidate.rs
                       ▼
        ┌─────────────────────────────┐
        │ temper.harness (M0, done)   │  seatbelt sandbox, 236 ms/candidate
        │  emit→check→build→run→JSON  │  serial compile queue
        └──────────────┬──────────────┘
                       ▼
        temper.steer.l0 — scoring + selection + run record (JSONL)
```

New modules (add to repo; do NOT modify m0 modules' public APIs):

- `temper/model/__init__.py`
- `temper/model/runtime.py` — device/dtype resolution (`mps` bf16 → CPU fallback
  with loud warning + slowdown notice), resident model holder (load once per
  CLI invocation), memory-watermark probe (`torch.mps` counters).
- `temper/model/dream.py` — loader (`trust_remote_code`) + thin wrapper over
  the model's own `diffusion_generate` **or** the forked loop if the spike
  found built-ins insufficient → decision recorded from spike report §6.
  MUST expose: `sample(spec_prompt, n_particles, canvas_len, steps, temps)`
  → list of raw decoded strings.
- `temper/model/extract.py` — extract Rust code from model output
  (fenced-block and bare-source heuristics; reject prompt echo).
- `temper/authoring/__init__.py`, `temper/authoring/draft.py` — GLM-5.3
  property drafting via Synthetic API (OpenAI-compatible chat endpoint;
  key from env `SYNTHETIC_API_KEY`; model id `hf:zai-org/GLM-5.3`).
  Output = draft `properties.rs`; REQUIRES `--approve` interactive step or
  `task.toml` flag `properties_approved = true`; never auto-approves.
- `temper/steer/__init__.py`, `temper/steer/l0.py` — K-particle loop,
  scoring (§5), selection, run record.
- `temper/cli.py` — entry points (§6).

## 3. Decisions carried in from research (do NOT relitigate)

- Inference: PyTorch + MPS, bf16. MLX and llama.cpp ruled out (no per-step
  logits; unverifiable diffusion support). Fallback model if Dream-Coder
  misbehaves: `GSAI-ML/iLLaDA-8B-Instruct` + ML-GSAI generate.py port.
- CPU fallback exists but is a debugging path, not a real mode.
- Candidate crates: zero third-party deps (M0 guarantee). The model's Rust
  only ever touches std + the vendored proptest in test code.
- Model likelihood score `log p_model(x)` needed later for L1; L0 uses the
  cheap proxy: negative mean per-token predictive entropy over committed
  tokens (computable from `output_history` if available; else omit and note).
- Canvas 256–512 tokens; K bounded by spike measurements (start K ∈ {4,8}).

## 4. Diversity across particles (L0's only steering knob)

Diffusion sampler diversity is not just temperature — remask ORDER changes:
- per-particle `temperature` sampled from {0.2, 0.4, 0.6, 0.8},
- alternating remasking algorithms if exposed (`entropy` vs low-confidence),
- distinct system-prompt suffixes NO (confounds scoring) — keep prompt fixed.
Record per-particle config in the run record for later analysis.

## 5. Scoring (what "best" means at L0)

score(x) = 100·1[compiles] + 20·1[clippy-clean or warning-free build](v1: skip clippy)
         + 1000·1[tests pass] − 5·(reject fraction)
         + tie-break: higher model pseudo-likelihood if available, else first.
Verbose per-candidate record: status chain (compile_error→...→pass/fail/abort),
diagnostics (codes+spans), minimal failing input + persisted seed on fail,
cycle wall times. All rows to `bench/<task>/runs/<ts>.jsonl`.

Seed corpus dir `bench/<task>/corpus/<strategy_hash>/` collects persisted
failing seeds — this directory is the M3 adversarial generator's asset.

## 6. CLI

```
temper draft --task bench/sort          # GLM-5.3 drafts properties.rs (needs approval)
temper gen   --task bench/sort -K 8 --steps 32 --canvas 384
temper eval  --task bench/sort -K 8 --repeat 3   # pass@k table from run records
```

## 7. M1 acceptance criteria

1. `temper gen` runs end-to-end on ONE bench task on this laptop, producing a
   run-record JSONL with K scored candidates and a selected best.
2. At least one bench task where the L0 winner passes tests that ≥1 other
   particle failed (verifier did real work — else pick another task).
3. Bench suite shipped: 3 starter tasks with human-approved properties
   (sort, ring-buffer push/pop order preservation, int-add-with-overflow-wrap
   or small parser); documented procedure for adding more.
4. `docs/m1-report.md`: pass@k table, cycle timing model-side vs harness-side
   shares, memory peak, spike-prediction check, deviations, M2 open issues.

## 8. Explicitly out of scope for M1

- Mid-denoise intervention, SMC resampling, ReMDM re-masking (M2).
- Adversarial seed-side sampling / corpus weighting (M3).
- Fine-tuning or RL. Dream-Coder weights are consumed as-is.
- Any network access from generated crates (harness enforces; keep).
