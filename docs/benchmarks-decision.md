# temper — Benchmarks & Evaluation Decision (DRAFT, pending final cross-check)

Status: CONSOLIDATED — 6 research lanes (2026-10-18), all complete.

---

## 1. Decision

**Primary benchmark: MultiPL-E Rust track** (`humaneval-rs` = 156 tasks, `mbpp-rs` = 354 tasks —
measured from pinned HF dataset `nuprl/MultiPL-E`, revision `3a9e8d22…`; stale citations say "161").
**Secondary: Multi-LCB Rust track** (contamination-resistant, LiveCodeBench v4–v6 problems, 12 languages,
ICLR 2026 paper, 24-model leaderboard).
**Bench-internal: hand-written proptest tasks** (existing `bench/` suite) for the properties-specific claims
that no external benchmark measures.

### Why MultiPL-E wins
- Scoring container is **architecturally decoupled from inference**: write completion files
  (`{name, prompt, tests, completions}`), container executes them sandboxed (`--network none`,
  timeouts, per-language toolchain) and computes unbiased pass@k. Steering code never touches scoring
  machinery → clean steered/unsteered ablations, reproducible numbers.
- Grader = synthesized Rust unit tests compiled against the completion. Our verifier (proptest) stays
  **held-in**; we never steer on scoring tests (UnMaskFork's public/private mismatch lesson).
- Actively maintained (last commit Jan 2026), official container, per-language golden reference solutions
  for smoke tests.
- Caveat, worn openly: 2022 translations, contamination exposure (see §4). Mitigated by Multi-LCB secondary.

### Container mechanics (live-verified by lane, 2026-10-18)
- Scorer is fully decoupled: `evaluation/src/main.py` builds `program = prompt + completions[i] + '\n' + tests`;
  input is `<name>.json(.gz)` files `{prompt, tests, completions, language:"rs"}`, run via `--dir/--output-dir`.
- **GOTCHA 1**: `stop_tokens` are NOT enforced by the scorer — retaining a stop sequence duplicates the
  closing brace → spurious SyntaxError. Adapter must strip stop markers from completions.
- **GOTCHA 2**: `rustc` invoked with NO `--edition` → **Rust 2015**. Dream-Coder will emit 2018+ idioms
  (`dyn Trait`, `async`) that fail to parse. Decision needed (M2 spec): score official-default (2015) for
  conservatism AND prompt-normalized; or patch container to `--edition=2021` for internal rows. Report both.
- **GOTCHA 3**: official image is **amd64-only, build-dated 2024-07-12, rustc 1.75.0**; on M3 Max it runs
  under Rosetta. Consider building an aarch64 image pinned to our M0 toolchain; keeps host/container parity.
- macOS Docker bind mounts came up empty (file-sharing artifact, not a harness bug) — use `docker cp`.
- pass@k via repo's `pass_k.py` (numpy only).
- Repo is maintenance-mode (last main commit 2026-01-28; README redirects to successor **Ag-LiveCodeBench-X**,
  arXiv 2508.04865 — check this successor at M2 kickoff via a free-tier lane).
- Alt harnesses if needed: BigCode Evaluation Harness (vendors `multiple-rs`); OpenCompass `rs` avoided
  (grades via third-party HF Space).
- No dLLM MultiPL-E numbers of any kind except LLaDA2.x/MoE **aggregate-only** (flash 74.87, mini 67.46,
  LLaDA-MoE-instruct 52.53) — not Rust-attributable. Stable-DiffCoder's per-language table omits Rust.

## 2. Baseline matrix (all through the same container, matched NFE)

| # | System | Purpose |
|---|--------|---------|
| B1 | temper L0 (Dream-Coder-7B, best-of-K + proptest rerank) | Our L0; sanity-check against published BoN rows |
| B2 | Dream-Coder-7B unsteered (temperature samples, same counting) | The "model alone" table — **first published dLLM Rust numbers** |
| B3 | Qwen2.5-Coder-7B (AR twin of Dream-Coder's init) | Self-normalizing control: isolates diffusion-search effect from model quality |
| B4 | UnMaskFork-style MCTS reimplementation (its code never released — VERIFIED negative) | Closest direct comparator; public-tests-as-reward vs our held-in PBT |
| B5 (optional) | LLaDA-8B-Instruct unsteered | Second dLLM reference row for the table |

Published numbers B1/B4-class rows must be compared against (NFE = 12,288 budget, pass@1):
LiveCodeBench BoN(DCoder) 18.0; UMF 28.0 (2-model), 32.0 (3-model), climbs to 30.0 @ NFE 24,576;
HumanEval+ BoN 71.0 / UMF 88.0; MBPP+ BoN 66.0 / UMF 72.0. (arXiv 2602.04344, verified.)
Prism (single LLaDA-8B, self-verified): HumanEval 79.27 @ NFE 2,480, MBPP 38.20 @ NFE 2,576 (arXiv 2602.01842).
Saber (decoding floor): +1.9% pass@1 at 251% speedup (arXiv 2510.18165).

## 3. Measurement protocol (non-negotiable)

1. **X-axis = NFE** (forward passes), not sample count — partial trajectories/caching make per-candidate
   cost variable. All run records log NFE per candidate. Report ≥4 budgets (UMF: 768→24,576).
2. **Also log wall-clock including verifier execution** — in code evals test execution dominates
   (Orlanski et al., arXiv 2506.10056). Our 236 ms sandboxed cycle is an asset to surface.
3. **Coverage ≠ selection.** Monkeys paper (arXiv 2407.21787): MATH oracle coverage 98.4% @ k=10k but
   majority-vote/reward-model selection plateaus at ~41%. Our claims must be about the **selection axis**
   (verifiable pick beats MV/RM plateau), not raw pass@k coverage.
4. **Statistical hygiene**: Bayesian CIs ("Don't Pass@k"), ≥20–100 samples/problem, fixed MultiPL-E
   protocol points (pass@1 @ T=0.2; pass@10/100 @ T=0.8). Honest counting: extraction failures count
   against pass@k.
5. **Answer-distribution-skew control** (Yona et al., NAACL 2025) as sanity check that coverage gains
   aren't benchmark-artifact.
6. **Lifetime NFE accounting incl. drafting model calls** when comparing to self-verified systems
   (Prism counts SVF calls separately — we must too).

## 4. Contamination posture

- Qwen2.5-Coder documented 10-gram decontamination (arXiv 2409.12186, p.9), but Riddell et al. 2024 found
  3.6–20.8% of HumanEval/MBPP solutions in Pile/Stack plus semantic leakage → **MultiPL-E rows get an
  explicit contamination caveat; LiveCodeBench-Rust / Multi-LCB rows are contamination-controlled**
  (release-date filtered). Both reported.
- **"RustEval" does not exist** (phantom, traced to a misattributed survey row pointing at a 2016 IR tool).
- **"VeriContest" (946 Rust problems, flash-lane claim) could not be found by any other lane — treated as
  phantom, dropped.**

## 5. Contribution claim, post-literature-search

Nobody has published Rust numbers for any dLLM (Dream, Dream-Coder, DiffuCoder, LLaDA all Python-only —
verified). The open niche, checked by two independent lanes:
**Rust × held-in property-based adversarial verification × seed-corpus steering**, at matched NFE,
judged on the selection axis against best-of-N coverage curves (Monkeys scaling-law shape as the null
model: coverage ≈ exp(a·k^b); per-family S-curves with common slope — a steering method should show a
horizontal offset, not just move along the curve).

Implication for M2 sequencing: **evaluation adapter (temper → MultiPL-E completion files + container
scoring + NFE accounting) ships before any L1 steering**, because (a) every later claim hangs on it,
(b) running B2/B3 immediately gives a sanity check of the pipeline vs published BoN rows, (c) it's cheap.

## Appendix: primary sources (verified lanes)

- MultiPL-E: github.com/nuprl/MultiPL-E; HF dataset nuprl/MultiPL-E @ 3a9e8d22…; Cassano et al. 2023.
- Multi-LCB: ICLR 2026, 12-language LiveCodeBench-class; Rust penalty (Rust pass@1 ~0.33–0.39 vs Python ~0.48).
- UnMaskFork: arXiv 2602.04344 (ICML 2026), Sakana AI; NFE-budgeted MCTS over unmasking actions; no code release.
- Prism: arXiv 2602.01842 (ICML 2026); code github.com/viiika/Prism.
- Saber: arXiv 2510.18165 (ACL 2026); github.com/zhaoyMa/Saber.
- Large Language Monkeys: arXiv 2407.21787; Apache-2.0 repo (4 datasets: GSM8K/MATH/CodeContests/MiniF2F);
  SWE-bench/Moatless harness NOT released.
- Snell et al.: arXiv 2408.03314 (FLOPs-matched protocol).
- Wu et al.: arXiv 2408.00724 (Pareto accuracy-FLOPs; voting convergence bound).
- Orlanski et al.: arXiv 2506.10056 (verifier throughput frontier).
- Yona et al.: NAACL 2025 (skew control). "Don't Pass@k": Bayesian CIs.
- Riddell et al. 2024: contamination quantification.
- Dream-Coder: arXiv 2509.01142 (HKU/Huawei); Qwen2.5-Coder-7B init; 322B adapt tokens.
