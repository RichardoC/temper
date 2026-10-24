# temper — final report: the toot thesis, measured

This document is the program-level record for the whole investigation
(M0 → M3.1). Origin: a toot (feed.hella.cheap/@bob, 2026-10) proposing that an
open-weights *masked diffusion* language model be used to generate code,
**jointly searching code space and input space at inference time**, MCMC/SMC
style, with the Rust toolchain (cargo + proptest) as the inference-time
discriminator. Repo: this one. Every claim below is measured in a committed
cell; every number is cross-referenced to the milestone report that owns it.

---

## §0 The question, and the answer

**Question.** At fixed inference budget (NFE — model forward passes), does
verifier-guided search over a diffusion LM's outputs beat plain diverse
sampling plus a final filter? And can the input-space arm of the search —
proptest harnesses written *at inference time* — exist at 7B open-weights
scale?

**Answer, composite and honest.**

1. **Code-space generation scales with budget, threshold-shaped, not linear.**
   Honest pass@1 rose 0.16 → 0.20 → 0.44 → 0.64 across 384 → 512 → 1024 → 2048
   NFE/problem (M2.3/M2.4b, corrected instrumentation) — still rising at the
   top of our budget; "capability floor" and "budget-linear" were both
   rejected by the data.
2. **In-loop verifier-conditioned reallocation does not detectably beat
   diverse-sampling-plus-filter at matched NFE.** The engine's best cell was
   17/25 — the program's best — but the +3 over the null decomposed into
   *better draws and better picks*, and the one flip the causal story needed
   (HumanEval_25) did not replicate at fresh seeds (M3.0-P3). Mid-decode
   evidence-action loops are dead at this scale: informed recall of
   "obviously-dead" canvases before full commit measured ≈ 0 (D4).
3. **Knowledge injection is not the missing piece.** A cell that handed the
   model the exact idiom it was failing to use (C-hint) followed the hint
   *verbatim* and still failed 4/4 (M3.0-P0.5): the hard problems are
   mode-locked, not knowledge-gapped.
4. **Input-space probes cannot be authored at inference time by any model in
   the stable.** Compiling-proptest first-draft rate: 0 % (Dream-Coder-7B),
   16 % (Qwen2.5-Coder-7B, the strong AR twin) — across four distinct
   failure mechanisms, including an instruction-content effect that
   collapses Qwen's MPS decode logits to an exact zero vector (M3.1).
5. **The selection mechanism itself is oracle-grade where it exists.** With
   50 frozen hand-authored harnesses, property verdicts agreed with
   container ground truth on **64/64** judged canvases — zero mis-ranks —
   recovering a clean attributed flip (+1) with zero regressions, and a
   post-hoc tiebreak fix (never deferring to confidence inside the
   max-pass tier, export-aware) reaches **exactly the oracle 16/25** on the
   same banked ensemble (M3.1 §6).

The binding constraint, measured three ways: **supply, not intelligence.**
The lazy summary of the whole program: *the detector works; the discriminator
can't be built at this scale.*

---

## §1 The program as it ran

| milestone | what it did | verdict | report |
|---|---|---|---|
| M0 | sandbox + harness (seatbelt, cargo-in-container scoring) | ALL STEPS PASS | m0-report.md |
| M1 | best-of-K + proptest rerank on a 1-problem task | works end-to-end; shrinking produces minimal counterexamples | m1-report.md |
| M1.5 | benchmark selection due-diligence | humaneval-rs vendored, container-pinned | m1.5-report.md |
| M2.0/M2.1 | honest scoring discipline (dual-edition, extraction failures = failures) | honest pass@1 instrumented | m2.0-report.md |
| M2.2 | corruption-vs-budget forensics | honest floor 0.04, artifact-classified | m2.2-report.md |
| M2.3 | steps ablation 96/128/256 (+ MPS compile-cache leak fix, token-neutral proven) | 128→256 jumps 4×; threshold-shaped | m2.3-report.md |
| M2.4a/b | s512 cell; AR twin control | corrected staircase 0.16→0.64; AR twin 0.92; **stop-token strip-amputation bug caught** | m2.4a/b-report.md |
| M3.0 P0–P2 | instrument build (D1–D7), C-hint, reachability probe, P2 bit-identity gate | C-hint 0/4 (mode-lock); T-cliff at 0.8; P2 caught `no_grad` kernel bug + checkpoint re-seeding | m3.0-report.md |
| M3.0 D4 | ghost-closure validity on real canvases | informed recall fires only at full commit → C-kill demoted per frozen rule | m3.0-report.md |
| M3.0 P3 | the C-cell matrix at NFE 2048 | C-null 14, C-screen 14, **C-full 17 (program best)**, SHIP PASS / ATTRIBUTION FAIL | m3.0-p3-report.md |
| M3.1 | property-reward selection | composite negative: generator wall + 64/64 discrimination + coverage ceiling | m3.1-report.md |

The P2 equivalence gate deserves its own line: it asked whether the steering
engine, stripped of its actions, reproduces the plain sampler *bit-for-bit*.
It passed 4/4 — after catching two engine bugs (a missing `torch.no_grad()`
that silently selects different MPS kernels, and RNG re-seeding at checkpoints
that made particles pairwise-identical) that would have poisoned every steered
measurement downstream. Every "the engine changed nothing" claim in this repo
rests on that gate being green.

---

## §2 The five findings, with the numbers that make them

**(F1) Budget response is a staircase with an unclimbed top.**
0.16 → 0.20 → 0.44 → 0.64 across 384/512/1024/2048 NFE per problem
(honest pass@1, official container verdicts, extraction failures counted as
failures). The 512-step cell was still improving when the budget ran out —
there is no evidence any of these curves plateau within reach.

**(F2) Instrument bugs are the program's most frequent discovery.** Seven
load-bearing bugs were caught before they could fake a result: the GOTCHA-1
stop-token stripper (amputated 64–76 rows per cell into guaranteed parse
errors — found by building the AR control row), the MPS compile-cache leak
(335 GB driver cap — fixed by the token-neutral fixed-shape sampler), the
`no_grad` divergence (P2), checkpoint RNG re-seeding (P2), the tokenizer's
397-id unmappable band that crashes decode (`skip_special_tokens` cannot
drop them), the EOS-residue ~45 % false-kill rate in the checkpoint renderer
(outcome-neutral under the demoted config, fatal to any kill-rung re-enable),
and the machine-generated-instruction NaN collapse. The lesson generalizes:
in systems where the "detector" is also software you wrote, bit-identity
gates and control rows are not optional.

**(F3) Steering attribution failed its own gate — that is the finding.**
C-full 17/25 vs both nulls at 14/25, zero regressions, sign-test p=0.125 on
the primary (n=25 needs ≥5 one-directional discordants for α=0.05 — stated on
every gate line). The four canvas divergences from the offline counterfactual
were all *different draws*, not different rankings of the same evidence. The
single outside-cohort flip (HE_25) failed 2-game replication; C-screen later
passed 25 on C-null's own ensemble, reclassifying the flip as partially an
emission artifact. Verdict per pre-registered text: **SHIP PASSES
(capability), ATTRIBUTION FAILS (mechanism)**.

**(F4) Where the model is blind, it is blind by temperature.** The
reachability probe (108 canvases, 3 temps × K=4) measured a *cliff*, not a
slope: T=0.4 → 0/36 passes; the live band starts at T≥0.8. Four problems
({0, 18, 19, 21} at first) looked unreachable at every probed temperature —
and an independent instrument agreed a month later: the union of all 22
banked diffusion cells contains no container-OK canvas for them at any
budget. Two instruments, same wall.

**(F5) The selection leg: perfect, uncovered.** Oracle 16/25 vs 14/25 picked —
the ensembles *saw* two more solutions than every emission rule recovered
(M3.0). Property evidence fixed exactly what it could reach: 64/64 agreement
with the container, one clean attributed flip, zero regressions, and the
post-hoc v4 tiebreak (export-aware, confidence-free inside the max-pass tier)
recovers the oracle exactly. What it could not fix is supply: 8/25 problems
had no surviving harness, four of which have no correct canvas anywhere, and
the harness *author* — at 0 %/16 % first-draft compile rate — is the measured
wall.

---

## §3 What was NOT established — in full

- No claim that the engine's reallocation mechanism *works*; what was
  established is that **at matched NFE it is indistinguishable from
  diverse-sampling-plus-filter** at the demoted (kill/clone-off,
  late-window-advisory) scope we were allowed by the evidence discipline.
- Nothing beyond one benchmark (humaneval-rs, 25 problems), one model family
  pair (Dream-Coder 7B diffusion; Qwen2.5-Coder 7B AR), one machine class
  (M3 Max, MPS). n=25 with sign tests is honest but weak; the exchangeable
  pooled statistics (Jeffreys) are printed everywhere as reporting-only.
- M3.1's 64/64 discrimination is measured on one banked ensemble and 50
  hand-authored harnesses. It is a *demonstration that the signal is
  oracle-grade when covered*, not a coverage claim; the claim shape changed
  from generated-at-inference-time (measured dead) to hand-frozen (measured
  perfect) and both halves are reported as themselves.
- The never-solved set {0, 18, 19, 21} bounds every cell in this repo at
  21/25 regardless of selection rule.

## §4 The methodology record — why the numbers are believable

Pre-registration with dated, labeled amendments (never silent): the design
docs freeze cells, gates, ranking rules and replications before generation;
three supervisor-invented thresholds (the 90 % compile bar, the 23/25
coverage gate, G1's near-full-coverage assumption) were corrected against
measured structure during M3.1 and are owned in the report's methods section.

Nulls at every fork: C-null (answers the "is it just the ladder?"), C-hint
(delivery vs search), C-screen (in-loop vs offline rule). Paired comparisons
on shared seeds; per-problem sign tests primary; pooled Jeffreys demoted to
reporting-only because problems are not exchangeable. Honest folding of
extraction failures as failures. Dual cost accounting (NFE + wall-clock) on
every cell. Replication clauses for single-problem claims (the 2-game
failure that spiked HE_25 created the 4-game standard). Frozen verdict
configs with tamper-checked shas in every run record (D7). The kill
protocols were pre-registered (D4 demotion executed per written rule), and
D2 replay logs make every steered game re-derivable from disk.

## §5 Cost ledger

Per-cell walls are measured (each report carries its own `gen_wall`); the M2 and
probe rows are sums over their listed cells.

| phase | GPU wall | notes |
|---|---|---|
| M2 cells (96/128/256/512 + AR) | ~26 h | includes one leak-diagnosis relaunch |
| M3.0 probe + P2 + pre-PIlot work | ~11 h | 108-canvas probe, equivalence runs |
| C-null@2048 | 14.54 h | 200 canvases |
| C-full@2048 | 14.17 h | 19 games + 6 resumed post-crash |
| flip replication | 2.84 h | 5 games |
| re-decision probes, pre-flights, smokes | ~0.5 h | |
| M3.1 total | **≈2.1 h** | 0.13 h/16 % pilot/0.55 h probes; engine cell (15 h) + P-25re (4.7 h) deliberately not run |
| **total** | **≈70 h GPU** | on one 48 GB laptop |

Program spend on model API traffic (parent/worker sessions): ~$22 at the
mid-program checkpoint; the M3.0/M3.1 build and report phases were
client-model work on top. NFE is the primary budget unit everywhere:
every cell reports `nfe_total` and asserts it against its envelope.

## §6 The successor spec — what any future attempt needs, per these measurements

1. A compiling-harness generator with a **measured** first-draft compile
   bar (ours: 0 %/16 % — the wall). Without it, property-evidence selection is
   capped by coverage the same way ours was: oracle-grade discrimination went
   15/25 because it could only speak on 17.
2. Coverage-first cell design: instrument supply per problem is the
   constraint, not ranking quality within the covered set.
3. The v4 tiebreak shape: max-pass tier must never defer to
   commit-confidence, and must prefer extraction-clean candidates.
4. Skip mid-decode evidence-action loops at this model scale (D4 + P3);
   spend the budget on draws and commit granularity instead — the ladder
   trade (8×256 vs 4×512) is real and *problem-dependent*: the
   budget-fixable cohort wanted granularity, not diversity.
5. Budget: no plateau was reached — 2048 NFE/problem was still climbing.

## §7 Disposition

- M3.2 (rejuvenation) is **parked, unstarted** — nothing pending.
- Repo state: HEAD `49b8eda`; 384 tests green; both config freezes
  (v2 `a63b2b36`, v3 + harness manifest) tamper-verified; all cell artifacts,
  guard logs and replay records committed; `temper steer-replay` can
  re-derive every steered game offline.
- Reproduction entry points are in each milestone report; the two
  capstone cells: `docs/m3.0-p3-report.md` (the C-matrix) and
  `docs/m3.1-report.md` (the property leg).

## §8 The toot, answered

> The idea was: diffusion LLM writes Rust, and *while it writes*, property-based
> tests search input space, cargo runs them, and the results steer the decode —
> MCMC over code and inputs jointly, the toolchain as the verifier.
>
> We built it and measured it. Short version: the search works, the verifier
> works, the joint search can't be built at 7B scale — for now.
>
> – Code generation gets better with budget in a staircase, not a line:
>   0.16 → 0.64 honest pass@1 from 384 → 2048 sampling steps, no plateau.
> – Steering the decode with toolchain feedback bought a program-best 17/25,
>   but the pre-registered attribution gate couldn't certify the mechanism:
>   +3 over the null, of which the one decisive flip failed to replicate.
>   Verifier-in-the-loop ≈ diverse sampling + a final screen, at matched
>   compute. That's the null result, and we're keeping it honest.
> – The property arm died where nobody predicted: *writing the harness*.
>   0 % of diffusion drafts and 16 % of AR drafts compiled against the
>   target's signature. Meanwhile 50 hand-written proptest harnesses judged
>   candidate code **64/64 correct vs the benchmark** — zero mis-ranks —
>   and hit the achievable ceiling (16/25 = oracle) the moment the
>   tiebreak stopped deferring to confidence.
>
> So: the discriminator is perfect and can't be manufactured at this scale;
> the steering mechanism is measurable and (at this scale) null. The open
> problem is supply. All cells, nulls, gates and forensics are public in the
> repo — including the seven instrument bugs we had to catch before any of
> these numbers meant anything.
