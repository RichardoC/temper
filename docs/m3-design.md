# M3 design — L1: the steering engine (SMC substrate, verifier ladder, budget-matched gates)

Status: DRAFT for parent review (design-only milestone; no code, no generation).
Owner of the why: `docs/benchmarks-decision.md` (§5 contribution claim, §3 measurement
protocol). Evidence base: the M2.4b-corrected instrument
(`docs/m2.4b-report.md` — READ FIRST; the committed M2.3/M2.4a numbers and the
M2.4a L1 target list are SUPERSEDED by it), plus fresh CPU-only forensics run
for this doc on the committed corrected-basis exports (§1.2, §3.2 — no GPU,
committed artifacts only).

This doc supersedes: the L1 sketch in `docs/m2-design.md`'s "M2 sequencing"
note, in M2.3's "§M2.4 notes item 2" (written against the amputation-artifact
instrument), and in M2.4a's "Implications for L1 steering" target list (M2.4b
§M2.5 formally reopened all of it).

---

## rev1 changelog — adversarial-review modifications (2026-10-21)

Two independent adversarial reviews of the rev0 draft were commissioned by the
operator (both MODIFY-THEN-PROCEED) and are part of the project record:

- **review A (Kimi-K3)**:
  `/Users/richardf/.pi/agent/sessions/--Users-richardf--/subagent-artifacts/f292ab77-5102-4630-a316-07ac2e629f5d_worker_output.md`
- **review B (GLM-5.3 — a second instance, not the author)**:
  `/Users/richardf/.pi/agent/sessions/--Users-richardf--/subagent-artifacts/29c494a4-e812-488a-b7cc-772c229d0b01_worker_output.md`

Both verdicts converged on the same two missing nulls (one each, complementary)
and the same statistics/gates critique. rev1 adopts all of the following; the
body sections below carry the changed text (old text where conflicting is
SUPERSEDED by the rev1 passages):

| # | rev1 change | where | review |
|---|---|---|---|
| R1 | Reframe: deliverable = engine + mechanism question, not a capability number; AR-substrate paragraph | §0.5–0.6 | A1, B1 |
| R2 | C-hint cell (knowledge-injection null), scheduled FIRST | §4.1, §5 | A2 |
| R3 | C-screen counterfactual (static-filter null), primary attribution delta | §4.1, §4.2, §4.3, §5 | B2 |
| R4 | Paired sign-test primary; Jeffreys demoted; pre-registered primary comparison; flip replication | §4.2 | A3, B4 |
| R5 | Deepened, cohort-stratified probe; PARTIAL forces EV rewrite; NO-GO s512 recheck | §5 (P1) | A4, B4 |
| R6 | Freeze-before-cells (verdict config sha); D4 demotion rule; C-null emission spec; §4.3 isolation fixed | §4.1, §4.3, §7 (D7) | B2, B3 |
| R7 | C-null EV modeled; reviewers' falsifiable predictions on record | §6.1, §11 | A, B |

Nothing in rev1 removes rev0's strengths (P1 falsifier economics, §3.2
ghost-closure negative result, D-deliverables, commit-stickiness).

---

---

## 0. What changed since the last L1 sketch (and why the design moved)

The M2.4b correction rewrites the problem L1 must solve:

| fact | committed basis | corrected basis (M2.4b) |
|---|---|---|
| Dream s256 (B2) honest pass@1 | 0.16 @ NFE/prob 1024, 7.21 h | **0.44** (11/25 problems OK 4/4) |
| Dream s512 honest pass@1 | 0.16 @ NFE/prob 2048, 14.48 h | **0.64** (16/25) — budget response still rising |
| Residual debug mass @ s512 | "83% compile-dead" | 16 SyntaxError + 12 Exception rows + 8 exclusions (92 scored) |
| AR twin (B3) | not built | **0.92** in 9.35 min, NFE 333/prob, ~1.4k position-forwards/prob |
| Best-of-K headroom | p@4 > p@1 presumed | **zero**: p@1 = p@4 in every cell, both engines, at K=4 |
| Diversity @ T=0.2 | "collapse" qualitatively | s256: 13/25 problems byte-identical across seeds; s512: 20/25 |

Consequences for this design:

1. **The bar is now a curve, not a point**: steering must beat the *corrected*
   unsteered staircase (0.16/0.20/0.44/0.64 at NFE/prob 384/512/1024/2048) at
   matched NFE — with the same-harness AR control at 0.92 as the ceiling
   reference (M2.4b §M2.5: "L1 makes the dLLM competitive with its AR twin" is
   the strongest available framing).
2. **"More steps buys 0.20" is the null model**: steps 256→512 costs +1024
   NFE/problem and returns +0.20 honest pass@1. The steering claim must
   show what the same NFE buys when spent inside the denoise loop *under
   verifier control* — otherwise just raise steps.
3. **Selection-headroom at K=4 is exactly zero** (p@1 = p@4 everywhere):
   post-hoc filtering (a Prism-style verifier filter over finished samples)
   has nothing to filter on this slice. Any steering value must come from
   *doing* things mid-decode (kill/clone/re-decode), not from choosing among
   finished identical canvases. This is measured, not assumed.
4. **The M2.4a L1 premise — "83% of canvases are compile-dead weight that
   resampling should kill early" — is now 39% at s256** (36/92 scored) and
   17% at s512, and (§1.2) the mass is no longer mostly parse corruption:
   it is dominated by *deterministic, type-level, single-mode blunders* that
   survive doubling the budget.
5. **(rev1, reviews A1/B1) The capability claim is dead; the mechanism claim
   lives — and this doc now says so up front.** The AR twin's
   0.92-in-9.35-minutes refutes every practical/capability reading of this
   program; the design's own realistic steered ceiling (0.72–0.84) is
   *below* the twin's already-measured point. M3.0's deliverable is
   therefore explicitly **(a) the engine** (substrate, verifier ladder,
   metering, replay, held-in guards) and **(b) one mechanism question,
   measurable or falsified**: *does verifier-controlled compute reallocation
   inside a masked-diffusion decode beat diverse-sampling-plus-filter at
   matched NFE?* A score of 14/25 without an answer to (b) is not a result
   — it is bookkeeping. The publishable facts of M3.0 are the engine, the
   measured mid-decode-verification validity curves, and the signed answer
   to (b), whatever its sign.
6. **(rev1) Why demonstrate on the dLLM substrate at all, given M2.4b §M2.5
   item 3 recommends AR as the L1 base?** Honest answer, stated once: AR
   verifier-signal resampling is real and likely strong — but it is a
   *different* mechanism, and the two things this program studies exist
   only on the diffusion side: (i) **mode-retention** — whether a
   masked-diffusion decode distribution retains semantically distinct
   correct modes that decoding policy can select (the question no AR
   prefix process can even pose the same way, because an AR canvas is a
   committed left-to-right path), and (ii) the §6.2 item-5 asymmetry:
   commits are monotone, so kill-on-committed-fatal-evidence is *sound* in
   diffusion in a way it never is mid-AR — while swiss-cheese mid-canvas
   verification has no AR counterpart either (§3.4). Both the burden and
   the advantage are diffusion-specific. The AR-substrate recommendation is
   recorded, engaged, and deliberately deferred to a follow-on lane; it is
   an *answer to a different question* than the one this doc asks.
   M3.0-as-syntax-rescue is infrastructure; the toot's property-steering
   thesis is M3.1's content and M3.2's frontier (reviews A1/B1 both:
   say-so-louder).

---

## 1. The problem L1 must solve, re-derived on the corrected basis

### 1.1 The corrected staircase (the curve L1 must beat, same NFE units)

dLLM-internal NFE = one full-canvas denoise-step forward (prompt + 384 canvas
positions re-encoded); exactly commensurable between steered and unsteered
Dream cells. Wall and position-forwards carried per the decision-doc
protocol (cross-engine tables must show both; wall is the honest tiebreaker).
Container = official rustc (2024-07-12 image, 1.75.0); honest =
exclusions folded in as failures; official = normalized = honest on every
measured cell.

| cell | NFE/prob | wall | hon p@1 | passers | scored-fail structure |
|---|---|---|---|---|---|
| s96 | 384 | 6.18 h | 0.16 | 4 | 60 SE + 8 Ex of 84 scored |
| s128 | 512 | 8.82 h | 0.20 | 5 | 72 SE + 0 Ex of 92 |
| s256 = B2 | 1024 | 7.21 h | **0.44** | 11 | 36 SE + 12 Ex of 92 + 8 excl (17,19) |
| s512 | 2048 | 14.48 h | **0.64** | 16 | 16 SE + 12 Ex of 92 + 8 excl (17,19) |
| B3 AR twin | 333 (token units) | 0.16 h | **0.92** | 23 | 12 wrong-output ×4, 19 half; 0 exclusions |

Comparators for the honest NFE conversation (decision doc §2): UMF's
NFE/problem = 12,288 (swept 768→24,576); Prism 2,480 (self-verified LLaDA-8B);
our matched band is 1024–2048 — 6–12× under UMF's regime, inside Prism's. Our
instrument reports NFE + wall + verifier-time + position-forwards; every
published claim must carry wall next to NFE (M2.4b carry-forward #a).

### 1.2 What the residual disease actually is (fresh forensics, this doc)

I re-diagnosed all 48 non-OK scored rows of the corrected s256 cell and all 28
at s512 by reconstructing `prompt + completion + "\n" + tests` from the
committed fix-basis exports and running host rustc 1.99.1 `--edition=2021`
(CPU-only, committed artifacts; script commands in §10). Every failing problem
fails **identically across all 4 seeds** at s256 — one deterministic failure
signature per problem, not stochastic noise:

| class | problems (s256) | rows | rustc evidence | still sick at s512? | what signal catches it |
|---|---|---|---|---|---|
| **type-level API blunder** `f64: Ord` | 0, 20, 21 | 12/36 SE | E0277 `sorted.sort()` on `Vec<f64>` — needs `sort_by(partial_cmp)` | **YES** (all 12 rows), unchanged across budget ×2, ×2 seeds density | stable-rustc type-check (35 ms) |
| **type-annotation ambiguity** | 4 | 4/36 SE | E0282/E0283 type annotations needed | YES (4 rows) | type-check |
| **move/borrow violation** | 12 | 4/36 SE | E0507 cannot move out of index of `Vec<String>` | clears at 512 (becomes OK) | type-check |
| **true parse corruption** (delimiters, mid-expression) | 6, 9, 10, 25 | 16/36 SE | unexpected/mismatched closing delimiter; expected identifier found `=` | ALL clear at 512 | type-check (trivially); rung-0 shape heuristics partially |
| **wrong-but-compiles semantics** | 11, 18, 24 | 12 Ex | compiles; panics on the held-out fixtures (e.g. HumanEval_11 consumes 2 chars/loop with a double `chars.next()`; 18 wraps `str::find` logic; 24 misses a divisor) | YES (all 12) | only properties (M3.1 rung R3) or held-out scoring |
| **prompt-shape/exclusion pathologies** | 17, 19 | 8 excluded | 17: complete-looking body then prose echo after the fence (+ duplicated match arm `'o'` — mode collapse); 19: dies mid-`HashMap` construction | YES (both, deterministic across 2 cells) | rung-0 fence/prose-shape + (19) type-check sometimes |

Three load-bearing readings:

- **"SyntaxError" is a container label, not a rustc parse verdict.** At the
  corrected s256 point, only 16/36 of the compile-dead rows are parse-level;
  20/36 are type-level (E0277/E0282/E0283/E0507) and a `todo`-closure
  type-check catches them. **A parse-only reward rung would catch 44% of the
  compile mass; a full type-check rung catches 100% of it** — and type-check
  on stable rustc costs the same as parsing (§3.2).
- **The resistant core is made of confident single modes.** Problems 0/20/21
  fail with byte-identical, type-wrong code in every seed at every budget
  measured (including s512). Steps do not fix them; resampling identical
  seeds does not (T=0.2 collapse); only diversity + verifier-directed
  compute can. Meanwhile the AR twin passes 20 and 21 outright and shares
  Qwen2.5-Coder-7B as Dream's init — the information lives in the family;
  the question is Dream's decode distribution reachability (§6, §7 probe).
- **Failure mass concentration is opportunity**: the classes that budget
  already fixes cheaply (parse corruption 6/9/10/25 + borrow-class 12 —
  doubling to 512 = 0.44→0.64 buys exactly those 5) are NOT where steering
  has a unique story at 2048; the problems only steering can reach are
  the type-blunder trio (0/20/21), E0282 (4), the semantic three (11/18/24),
  the exclusion pair (17/19) — 7–9 problems; the realistic steered ceiling on
  this slice is ~0.72–0.84 with the AR row 0.92 as reference.

---

## 2. Decision 1 (Q1): the steering substrate — a temper-owned denoise loop
driving checkpointed, budget-first sequential SMC

### 2.1 What the model actually exposes (verified in vendored source + our fork)

Facts pinned by reading
`~/.cache/.../Dream-Coder-v0-Instruct-7B/generation_utils.py` and
`temper/model/dream_fixed.py`:

- The native `_sample` loop is *per-step simple*: state = one canvas tensor
  `x` (prompt + canvas, masks = `MASK_TOKEN_ID`); each step = one full-canvas
  forward, one confidence-ranked commit of
  `⌊n_mask·(1−s/t)⌋` tokens; **commits are monotone** (committed positions
  never re-enter `mask_index`; only the pre-loop tokens hook can rewrite `x`).
- Both native hooks honor **return-value replacement**:
  `logits = generation_logits_hook_func(...)` and
  `x = generation_tokens_hook_func(...)` — verified in both the vendored loop
  and our fixed-shape port. The surface supports: (a) per-step logit shaping,
  (b) canvas rewriting (re-masking/fencing) mid-loop.
- The MPS substrate is **single-canvas sequential only** (batched K>1 NaNs,
  spike §0; fixed-shape tail is batch-1/entropy/alg_temp=0 — `dream_fixed.py`).
- Commit schedule at steps S: after step k, remaining masks ≈ n·(1−k/S)
  (exact product form under `linspace(1, eps, S+1)`; int-floored per step) —
  **checkpoint k ≈ k/S of the canvas committed**, on average. Real commit
  ORDER is confidence-ranked (structure and high-confidence spans first —
  this is why the M2.2 corruption forensics saw `.sort()`-class tokens land
  early and broken).

### 2.2 The substrate decision

**The engine is a temper-owned step loop ("steer loop"), a strict
generalization of `_sample_fixed`, not a hook-only add-on.** Rationale: SMC
needs cross-particle state (kill/clone/genealogy) at *inside the loop*
granularity; `diffusion_generate` runs one canvas start-to-finish and its
hooks alone cannot maintain a particle ensemble. The steer loop:

- reuses the fixed-shape sampling tail verbatim
  (`_entropy_sample_tokens_padded` + the tie-free masked-row order) so the
  MPS memory fix stays load-bearing (M2.3 phase-3) and unsteered-equivalence
  is checkable;
- keeps K canvases in process memory (each `[1, prompt+384]` int64 ≈ 5 KB;
  output_history clones ≈ 2.5 KB/step — negligible), advances ALL live
  particles round-robin **batch-1 forward at a time** (1 NFE per
  particle-step — sequential execution is merely serialized batch-SMC; NFE
  is invariant to the serialization);
- per-particle deterministic RNG: `torch.manual_seed(f(problem, base_seed,
  particle_id, resume_epoch))` at each (re)start; every decision and stream
  sha enters the steer log → replayable offline with zero GPU (§7 D2);
- NFE = every executed forward (kill stops accrual); verifier time is never
  NFE — it is a separate axis (`verifier_wall_s`), as the instrument already
  prescribes.

**Particle structure = checkpointed SMC with hard budget envelope.** One
"game" per (problem × envelope): budget B = NFE/prob cap (B ∈ {1024, 2048}).
K_max_eff ≤ 8 canvases live-registered. Checkpoints at step grid
`{round(S·0.25), round(S·0.50), round(S·0.75), S}`. At each checkpoint:
rungs fire (§3), then the policy acts:
- **kill**: a particle with *positively-diagnosed* fatal evidence
  (informed-class rustc errors inside committed spans; rung-0 integrity
  violations) is marked dead → stops consuming NFE;
- **clone**: while budget remains and live particles < K_max_eff, spawn
  children from the *healthiest surviving parent's checkpoint snapshot*
  (canvas COPY + fresh RNG seed + temperature +0.2 step, capped 0.8):
  children inherit the committed healthy prefix and re-sample the remaining
  holes — this is the FK/SMC resampling step: reallocate compute toward
  high-weight genes. Fresh-canvas replacement (new seed, same temp ladder)
  is the alternative spawn since a checkpoint snapshot may have no healthy
  parent; v0 uses: clone-if-healthy-parent-exists else fresh;
- **weights for the clone lottery**: v0 is deliberately coarse — binaries
  from the ladder (clean = 1, any positively-diagnosed fatal = 0 → dead);
  among clean parents the lottery is uniform; the **final answer** per game
  is the last checkpoint's best survivor ranked by
  `clean > mean_commit_conf` (the model pseudo-likelihood proxy M1 already
  established). Finer rewards (error-count, property gradations) are
  deferred to M3.1 (§5) — reward-shape tuning is a hacking surface we refuse
  in v0.

Why sequential-SMC at all, given K=4 BoN would be simpler: because the
measured substrate facts say BoN-at-T=0.2 is *empty* (p@1=p@4, byte-identical
canvases — the selection axis has literally zero variance to select on), and
because kill-savings + clone-conditioning are the only mechanisms that make
same-NFE compute buy different outcomes. The whole design stands or falls on
**diversity injection** (temp ladder for children) + **positive-evidence
killing** (§3.4) + **prefix-conditioned clones**; each of the three is
individually ablatable — the matrix is §4.3.

Also named and rejected for v0:
- **Logit shaping / reward gradients** (reward-guided diffusion à la
  classifier-guidance, DiffuCoder's SDDPG policy-gradient, VRPO): needs a
  differentiable or learned reward proxy; ours is discrete 0/1 rustc/proptest
  signal with no gradient path; faking one (token-level learned reward model)
  is a research project with reward-hacking risk, not an engineering
  milestone. Out of scope for M3; revisit only if M3.0–3.1 fail.
- **UMF-style one-token-at-a-time MCTS**: 12,288-NFE regime, public-test
  reward (their PublicSearch split — the exact leak our held-in discipline
  forbids; M3 decision doc §1 "never steer on scoring tests"), and
  tree-per-node expansion cost too high for a 1024–2048 band.
- **Batched particle forwards**: MPS NaNs (spike §0). Round-robin
  serialization instead; NFE accounting is unaffected.

---

## 3. Decision 2 (Q1+Q2): the verifier ladder — what it may see, when it fires,
what it costs (measured)

### 3.1 What the verifier can safely see mid-decode, and the hard rule

Mid-denoise canvases are **not partial Rust programs**:
- masked positions are unknown-slots (decode as `<|mask|>` residue), and the
  committed set is confidence-ordered swiss cheese, NOT a prefix — Rust has
  no incremental/prefix parser contract the way AR decoding has;
- masked diffusion cannot be "parsed incrementally" the way tree-search AR
  decoders exploit (their "completed candidate prefixes" are always-valid
  prefixes; ours are interleaved holes) — only the hole-free detectors
  (rung 0 below) need no assumption about what hides in the holes;

**The ghost-closure scheme (defensible mid-decode signal)**: at a checkpoint,
render the canvas's **fenced code region** (extract_rust over the decoded
text — shared pipeline with the offline instrument), ghost-close it
deterministically, and type-check as a **library crate**:

```
ghost-closure(program) = problem_prompt (verbatim, ends "…fn name(…) -> T {")
                       + todo!()-substituted hole-runs in the rendered body
                       + balanced delimiter tail computed by brace/paren scan
(no vendored tests, no main, no execution: rustc --edition=2021 --crate-type lib --emit=metadata)
```

- Every contiguous masked run in the body is replaced by the stdlib universal
  expression-hole `todo!()` (expression & statement positions type-check;
  type/attribute positions do not — that limitation is handled by the
  verdict filter below, and is WHY the naive rule fails, §3.2).
- **The verifier NEVER sees the MultiPL-E `tests` string** — not at steer
  time, not in ghost-closure. Closing as `--crate-type lib` needs no
  `main`; the `tests` fixtures stay held-out for the container only. The
  "prompt+current-body+tests ghost-closure" variant from the original sketch
  is REJECTED on the held-in discipline (decision doc: steering must never
  see scoring tests; UMF's public/private mismatch is the named precedent).
  Enforcement is structural, not procedural: the reward builder receives a
  **steer-view whitelist object** `SteerProblemView(name, prompt, fn_name,
  signature_line)` constructed from the vendored task dict by an allowlist
  constructor (`payload = {k: td[k] for k in ("name","prompt")}`); a unit
  test pins that no reward-side module reads `tests` (D5, §7).
- Mid-decode reward source is only: problem prompt (the model sees it too —
  symmetric information), the canvas, rustc's stderr, and (M3.1) APPROVED
  proptest properties. 17-class prose-echo detection needs only fence
  geometry. No scoring-test-adjacent signal anywhere in the return path.

### 3.2 The rungs — with the measurements that size them (host core, no GPU)

All latencies measured for this doc on this M3 Max host, stable rustc 1.99.1
(nightly-only `-Zparse-only` REJECTED — no nightly dependency in the loop):

| rung | signal | cost (measured) | fires |
|---|---|---|---|
| **R0 shape** (free, pure-python) | extract_rust façade over the rendered canvas: fence integrity; prose-after-closing-fence (the HumanEval_17 class); mask-residue sanity (expected residual mask share at step k ≈ (1−k/S)±tolerance — catching stuck canvases); must_define fn-header presence late in decode; brace trajectory (running unbalance beyond what `k/S` holes explain) | <1 ms | every checkpoint |
| **R1 ghost type-check** | rustc `--crate-type lib --emit=metadata` on the §3.1 ghost-closed render | **p50 31 ms / p95 38 ms / max 40 ms** (n=552 timed batch over the s256 corpus with synthesized holes; process-spawn dominated) | every checkpoint per live particle |
| **R2 anti-stub** (R0-family rule) | committed stub markers (`todo!(`, `unimplemented!(`, `panic!` literal) — note commits are sticky, so a committed stub NEVER self-repairs → sound mid-decode kill category; rung exists because R1 is blind to compiles-looking stubs | free | every checkpoint |
| **R3 property-check** (bench tasks now; humaneval in M3.1) | M0 harness run (seatbelt, proptest 256 cases) on the *completed* body fragment + approved properties | 236 ms–5.9 s/candidate (M0/M1 measured) | final checkpoint only, ≤ K_max_eff calls/game |

Verifier arithmetic at the worst planned cadence (K_max_eff 8 × 4
checkpoints): R1 ≈ 8×4×35 ms ≈ 1.1 s per problem ≈ **0.1%** of a
2048-NFE problem's decode wall (~35 min); R3 at the final checkpoint,
worst case 8 × 5.9 s ≈ 47 s ≈ 2.2% — hence R3 is final-only and its wall
is metered separately and capped (kill criterion §5); at the 1024 envelope
the same absolute cost is ~0.4% / 4.4% respectively — D1 re-measures on
live cells and the §5 kill-criterion-4 (verifier wall > 10%) is the guard.

**Honest evaluation of the naive use of R1 (this is the negative result that
shapes the design).** I measured a simulation pilot on the committed s256
corrected-basis corpus (44 OK rows + 36 SE rows): canvases simulated by
poking word-level hole-sets (30% of word units, nested subsets for
checkpoint fidelity), todo!()-closure, stable-rustc type-check:

- **Any-error ⇒ kill is unusable mid-canvas**: at 25–100% hole density,
  32–42 of the 44 OK rows also produce *some* rustc error (the
  `todo!()`-closure manufactures parse noise at hole boundaries). Naive R1
  would murder ~80–95% of healthy particles.
- **Informed-class-only verdicts are high-precision, low-recall** (filtering
  to E0xxx classes: E0277/E0282/E0283/E0507/E0382-style substantive
  diagnostics, and requiring the error to persist/point inside committed
  content away from hole-adjacent spans): false-kill 0–2/44 (≤4.5%) at every
  simulated commit level; true-flag on dead rows rises as the canvas fills.
  Total-committed fraction ≈ 1 − 0.30·ck (ck = held-back share of the 30%
  hole set): 0/36 at 70–85% committed, 1/36 at ~85%, **5/36 at ~92.5%,
  11/36 at ~96.4%, 20/36 at full commit** (the last = exactly the 20
  type-level rows of §1.2) — with **0/44 false kills** (informed channel,
  all levels).
- **At the final checkpoint (no holes) R1 is exact on this corpus**: 36/36
  true fatal detection, 0/44 false — the completed-body screen is a
  reliable, ~35 ms per-particle verdict; every emitted completion can be
  guaranteed non-compile-dead *when any type-clean particle exists in the
  game*.

Reading it honestly: mid-canvas, the *safe* kill signal is sparse — the
sim (uniform-random commit order) shows recall 0–31% before the final
checkpoint; the real loop's confidence-ordered commits commit structural,
fatal-in-substance spans EARLY (`.sort()` configs, junk declarations), so
real recall should exceed the sim's — but that is an assumption to
measure, not assume (D4, §7, re-measures on live pre-flight canvases from
the first steer probe). The design consequence is a **patience rule**:

### 3.3 The verdict & action policy v0 (positive-evidence-only)

1. **Absence of evidence never kills.** Mid-canvas, only *positively
   diagnosed* fatal evidence acts: informed-class rustc errors whose primary
   span sits inside committed (non-ghost) content, stable across ≥2
   consecutive checkpoints OR at a stage the rung has measured reliable
   (D4 decides the actual rule; start: 2-checkpoint persistence + span-
   localization); R0/R2 violations (prose-after-fence, committed stub,
   irreparably divergent mask-share).
2. **The final checkpoint is a hard screen**: rung R1 on the completed body;
   the game's emitted answer is the highest-ranked R1-clean survivor
   (Ranking: R1-clean > R3 result (when present) > mean_commit_conf); if no
   CleanSurvivor exists, emit the max-confidence particle — honestly counted
   (its SyntaxError is a scored failure, as today).
3. Kills triggered mid-canvas free NFE for clones/fresh replacements within
   the envelope; nothing is ever permitted to exceed B (envelope guarded in
   the loop, asserted in unit tests; D3).
4. Weights, in v0, stay binary/clean (§2.2) — deliberately un-hackable; the
   only signal a policy can learn to game is "R1-clean", whose anti-hack
   guards are R2 + (M3.1) R3.

### 3.4 Why partial-program verification is genuinely hard here, and what
prior work does and does not hand us (honest placement)

- AR tree-decoders run incremental *prefix* parse/compile gates because the
  token stream is a valid-ish prefix at all times (validated grammars,
  syntax-aware decoding). Masked-diffusion canvases destroy the prefix
  property — the literature has no ready machinery for swiss-cheese
  programs; our todo!()-ghost-closure + informed-span filter is a defensible
  substitute, bought with the measured precision/recall profile above.
- Reward-guided diffusion (per-step gradient nudging) needs differentiable
  rewards (FreDoM-class) or a trained proxy — ours are discrete
  pass/compile flags; direct mapping unavailable, and learning a surrogate
  is out of scope for a Two-phase budget (revisit only on M3.0/3.1 failure).
- UMF (MCTS over unmasking actions): 12,288-NFE regime ≫ our band;
  per-node expansion cost; public-tests-as-reward rejected on held-in
  grounds. Not a match at matched budget.
- Prism (self-verified filtering, SVF-trained verifier): closest
  published comparator to our band (2,480 NFE); it verifies COMPLETED
  samples — a static filter. Our zero-headroom measurement (p@1=p@4 at
  K=4, collapsed seed diversity) is exactly why static filtering cannot be
  the whole mechanism here; p@k vs k curves must be reported at their
  protocol T-points anyway (§4).
- Dream-Coder's own decoding has no steering hooks beyond the ones we own —
  "diffusion search" work at 7B-scale with Rust + held-in properties is our
  niche (decision doc §5: horizontal-offset claim vs Monkeys null model).

---

## 4. Decision 3 (Q3): the budget-matched evaluation plan

### 4.1 Cells and the envelope semantics

- **One steering game per problem per envelope.** The steered cell's unit is
  a budget-B play: B NFE spent under policy → one emitted completion →
  container verdict (binary). NFE/prob comparisons against the unsteered
  staircase are then exactly commensurable: the unsteered anchors spend the
  same per-problem NFE across K canvases, and on this slice their per-problem
  OK counts are 0 or 4 (all-or-nothing — measured), so their honest p@1
  equals the fraction of problems with ≥1 OK — the same per-problem binary
  an envelope play yields; on slices with partial counts the unbiased
  estimator counts them automatically and stays comparable. A steered cell
  may run ≤2 games per problem at B/2 each for extra Bernoulli draws if
  phase-0 shows game-among-games variance; default: 1.
- Primary cells (all humaneval-rs first 25 problems, same INSTRUCTION
  template sha, base_seed+i, official container, dual edition, norepair +
  repair both scored):

| cell | policy | NFE envelope | wall est. | purpose |
|---|---|---|---|---|
| **anchor-1024** | — (the corrected s256 row, committed, no re-run) | 1024 | 7.21 h (banked) | the 0.44 line |
| **anchor-2048** | — (corrected s512, committed) | 2048 | 14.48 h (banked) | the 0.64 line |
| **C-hint** *(rev1, review A add; scheduled FIRST in §5)* | unsteered, K=4, steps=256, T=0.2, problems {0,4,20,21} (the deterministic type-trap quartet), prompt + one pre-committed generic anti-trap hint line (exact text frozen in verdict-config before launch; e.g. "for `Vec<f64>` sorting use `sort_by(|a,b| a.partial_cmp(b).unwrap())`; prefer explicit type annotations") | 10 | ~1.2 h | knowledge-injection null: if a free hint reproduces the flips, syntax-rescue is information-delivery, not search — dead early and cheap; if not, steering's reachability result gets stronger |
| **C-null** | temp-ladder diversity only (T ∈ {0.2,0.4,0.6,0.8} round-robin), no rungs, no kill/clone, K_max_eff to budget. *(rev1 emission spec, closing review B's spec hole)* Emission rule, pre-committed: the T=0.2 canvas if present in the ladder, else the lowest-T canvas; per-completion container statuses and the full ensemble recorded for C-screen | 2048 | ~14.9 h | attribution control: diversity alone |
| **C-screen** *(rev1, review B add; computed, not generated)* | no generation — container-score C-null's FULL 8-canvas ensemble (container-minutes), then offline counterfactual from committed sidecars: apply the FROZEN informed-class filter to completed canvases and emit "best R1-clean by confidence (ties: lowest temp)" | 0 extra (2048 envelope of C-null) | ~minutes | static-filter null: temp ladder + final rustc screen, ZERO mid-decode machinery — the Prism-style known-art mechanism the design must beat or be reduced to |
| **C-full** | R0+R1+R2, kill, clone-from-checkpoint, children temp +0.2 | **1024** & **2048** | 7.5 h & 14.9 h | the L1 rows |
| **C-kill** (described, run if budget permits) | early-aborts only, no cloning (fresh replacement at fresh temp) | 2048 | 14.9 h | separates "kill-savings" from "clone-conditioning" |
| **C-b512** (stretch) | same policy as C-full | 512 | 3.7 h | sub-anchor leverage point |

- Steering runs entirely through the existing bench driver via its
  `sample_fn` injection seam (the M2.4b Qwen path — proven instrument-unchanged):
  `temper bench-multipl-e --engine steer --steer-config <json>`; meta gains
  `mode:"steered"` + `steer_config_sha256`; resume config-match includes the
  steer-config sha; steer-side event log per problem in
  `<out>.steer/<problem>.jsonl` (D2). Export/score paths untouched (the
  post-M2.4b instrument is load-bearing and regression-tested — we do not
  edit `export.py`/`to_body_fragment` semantics at all; steered rows also
  never receive L0.5 repair at steer time; export-time repair remains the
  a-priori-known norepair/repair column pair).

### 4.2 Statistics: "Don't Pass@k" Bayesian gates (compute-cheap, decision-grade)

- Estimator: per-problem binary outcome (game's emitted completion verdict);
  honest = exclusions/gen-errors folded as failures; official + normalized
  both reported.
- **(rev1, reviews A3/B4 — gates rewritten) Decision quantities, in order of
  authority:**
  1. **Paired migration matrix + one-sided sign-test (PRIMARY).** Problems
     are deterministic single draws, NOT exchangeable Bernoulli draws — the
     pooled Jeffreys Beta(0.5+x, 0.5+25−x) treats them as iid and is
     demoted to a reporting convenience, printed only with that caveat.
     Every gate decision is made on the paired per-problem migration vs the
     banked anchors and on sign-test p-values with ex-ante
     detectable-effect statements (D6 emits McNemar/sign-test numbers for
     every gate).
  2. **Pre-registered primary comparison (single):** C-full@2048 vs
     C-null@2048, paired — its outcome decides the mechanism question in
     §0.5(b). Every other comparison (envelope response, C-hint/C-screen
     deltas, 1024 cells, M3.1 gates) is secondary/exploratory and labeled
     such. This composes the ~6 gate-families' multiplicity down to one
     primary; no FWER formality, one honest sentence and one primary.
- **SHIP GATE (M3.0)**: C-full@2048 ≥ **15/25** honest *with paired margins*:
  zero net regressions among the 11 corrected s256 passers, sign-test vs
  anchor-1024 reported, and **≥ 2 confirmed-attributed flips** (flips that
  survive the §4.3 attribution battery incl. replication ≥ 2 games/problem
  on the flip set, P ≈ 0.55–0.65 conditional on P1-GO). Language note:
  underpowered for effects < +3 problems — every ship/no-ship line says so.
  (rev0's bare "14/25, P(θ>0.44)≥0.85" is superseded: it shipped ~12%
  one-sided false-positive per cell on pooled-noise alone.)
- **STRONG GATE**: ≥18/25 (0.72) at ≤2048 with the same paired discipline —
  i.e., steering buys what the next doubling of unsteered budget cannot
  (projected from the budget-invariance forensics §1.2, stated as
  projection; a 9-problem s1024 spot ~5 h is run ONLY if the strong gate
  becomes load-bearing for the headline).
- **Attribution GATE (rev1, the causal story)**: C-full@2048
  **> max(C-null, C-screen, C-hint-relevant-cohort)** on the paired matrix,
  margin ≥ +2 problems, with at least one flip OUTSIDE the
  hint-fixable/ladder-reachable cohorts (type-trap quartet / anything in
  C-screen's clean set), flip-set replication green. C-full − C-screen is
  the mechanism's incremental yield and the single number the §0.5(b)
  question is priced by; C-hint's quartet outcome bounds the
  information-vs-search decomposition; if C-full = C-null = C-screen,
  steering-as-reallocation is a measured null and is reported as one
  (phase kill, §5 — including the ambiguous zone review B flagged: any
  non-ship outcome with C-full ≈ either null is pre-registered negative,
  not iterated past).
- Paired analysis accompanies every gate: per-problem migration matrix vs
  corrected s256/s512 (m2_4a_migration.py extended), plus the Monkeys
  framing: steered β-offset along the NFE axis (does the steered point sit
  LEFT of the unsteered staircase at equal accuracy — horizontal offset,
  not curve-riding).
- Jeffreys pooled posterior reference values (reporting-only, rev0
  provenance): x=14/25 → P(θ>0.44)=0.886; x=16 → 0.978; x=18 → P(θ>0.64)=0.795.
- Protocol honesty: T=0.2 is the B2 anchor point; C-full's temp ladder is an
  explicit protocol deviation, WHICH IS WHY C-null exists at exactly the
  same ladder; report both against anchor-1024 with the deviation printed.
  pass@10/100 stays out of reach at K≤8 (n≥k requirement) — the 20-100
  sample/problem protocol points stay future work at full-156 scale.

### 4.3 The isolation matrix (rev1 — three mechanisms, each isolated; review B's §0.3-vs-§4.3 contradiction removed)

A steered cell bundles three separable mechanisms: (i) diversity injection
(temp ladder), (ii) final-screen selection over completed diverse canvases
(Prism-style known art), (iii) mid-decode kill/clone budget reallocation
(the novel part — called what it is: **verifier-conditioned budget
reallocation**, not logit-level guidance; M3.0 gives the verifier no
logits). rev1 makes each isolable by mandatory cells/computations:

| mechanism | isolated by | status |
|---|---|---|
| (i) diversity ladder | anchors (T=0.2) vs C-null | mandatory cell |
| (ii) final-screen selection | **C-screen** counterfactual from C-null's ensemble (exact, free) | mandatory computation |
| (iii) mid-decode machinery | C-full − C-screen (primary delta) | mandatory cell |
| information vs search | **C-hint** on the type-trap quartet | mandatory cell (first) |
| kill-vs-clone | C-kill vs C-full | optional (runs only if the primary delta ≥ +2 and is contested) |
| rung sensitivity | C-full minus R1 (R0-only @1024) | optional |
| envelope response | C-full@{512,1024,2048} | 1024 mandatory-ish (cut order last), 512 stretch |

Naming honesty (review B): nothing in M3.0 steers sampling distributions —
information flows only via compute allocation and RNG/temp choice; every
place this doc says "steering" in reference to M3.0's mechanism is this
reallocation sense. M3.1/M3.2 are where the signal reaches INTO the
decoder (ranking weights / re-masking).

Every cell reuses banked anchors where identical config permits; nothing is
re-run "for symmetry" (GPU wall is the binding constraint; §9).

---

## 5. Decision 4 (Q4): phasing, gates, kill criteria

**Sizing fact per cell**: canvas forward ≈ 1.03-1.04 s/NFE (M2.4a/b LS
fits); C-full@2048 ≈ 14.9 h wall + R1+R3 verifier ≈ +1–3% + container
scoring minutes. Probe ≈ 5.5 h. Primary P0–P3 path ≈ **43 h GPU +
scoring** (§9); full ablation matrix ≈ 62–74 h; the gates cut from the top.

### M3.0 — syntax/type-reward steering prototype ("TypeCheck-SMC v0")

- **P0 (instrument, CPU-only, mandatory first — project rule
  instrument-before-engine)**: D1–D6 of §7. Exit: all D-acceptances green.
- **P0.5 (C-hint cell, ~1.2 h GPU — rev1, review A; scheduled first because
  it is the cheapest falsifier in the program)**: problems {0,4,20,21},
  unsteered K=4 steps=256 T=0.2, prompt + the pre-committed hint line,
  container-scored. Outcomes (either strengthens the program): passes ≥ 3/4
  ⇒ type-trap cohort is information-fixable — their flips are attributed to
  knowledge delivery and the mechanism claim must live elsewhere
  (semantic/exclusion cohorts, M3.1); passes ≤ 1/4 ⇒ the traps are
  mode-lock, not knowledge — steer-reachability becomes the more
  interesting claim. Either way the §4.2 attribution gate's C-hint term is
  populated. (review A's own bet: passes all four ~free.)
- **P1 (mode-existence probe, ~8.5 h GPU — rev1-deepened, reviews A4/B4)** —
  the falsifier for the load-bearing assumption (§6): 9 problem set {0, 4,
  20, 21, 11, 18, 24, 17, 19} × **3 temps {0.4, 0.8, 1.2}** × K=4 ×
  steps=256, container-scored, reachability CURVE reported (not just a
  ladder boolean). **GO, cohort-stratified**: ≥4/9 overall AND **≥2/4 of
  the type-quartet** (0/4/20/21) AND ≥1 non-type problem. **PARTIAL
  (2–3)**: the EV in §6.1 is REWRITTEN from measured reachability before
  any C-cell launches (no inherited thresholds; review B's garden-of-
  forking-paths closure); the 1024-envelope cell is still first on the cut
  list. **NO-GO (<2)**: before canceling, run an s512-variant recheck on
  exactly the failing cohort (mode-at-coarser-commit-grid vs genuinely
  absent — ~+3 h); if still absent, pivot: cancel C-full; M3 becomes
  (a) property-feedback re-mask (M3.2 first), or (b) template-hardening +
  full-156 slice change; report honestly.
  Bets on record (§11): review A predicts the quartet is reachable under
  mild diversity; review B predicts the GO is driven by 17/19 with ≤1–2 of
  the type quartet.
- **P2 (engine equivalence & bench-internal)**: steer loop in "all-off" mode
  must reproduce `dream.sample` canvas outputs **bit-identically** on a
  4-canvas s128 spot-check (same method the M2.3 patch was gated by);
  NFE/verifier metering unit tests; D2 replay re-derives every P2 decision
  from logs alone. Bench-internal proving ground = bench/sort,
  bench/ringbuf, bench/wrap-sum (R3 live there: approved properties; also
  the corpus/replay machinery stays identical). Mechanism gates only, NO
  capability claim here (ringbuf has never passed).
- **P3 (humaneval cells — rev1 order)**: D7 verdict-config freeze committed
  (sha in every subsequent record) → C-null@2048 first (it feeds both the
  attribution gate and the free C-screen) → **C-screen computation**
  (container-score the C-null ensemble, offline frozen-filter
  counterfactual — done BEFORE C-full result analysis so the primary
  delta's yardstick exists ahead of the steered rows) → C-full@2048 →
  C-full@1024 (+ C-kill@2048 only if the primary delta ≥ +2 and is
  contested), all scored on the official container, same counting as §4.
- **M3.0 SHIP GATE**: §4.2 ship gate + attribution gate + meters (verifier
  wall ≤ 5% of gen wall; NFE attribution exact; replay determinism green).
- **M3.0 KILL CRITERIA** (any one ⇒ stop/redirect, each with its detector):
  1. P1 NO-GO (distribution lacks alternatives) — probe.
  2. Equivalence fails after two fix attempts — instrument invalid, step
     back to hook-only designs with reranking (report why).
  3. C-full ≤ anchor-1024 point estimate AND ≤ C-null (steering adds
     nothing over diversity/budget) — M3.0 declared negative result;
     freeze numbers, pivot to property-feedback or template work.
  4. R1/R3 verifier wall > 10% of gen wall (engine turning into a
     verifier-farm; redesign cadence/rung placement via D1 data).
  5. Reward-hacking corner: pass@1 flat while the SyntaxError share
     collapses ≥50% AND Exception share climbs (steering squeezing
     completion quality into "compiles-but-wrong" modes) — detector
     enabled by design (failure-class histogram is already machine-readable
     in `temper eval`); response is M3.1 property-grade ranking, not
     rung removal.
- CPU-vs-container verdict-divergence check: all steer-time verdicts
  regenerate at export from sidecars in OFFLINE tests with host rustc; the
  container remains the only authority for scored outcomes (documented as
  with the M2.0 smoke).

### M3.1 — property-reward steering (the temper toot's core idea: proptest as the steering signal)

- Scope: R3 (approved property checks with minimal failing input as the
  gradation payload) becomes a **final-checkpoint weight** (gradations:
  no-extract < extract+compile-dead < compile+property-fail < property-pass),
  and (still v0-adjacent, binary-family weights) the emitted answer becomes
  the highest property-passing survivor, with proptest `minimal_input` +
  persisted failing seed recorded per game (the corpus machinery from M1.5
  gains a `game_id` lineage).
- humaneval-rs property slope: GLM-5.3 drafts properties **from
  `SteerProblemView` only** (docstring + signature — the same text the
  model sees; never the vendored `tests`), human approves per the M1 §4a
  discipline (~14 min/draft measured throughput; propose drafting only the
  up-to-9 M3.0 non-passers ≈ 2 author-days). The draft tool refuses to run
  if an unapproved draft is referenced. bench-internal tasks already have
  approved properties — the R3 path is proven there by default.
- Cells: rerun C-full@2048 with R3 enabled = C-prop; compare C-prop vs
  C-full (the property-delta cell) and both against anchors.
- SHIP GATE: C-prop − C-full ≥ +1 problem at 2048 with zero regressions,
  OR M3.0 already hit the strong gate (M3.1 deemed redundant if 18/25
  reached without properties; report the redundancy honestly).
- KILL CRITERIA: property authoring cannot keep pace with failures
  (throughput > ~20 min/problem sustained); R3 wall > 20% of gen wall;
  C-prop ≈ C-full on two consecutive cells.

### M3.2 — rejuvenation (span re-masking) — conditional

- Trigger (any): C-kill ≈ C-null while C-full > C-kill (pure
  prefix-conditioning value confirmed AND the family of dead-unless-repaired
  spans dominates); or D4 shows the same informed-class span recurring
  across ≥K/2 particles (post-hoc repairable-at-earlier-checkpoint); or
  M3.1 persistently reports property-fail spans static across siblings.
- Design sketch (for review, not commitment): at a kill verdict, re-mask
  the committed span at/adjacent to the error (or property-fail line) plus
  the following M tokens (M ≈ error-span length, minimum 8), rewind the
  commit-schedule by the re-masked fraction, continue decodes — a
  RePaint-style partial infill done by OUR loop (the native loop cannot;
  the pre-loop/step hook return-value trick is the alternative and stays
  the fallback implementation if the steer loop is rejected).
- Gate: on a 20-canvas pre-flight wind, rejuv-beats-fresh by ≥ +1
  problem-equivalent in simulated outcome rate at equal NFE; else the
  knob is dead and M3 concludes.
- KILL: re-masked canvases repeatedly re-commit the same corrupt tokens
  (sticky-commit amplification), observed as clone-family text collapse.

---

## 6. Honest expected value and failure modes (Q5)

### 6.1 Per-problem recoverability matrix (drives the EV)

| cohort | problems | mechanism in M3 | my per-problem P(recover) | notes |
|---|---|---|---|---|
| budget-fixable Already (parse + borrow corruption) | 6, 9, 10, 25 (parse-level) + 12 (E0507 move-out, also clears) — all 5 pass at s512 unsteered | kill+re-spend and T-ladder; NOT free in a steered game (steered canvases run at 256-class commit schedules + diverse temps, not at s512's finer commit grid) | 0.5–0.8 each in a steered 2048 game | C-full@1024 is the fair fight here: steering must deliver s512-grade canvases at half wall |
| type-blunder trio + E0282 | 0, 20, 21, 4 | diversity+kill+clone (R1 mid-canvas signal IS precisely this class — measured informed precision ≥95%, recall 0–31% pre-final in the random-commit sim, 56% at completion) | 0.25–0.5 each | the family knows the fix (AR twin passes all 4; Dream shares Qwen init); reachability unknown until the P1 probe |
| semantic-wrong trio | 11, 18, 24 | M3.1 property-grade; diversity helps in the meantime | 0.15 (M3.0) / 0.3–0.5 (M3.1 with property-pass selection among diverse siblings) | wrong-but-confident: commit-confidence ranking is blind here |
| deterministic exclusion pair | 17, 19 | R0 prose/fence + diversity + re-decode (19 sometimes flags E-classes) | 0.1–0.25 each | 17's prose echo is rung-0-detectable; both are prompt-shape problems template-A/B could also fix (that stays a separate labeled-cell option, not M3) |

Expected steered outcome at the 2048 envelope (assuming P1 GO,
post-M3.0+3.1, before the regression haircut): 11 base + ~2.5–4 of the 5
budget-fixable + ~1–2 type-blunders + ~0.9–1.5 semantic + ~0.3–0.5
exclusions ⇒ point estimate ≈ 0.60–0.76, minus an expected 0.5–1 problem
of T-ladder-instability regression (the ship gate does not merely watch
this, it demands net-zero) ⇒ working point ≈ 0.60–0.72. Distribution over
the SHIP GATE (**≥14/25**): P ≈ **0.55–0.7 conditional on P1-GO**, P ≈
0.35–0.45 unconditional; STRONG GATE (≥18/25): P ≈ 0.15–0.25.
At the 1024 envelope, subtract ~1–1.5 problems (the budget-fixable cohort
loses headroom and there is no s512-class anchor under it): P(ship-gate@1024)
≈ 0.35–0.5. Regression risk on the 11 passers
(temp-ladder destabilization): ~0.15–0.3 to lose ≥1 — this is why the gate
demands zero net regressions and the C-null run exists (C-null loss ⇒
ladder re-tuning; C-full loss with C-null safe ⇒ rung/clone semantics
bug — engine-forensics, not science).

**(rev1, review B: the null model gets an EV — the omission that hid
P(attribution gate)).** C-null@2048 (ladder only, same envelope): 11 base
(structurally mechanical — the ladder keeps a T=0.2 canvas and the
pre-committed emission prefers it; zero-regression on the null side is
near-guaranteed and is NOT evidence of engine excellence) + the
budget-fixable five at 0.5–0.8 each (the ladder captures most of this
cohort — s512 already bought them with budget alone) + the type-trap
quartet at 0.15–0.35 each (review B: the multimodal idiom is the ladder's
weakest reach) + semantics ~0.15 ×3 + exclusions ~0.15 ×2 ⇒ **point
estimate ≈ 0.52–0.64, i.e. C-null@2048 ≈ 13–16/25** — squarely
overlapping the ship noise band. This is why §4.2 requires ≥ +2 margin
AND ≥ 1 flip outside the hint/ladder-reachable cohorts: those are what
stop "C-null 14, C-full 15" from being sold as a mechanism.
**(rev1) Attribution-gate power statement:** with per-problem single
draws, P(C-full − C-null ≥ +2 under a machinery-null) ≈ 0.15–0.2 by
direct convolution over the cohort table (D6 reproduces this number at
gate-time from the measured C-null ensemble); adding the
outside-cohort-flip requirement plus ≥2-games replication pushes the
confirmable false-positive rate toward the 0.05 grade. Written here so
nobody re-derives it at decision time.

**Honest P(Beat-0.44 goal met within 2 phases [M3.0 + M3.1])**: ∼ **0.55**
(point estimate ≥14/25 ≤ 2048 NFE via ≥1 phase-3 cell + attribution + zero
regressions). Conditional on the load-bearing assumption below being TRUE
(P1-GO): ∼ **0.75**. Conditional on it being false: ∼0.1.

**The load-bearing assumption**: *Dream-Coder's conditional decode
distribution retains the correct alternative modes for the failing problems
at temperatures a mid-loop policy can reach (T up to 0.8 / re-decode with
diversity)* — i.e. the failures are confident-mode dominance (a sampling
problem: steering's home turf) rather than knowledge gaps (a training
problem: steering is useless). Evidence for: same-init family member
(B3/Qwen2.5-Coder-7B) passes 0/4/20/21/11/18/24 at T=0.2 within its
561-second total cell generation wall (all 25 problems); Dream's own
published Python HumanEval ≈0.57 implies the Rust gaps are decode-side at
the failing slice. Evidence against: 322B adapt tokens
may have specialized the modes away. The P1 probe exists precisely to buy
this fact for the price of 5.5 GPU-hours instead of a 15-hour cell.

### 6.2 Failure modes, mitigations, detectors (each listed with its knob)

1. **Steering amplifies mode collapse** (SMC degeneracy: all clones converge
   on one R1-clean gene): genealogy caps (clone cap 2/direct-parent),
   temp-floor ladder for children, gene-diversity meter per checkpoint
   (distinct-canvas-prefix count — the sha-method from M2.4a generalized
   mid-canvas) logged every game; alarm at ≥K_max_eff−2 common prefixes.
2. **Reward-hacking the type-check**: compiles-but-stub/compiles-but-wrong
   corner — R2 anti-stub rung; property rung (M3.1); detector = the
   Exception-share/syntax-fail-share scissors (§5 kill 5). Also mitigated
   structurally by binary weights (nothing to grade between; can't
   optimize a gradient we never give).
3. **Verifier/GPU contention** — measured NOT a risk at R0/R1 (§3.2:
   ≈0.1–0.4% of gen wall); R3 is final-only, serial, metered and capped.
   The real watch-item is sequential-canvas wall growth (M1's
   "sustained-load s/step drift" 0.92→1.9-2.1) — meter s/step per particle
   in the steer log; kill-criterion 4 (§5) protects the budget claim.
4. **False kills on healthy canvases** (partial-verification noise): the
   positive-evidence-only rule + informed-class filter (measured false-kill
   0–2/44 at all sim densities) + D4 re-measured on live canvases before
   C-full ships; residual damage bounded by the zero-regression gate.
5. **Kill-validity inversion** (the committed error that WOULD have
   self-repaired): impossible by mechanism — commits are sticky (verified
   in the loop source): a committed token never re-enters the mask set;
   there is no "later fix" for committed content; killing on committed-span
   evidence is negation-free. (This is a real, checkable, structural
   advantage over AR partial-completion steering.)
6. **Ghost-closure artifacts read as disease**: the §3.2 negative result is
   the design's own measurement — the naive rule was killed by its own
   pilot before it could cost a GPU-hour. D4's live-canvas re-measure sizes
   the final rule; the persistence/span-localization conditions stay until
   D4 says otherwise.
7. **Envelope leakage** (policy spends above B and silently wins by
   budget-cheating): slot-metered NFE assert (D3) + resume-config identity
   + the done-line totals audited by the same `scripts/m2_3_cells.py`
   family extended to steered rows.
8. **Instrument drift**: no `export.py`/`scorer.py` semantics change in the
   M3 code phase (the post-M2.4b export is regression-tested; keep the top-
   level-block test green); any desired repair/extract evolution lands as
   its own labeled-cell, never silently.

---

## 7. Instrumentation that must EXIST before M3.0 engineering starts (Q6 —
the instrument-before-engine deliverables)

D1 **Verifier-latency bench** (`scripts/m3_verifier_latency.py`, CPU-only,
runs on committed artifacts): on all 400 rows of the four corrected-basis
export sets (s96/s128/s256/s512, 100 each) + the raw sidecars: p50/p95 wall
for R0 (ms), R1 ghost-closure type-check (~35 ms; confirm at scale),
R1-per-checkpoint amortized, R3 harness cycle on completed fragments
(236 ms workload). Also: tokenizer decode cost per checkpoint render (expect
ms-level; must be a measured value, not an assumption). Acceptance:
latency table committed with p50/p95 per rung; R1 amortized ≤ 1% of canvas
wall verified.

D2 **Steer-decision replay log** (schema v1.3, additive to v1.2):
`<out>.steer/<problem>.jsonl`: events {game, particle, resume_epoch,
step, kind: spawn|checkpoint|verdict|rng|decision|kill|clone|emit, nfe_at,
canvas_prefix_sha256, committed_token_count, verdicts per rung
(rustc stderr sha + parsed informed-classes + spans), rng_seed, temp}.
Offline replayer (`temper steer-replay`) must re-derive every decision from
logs, no GPU, and unit-test-pinned including a poisoning case (tampered
verdict → replay mismatch). Acceptance: replayer green on a synthetic log
and on P2's spot-check.

D3 **NFE & budget metering**: ParticleResult extension, backward-compatible
schema additions in the v1.2 conventions:
`nfe_model`, `nfe_verifier_calls`, `verifier_wall_s`, with
`nfe_total`/`verifier_wall_s_total` aggregation reusing the v1.2 forms;
per-problem envelope counters in the done-line; assert `nfe_total ≤
n_problems × envelope` on the cell record. Acceptance: unit-tested equality
with hand-built fake games; the C-full record exports/scores through the
UNCHANGED instrument.

D4 **Ghost-closure validity bench on REAL canvases**: over a 2-problem ×
4-canvas GPU pre-flight only after the loop exists in "all-off" mode (this
is the P0→P2 boundary activity): for each checkpoint store the true
partial canvas; compute R1 verdict; report measured (false-kill rate,
informed-class recall curve with commit fraction) — the graph that decides
the final patience rule parameters. Acceptance: the pass rule parameters
are SET by this table and committed, not hand-waved.

D5 **Held-in guard**: `SteerProblemView` allowlist constructor + unit tests
asserting (a) no `tests` content reachable by the reward module (static
allowlist check), (b) adversarial case: an injected hypothetical task dict
with readable-looking tests still yields an empty view. Acceptance:
assert-tamper test committed; grep-rule in the CI checklist note.

D6 **Bayesian/Paired tooling** (`scripts/m3_cells.py`): Jeffreys pooled
posterior + MC P(θ>x) table (the §4.2 gates computed from results JSONs),
paired migration matrix anchors-vs-steered, failure-class histograms
reused from M2.1 taxonomy, per-cell {NFE, wall, verifier wall,
position-forwards} block compliant with the M2.4b NFE/wall dual-report
protocol. Acceptance: reproduces the §4.2 reference values from the
corrected-basis anchors exactly; sign-test/McNemar suite on toy paired
matrices green; Jeffreys posterior printed as reporting-only per rev1 §4.2.

D7 **(rev1, review B) Verdict-config freeze & D4 demotion rule** — two
  pre-commitments against in-sample reward-shaping:
  1. The informed-class list + patience-rule parameters form a versioned
     `steer_verdict_config.json`; committed with sha BEFORE any C-cell
     launches; its sha is recorded in every C-cell run record (resume
     config-match includes it). Post-freeze edits are a labeled-cell
     decision, never silent — the §8 razor applied to verdict rules too.
  2. D4's outcome is a decision, not a decoration: if live mid-decode
     informed-class yield on real canvases tracks the sim's recall curve
     (0–1/36 at ≤85% committed), the kill/clone cells demote in order
     (C-kill dropped first, C-full's mid-decode rungs reduced in
     checkpoint density), and the honest summary demotes to "ladder +
     final screen" — priced almost for free by the C-screen computation.
     Acceptance: freeze-config schema + tamper-detection test; the
     demotion rule text committed before P0.5.

---

## 8. Explicit non-goals & guardrails (binding on the implementer)

- NO fine-tuning, RL, learned rewards, or logits-per-token reward proxies.
- NO steering signal may read MultiPL-E `tests`, container outputs, or any
  scoring fixture; container verdicts arrive AFTER games end (generation
  with Rancher closed — the M2.3 guardrail stays; scoring only after ps-
  verified model exit).
- NO changes to `export.py`/`to_body_fragment`/scorer semantics; steered
  rows flow the same instrument; the regression test suite (126 tests)
  stays green, + the new D-tests.
- Fixed-shape sampler stays active (kill-switch retained); batch>1 stays
  forbidden (MPS NaN evidence); one resident model per process; the
  memory/generation-scoring guardrails from M2.2 Incidents are binding.
- The dream native path (`TEMPER_NATIVE_DREAM_SAMPLE=1`) remains available
  for regression contrast.
- Single-writer repo lane; phase windows commit at every boundary
  (project rule).

## 9. Cost accounting (honest GPU/wall, this host, s256-class setups)

| item | GPU wall | notes |
|---|---|---|
| D1–D7 instruments | 0 (CPU-only) | ~0.5–1 day of work |
| P0.5 C-hint cell (rev1) | ~1.2 h | 4 problems × K4 × 4.4 min — cheapest falsifier, runs FIRST |
| P1 probe (rev1-deepened) | ~8.5 h | 9 problems × 3 temps × K4 × 4.4 min + margin |
| P1 NO-GO s512 recheck (contingent) | ~3 h | failing cohort only |
| P2 equivalence (s128 × 4, two runs) | ~0.5 h | bit-identity spot-checks |
| C-null@2048 | 14.9 h | FIRST in P3 (feeds attribution gate + C-screen) |
| C-screen computation (rev1) | ~0 | container-minutes from C-null sidecars |
| C-full@2048 | 14.9 h | the primary mechanism row |
| C-full@1024 | 7.5 h | cut order last of the mandatory cells |
| C-kill@2048 (contingent) | 14.9 h | only if primary delta ≥ +2 and contested |
| C-prop rerun (M3.1) | 14.9 h | property-grade final |
| container scoring (all cells) | ~45 min total | cells are minutes each |
| **Phase-1 rev1 base path** | **≈ 48 h GPU + scoring** | C-hint → probe → C-null → C-full@2048 → C-full@1024 (cuttable) |

Full matrix incl. ablations ≈ 68–78 h; the base path first, extras cut by
the gates. If P0.5 + P1 jointly say the quartet is hint-free AND
unreachable, saved: ~36 h and the §5 pivot text is already written.

---

## 10. Appendix: measured facts in this doc + reproducibility

Fresh CPU-only forensics (committed artifacts; commands sketched for the
implementer's scripts — not new repo code):

1. **Failure decomposition, corrected s256/s512** — per-row local rustc
   (host 1.99.1, `--edition=2021`, prompt+completion+`tests` joined):
   s256: 12×E0277 (0/20/21), 4×E0282 (4), 4×E0507 (12) = 20 type-level;
   4×unexpected-closing (6), 4×unexpected-closing (10), 4×mismatched-
   closing (9), 4×expected-ident-found-`=` (25) = 16 parse-level; 12
   Exception rows compile-and-panic locally (181/11/24 panic replicated).
   s512: dead rows = 0/20/21 E0277 ×4 seeds + 4 E0282, nothing else.
   Every failing problem failed identically in all 4 seeds.
2. **Verifier rung latency**: stable rustc 1.99.1
   `rustc --edition=2021 --crate-type lib --emit=metadata` on
   humaneval-sized programs: 30–40 ms (spawn-dominated; constant across
   clean/error/parse-error programs); `-Zparse-only` is nightly-only —
   REJECTED. M0 cargo cycle 236 ms seatbelt; proptest 256-case
   1.75–5.9 s (M1).
3. **Ghost-closure pilot** (uniform word-hole simulation on the 92
   corrected s256 rows, nested reveal sets; §3.2 tables): any-error verdicts
   false-flag 32–42/44 OK rows during holed commitments; informed-class
   filter false-kill 0–2/44 with true-flag 0–1/36 at ≤85% committed,
   5/36 at ~92.5%, 11/36 at ~96.4%, 20/36 at full commit; at zero holes:
   any-error = 36/36 detection with 0/44 false.
   **R1 timing over the same 552-program batch: p50 31.4 ms,
   p95 37.2 ms, max 40.4 ms** (host rustc 1.99.1, spawn-dominated). Caveat
   printed in-doc: sim order is uniform-random; real commit order is
   confidence-ranked (evidence-forward), D4 re-measures live.
4. **Commit math**: remaining masks ≈ n·(1−k/S) (product form under
   `linspace` schedule + floor); commit stickiness verified in the vendored
   `_sample` source; hook return-value replacement verified in both
   native and fixed-shape loops.
5. **Gate arithmetic**: Jeffreys Beta(0.5+x, 0.5+25−x); x=13→P(θ>0.44)=
   0.79; x=14→0.886; x=16→0.978; x=18→P(θ>0.64)=0.795. Computation with
   `temper/.venv` numpy (2×10⁶ draws, seeded).

Banked/committed anchors used: corrected-basis exports/results
(`bench/multipl-e/exports/20261021T14*fix-*`,
`results/20261021T14*.json`), run records + raw sidecars s96–s512 + B3
(`bench/multipl-e/runs/*.jsonl(.raw)`), M0 cycle numbers
(`temper/docs/m0-report.md`), M1.5 ringbuf raws, M2.2 forensics.

Primary sources (decision doc §appendix, re-verified there): MultiPL-E
(Cassano et al.); UMF (arXiv 2602.04344; NFE regimes 768–24,576); Prism
(arXiv 2602.01842; SVF/self-verify, 2,480 NFE); Monkeys (arXiv 2407.21787;
coverage-vs-selection); Orlanski (arXiv 2506.10056; verifier wall matters);
Saber (arXiv 2510.18165; decode-floor comparator); Dream-Coder
(arXiv 2509.01142; Qwen2.5-Coder-7B init, 322B adapt tokens);
"Don't Pass@k" (Bayesian CIs, decision-doc §3 protocol); Yona et al.
(NAACL 2025 skew control).

---

### TL;DR for the reviewer

- **Substrate**: temper-owned single-canvas-per-forward denoise loop
  (generalizing the fixed-shape sampler; bit-identity-gated) driving a
  budget-first sequential SMC — checkpoints at {25,50,75,100}% of steps,
  kills ONLY on positively-diagnosed fatal evidence, clones re-decode
  inherited healthy prefixes with a +0.2 temp step, hard NFE envelope.
- **Reward ladder**: R0 free shape (fence/prose/mask-share/stub) →
  R1 todo!()-ghost-closure + stable-rustc lib type-check (~35 ms; tested:
  exact at completion, informed-class-only mid-canvas; naive any-error
  mid-canvas kills ~80–95% of healthy canvases — measured, hence the
  informed-class patience design) → R3 approved proptest properties
  (bench-now, humaneval via docstring-only GLM-5.3-drafted + human-approved
  in M3.1). Never the vendored tests. Structural allowlist guard.
- **rev1: capability claims foreclosed; mechanism claim pre-registered.**
  C-full@2048 ship = ≥15/25 paired (sign-test primary, Jeffreys demoted),
  18/25 strong; attribution = C-full > max(C-null, C-screen, C-hint) by
  ≥+2 with ≥1 flip outside the hint/ladder cohorts, flip-set replication;
  C-hint (1.2 h) and the deepened probe (8.5 h, cohort-stratified GO) run
  BEFORE any 15-hour cell. Honest P(ship within 2 phases) ≈ 0.35–0.5
  (rev1-revised, harder gates), riding on: *Dream's distribution retains
  the failing problems' correct modes at reachable diversity — the
  same-init AR twin passing 22/25 of them is the reason to believe it,
  and the probe is the reason to know it before committing.*

---

## 11. Appendix (rev1): falsifiable predictions on record

Both reviewers registered predictions BEFORE any M3 cell was generated.
Whoever is wrong, the record shows the design made them cheap to be wrong:

- **review A (Kimi-K3)**: every C-full flip will be hint-reproducible;
  C-full − C-null ≤ +1 problem; the hint cell passes all four type-trap
  problems at ~zero verifier NFE. If instead C-full beats max(C-null,
  C-hint) by ≥ +2 with ≥ 1 flip outside the hint-fixable cohort, review A
  concedes the mechanism is alive.
- **review B (GLM-5.3)**: P1 GO at 4–6/9 driven by the shape pair with
  ≤1–2 type-quartet; C-null@2048 ≈ 12–16/25 (median 14); C-full@2048
  15–18/25 (ship met, strong missed, P≈0.75); **C-full − C-screen ∈
  {0, +1}** — 60–75% of the shipped margin over anchor is ladder +
  rustc filter. If C-full − C-screen ≥ +3, review B concedes the SMC
  machinery survives in substance.
- **design self-prediction (rev0 §6.1 basis)**: C-null@2048 ≈ 13–16/25
  (the realistic-null EV now modeled, not assumed); ship conditional on
  cohort-stratified P1-GO ≈ 0.35–0.5 under the rev2 gates.
