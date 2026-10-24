# temper, explained for a physics undergraduate

You know how experiment works: hypothesis, apparatus, calibration, null run,
statistics, replication — and the part nobody puts on the poster, which is
finding out your instrument was lying to you. This is a write-up of a
computational-physics-style investigation into an idea from social media, run
the way you'd run a lab program. Background assumed: comfort with order-of-
magnitude reasoning, error bars, and being wrong. No machine-learning
background required.

## 1. The phenomenon and the question

A **language model** is a probability distribution over text, conditioned on
a prompt. "Generating code with an LLM" means drawing samples from that
distribution and hoping one of them is a program that works. One sample of the
function's code ≈ one trial; if any of your attempts works, the problem is
"solved."

A **masked diffusion LM** builds its sample differently from the GPT-style
models you've heard of. An autoregressive model writes left-to-right, like
handwriting — once a letter is inked, it's permanent. A diffusion model starts
with the whole canvas *masked* (every position a wildcard) and iteratively
*denoises*: each pass, the model looks at the whole canvas — prompt on one
side, masked slots on the other — and commits to more of the content, possibly
changing its mind about earlier guesses with later context. More passes =
more compute = a more considered answer. The count of model evaluations is
our **NFE** (number of function evaluations — one forward pass per denoise
step). NFE is the *budget*, like lumens or beam time: every comparison we
make is at fixed NFE.

The idea under test — proposed in a toot — was:

> Let the diffusion model write Rust code, and *while the canvas is still
> forming*, write property-based tests for the function it's writing; run
> them with the real Rust toolchain; use the results to steer the rest of the
> decode. Search over codes and inputs jointly, at inference time.

In physics terms: it's a proposal that a **cheap feedback channel into a
stochastic sampler** beats simply taking more samples and cutting them after
the fact — the difference between steering a beam through the experiment
versus spraying and putting a detector at the end. The whole program is the
measurement of that claim.

## 2. The apparatus

- **The sampler**: Dream-Coder-7B, an open-weights masked diffusion model
  (the "dLLM"). Plus its **control twin**: Qwen2.5-Coder-7B, an ordinary
  autoregressive model, run through the *same* harness with the same prompts
  and seeds. If a result depends on diffusion-ness, the twin tells you.
- **The experiment**: 25 small programming problems (HumanEval-for-Rust).
  Each problem shows the model the function's docstring and signature and
  asks it to complete the body, in one fenced code block, in one canvas.
- **The detector chain** (all of it real, all of it instrumented and
  timed): `rustc` type-checks (~35 ms) → a *container* we run the official
  tests in (like a glovebox: the code we generate never touches the host
  machine) → optionally **proptest**, a property-based testing tool.

Property-based testing deserves thirty seconds because it's the input-space
arm of the proposed search, and it's beautiful. Instead of writing
`assert(add(2,3)==5)`, you write the *property* ("adding an element to a
vector then computing the max never decreases the max") and the tool
generates hundreds of random inputs trying to break it. And when it breaks,
proptest **shrinks** the counterexample: it minimizes the failing input, like
finding the smallest reproducible configuration of a flaky instrument —
`minimize_failure()` is a real step in the debugging tradition.

- **Honest scoring discipline** (this cost us real work — see §6): a sample
  that can't be extracted into runnable code counts as a **failure**, not as
  "excluded from the dataset." No cherry-picking by downstream success. Every
  pass rate ("pass@1") you see below is with everything counted. And the
  official verdict only ever comes from the container: any *internal*
  checks we used to steer or select were never allowed to declare a problem
  "solved".

## 3. Methods you'll recognize

- **Null controls at every fork.** The pure sampling ladder (C-null); the
  ladder plus a handed-in hint with the exact idiom the model was misusing
  (C-hint); a purely offline emulation of the selection rule (C-screen).
  You can't claim your feedback loop did it unless the loop-less versions
  do worse — and by the measured amounts.
- **Paired runs on shared random seeds.** Same seeds, same temperatures,
  one changed variable. When we varied temperature across cells, the seed
  structure was pinned and *recorded in every run record* — the analog of
  a fixed beam configuration.
- **Conservative statistics, stated weakness.** n=25 problems, so the
  primary test was a one-sided sign test *per problem*, never a pooled
  average over problems — problems are as non-exchangeable as nights at
  the telescope. Pooled posteriors (Jeffreys) were computed and printed, in
  every table, as "reporting-only." Power statements appear on every gate:
  at n=25 you need five all-in-the-same-direction discordant problems to
  clear α=0.05; a +3 with a possible −1 simply can't, and we say so where a
  physicist would.
- **Pre-registration with a dated amendments log.** Cells, gates, ranking
  rules, and thresholds were frozen in design documents *before* generation.
  When a frozen threshold turned out to be structurally unreachable (below),
  it was amended with a dated entry citing the measurement that forced the
  change — never silently.
- **Fixed budget accounting.** Every cell reports NFE and wall-clock. Every
  claim of "better" is better *per NFE*.

## 4. What we found

**(a) The budget curve is a staircase, and it doesn't plateau where we
could reach.** Honest pass@1: 0.16 → 0.20 → 0.44 → 0.64 as NFE/problem went
384 → 512 → 1024 → 2048. Going 96→128 denoising steps moved *nothing*;
128→256 jumped 4×. That threshold shape killed both easy stories ("fixed
capability floor" — no; "linear in compute" — no).

**(b) The most frequent discovery was *bugs in our own detector* — seven of
them, all caught before they could fake a result.** Three examples: a
token-stripping bug in our export code that amputated the ends of perfectly
balanced function bodies and turned them into guaranteed syntax errors
(found by building the AR control row, which suddenly passed 23/25 and made
us look); a missing `no_grad()` that made the steering engine select a
*different MPS kernel path* than the plain sampler — same math on paper,
different floating-point results — caught only because we ran a bit-identity
gate asking the stripped engine to reproduce the sampler *byte for byte*;
and a tokenizer with 397 word-ids that have no text representation, which
crashes text decoding and cannot be filtered by "skip special tokens"
because they aren't special — it just happened to only bite one game in
twenty. Each of these would have quietly rewritten any conclusion it touched.
The lesson is not "LLM people are sloppy"; it's the lab lesson: **when the
detector is also software you wrote, the null run must be bit-identical to
something you already trust.**

**(c) Steering the decode with the toolchain: null at matched compute.**
The full apparatus — sample 8 candidate canvases per problem with a designed
temperature ladder, check them with `rustc` as they form, let the engine act
on that feedback, pick the best — produced the program's *best cell*, 17/25
versus 14/25 for the unsteered ladder. Looks like a win! But the
pre-registered attribution apparatus ate it: paired per-problem, the +3 was
not significant (p=0.125, and no honest n=25 test could make +3 significant);
the engine's picks differed from the offline rule's only by *drawing
different canvases*, not by ranking the same ones better; and the single
decisive flip that our causal story needed (a problem only the engine had
ever solved) **failed to replicate on fresh seeds**. Verdict, in the
pre-registered gate's exact words: capability SHIP, mechanism FAIL. The
finding stands: steering mid-write didn't beat sampling more and filtering
after.

(And a nice sub-null: when we *handed the model the exact code idiom it kept
failing to use, pasted in the prompt*, it used the idiom — and still failed
the tests, 4/4. The hard problems aren't missing knowledge; the model is
mode-locked.)

**(d) The generator itself has walls, mapped with two independent
instruments.** Four problems were never solved — not at any temperature
(T=0.4 gave literally 0/36 passes; the live band starts at T≥0.8 — a cliff,
not a slope), not at any budget, not by any cell, including the steering
engine's 17/25. An independent month-later scan of every canvas we'd ever
banked agreed: zero correct solutions exist for those four in the entire
corpus. Two instruments, same wall — as satisfying as two mass
spectrometers agreeing.

**(e) The property-testing arm is where it gets surprising.** The proposal
needs the model to *write its own proptest harnesses* at inference time.
Measured: **0 %** of the diffusion model's harness drafts compiled against
the target's signature — and it wasn't close (unbalanced delimiters,
hallucinated modules), reproducible across seeds. The strong autoregressive
twin? **16 %.** Even worse, one of our carefully engineered prompt
improvements turned out to *crash the model itself* — a specific skeleton
in the prompt collapsed the model's output distribution to an all-zero
logit vector from the third token on (a hardware/software decoded-value
degeneration, deterministic, seed-invariant, still unexplained in this
report; the forensic note is committed). So: **at this model scale, nobody
in the zoo can write the detectors.**

**(f) But the detectors themselves — hand-written — are oracle-grade.** We
wrote fifty proptest harnesses ourselves, from each problem's docstring
only, froze them, validated that each one accepts a known-correct program
(rejecting a correct canvas was the most interesting failure mode we saw —
twice it happened because the benchmark's own ranking accepted a program
that *violated the problem statement's printed example*), and then used
them to pick among the model's 200 banked candidate canvases, competing
head-to-head with the confidence-based rule at zero additional compute:
**property verdicts agreed with the official container on 64 of 64 judged
canvases. Zero mis-ranks.** Where the harnesses could speak, they never got
a candidate wrong that the benchmark got right, and they recovered the
extra solution the confidence rule fumbled. The selection mechanism is
sound. The final post-hoc diagnosis: with a fixed tie-break rule (never
defer to confidence when multiple candidates pass all properties), the
property rule hits the theoretical maximum for that ensemble, 16/25 —
exactly the oracle.

So the composite, and the sentence that summarizes two milestones:
**the discrimination is perfect where it exists; the supply does not exist
because the generators cannot be authored at this scale; and where supply
exists, the expensive steering loop adds nothing over sampling-plus-
selection at fixed compute.**

## 5. What this teaches about running experiments (the transferable part)

1. **Budget-shaped curves deserve budget-shaped analysis.** Our "budget
   response" was threshold-like; a forces-fitted linear model of it would
   have produced a confident, wrong extrapolation.
2. **The null run is the most valuable measurement in the program** — and
   ours had to be *bit-identical*, because "statistically identical" hides
   a different kernel path. If your control doesn't match your treatment to
   the byte, you don't have a control.
3. **Invent thresholds before data; let the data overrule them loudly.**
   Three of our pre-frozen gates (a 90 % threshold, a 23/25 target, an
   implicit coverage assumption) turned out to be structurally unreachable
   given real instrument limits — one required ground truth that provably
   doesn't exist for four of the twenty-five problems. All three are
   documented in dated amendments while the pre-registered verdicts stand.
   That's the difference between pre-registration and stubbornness.
4. **Replicate before you believe a single flip.** Our one beautiful
   mechanism-consistent success (a problem only the steered engine had
   ever passed) died on the first fresh-seed rerun. We kept the null
   reaction: the 4-replication standard for any single-problem claim.
5. **Count your failures as failures.** The dataset never gets to shrink
   to the working friends. That is the pre-registration + honest-reporting
   combo applied at 4 a.m.
6. **Report per-problem statistics, not pooled averages** — problems are
   not exchangeable, and anyone who pools them is flattening exactly the
   variation that carries the mechanism.

## 6. Where it goes next

The measured open problem is **supply**: harnesses can't be generated at
7B scale; therefore property-selection starves at coverage well before its
measured-discrimination ability becomes the limit. The successor experiment
writes itself from this report: strong/held generator + coverage-first
design + the never-defer-to-confidence tie-break + more budget (the budget
curve was still rising when we stopped).

For you, the reader: the full artifacts (instrument builds, freeze
discipline, per-run play-by-play, all seven bug forensics) are the documents
this one is a tour guide for. The best next skills this investigation
suggests for an experimentalist are boring and eternal: controls, replication,
instrument debugging, and writing down what would count as a null result
before you take the data.

— compiled from the temper program's milestone reports (M0 → M3.1); every
number above is from a committed, reproducible cell.
