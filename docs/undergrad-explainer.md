# temper, explained for a physics undergraduate

You know how experiment works: hypothesis, apparatus, calibration, control
run, statistics, replication — and the part that never makes the poster,
which is finding out your instrument was lying to you. This document explains
an investigation that was run that way, on an idea that started life as a
social-media post. We assumed comfort with order-of-magnitude reasoning and
being wrong; we assumed *no* machine-learning background, and we've tried to
define every term before using it. Tell us where we failed.

## 1. The phenomenon, and the proposed trick

A **language model** is a probability distribution over text. "Writing a
program with a language model" means asking the model to produce the body of
a function (we give it the problem description and the function's required
name and argument types) and then *sampling* — drawing one possible
completion out of the many it considers plausible. Each sample is a
candidate answer, like one roll of a loaded die whose biases come from
training.

There are two common ways for a model to build the text it gives you, and
the difference matters here:

* An **autoregressive** model (the ChatGPT-style kind) writes strictly
  left-to-right, one fragment at a time, and once it has written something it
  cannot go back. Like painting left to right with no revisions allowed.
* A **masked diffusion** model works instead like restoring a damaged
  fresco. It starts with the whole wall covered inplaceholders, then fills
  in *everything simultaneously, roughly*, and then keeps making passes over
  the whole picture — and each pass may *revise* earlier strokes in light of
  later context. The model we used, Dream-Coder-7B, is of this kind, and it
  is small enough to run on a good laptop.

Each pass over the draft costs one run of the neural network. So the total
number of passes is our **computing budget** — the equivalent of beam time.
Throughout this report, "budget" always means this count (25 problems at a
budget of 2048 passes per problem = 51,200 network runs, roughly fourteen and
a half hours on our machine). If we ever claim method A beats method B, the
claim only counts if both spent the same budget.

The idea we set out to test — again, from a social-media post — was:

> Let the diffusion model write a Rust program, and *while the program is
> still being painted*, have it also write checks (tests) for that program;
> run the checks with the real Rust compiler and test tools; and use the
> results to steer the rest of the painting. Search over candidate programs
> and test inputs *jointly*, live, while the compute is being spent.

In lab terms: the proposal hopes that a cheap feedback channel wired into
the live experiment beats "just take more data and cut later" — steering
the beam through the apparatus versus spraying and putting a counter at the
end. That claim is what the whole program measured.

One piece of vocabulary we need up front, because everything below is
measured with it. For each problem, the model gets some number of tries;
we then look at the 25 problems and ask: **for how many of them did at
least one try produce a program that passes the official hidden tests?**
That fraction is the **solve rate**. The problems are small but not
trivial — the sort of thing you could do yourself in ten lines, and the
official tests check actual behavior, not just spelling.

## 2. The apparatus

* **The two models.** Dream-Coder-7B, the "restores-the-fresco" model above.
  And, next to it, a **control twin**: Qwen2.5-Coder-7B, a well-known
  ordinary (left-to-right) model run through the *same* apparatus, same
  prompts, same random seeds. When a result depends on fresco-style
  generation being special, the twin is what tells us.
* **The problems.** 25 short programming puzzles (a Rust-language version
  of the classic HumanEval set). Each shows the model a description ("for a
  list of numbers, return the mean absolute deviation from the mean") and a
  required function shape; the model must write the body.
* **The detector chain** — all of it real software, all of it timed and
  logged like any detector:
  1. the Rust compiler, doing a quick type-and-syntax check (~35
     milliseconds — the Geiger click of the operation);
  2. a sealed, throwaway environment (a "container" — a sandboxed
     machine-in-a-box) where a candidate program is compiled and run against
     the **official hidden tests**. Grading is *exclusively* by these
     official tests. This matters: we also used our own internal checks for
     steering and selection, but they were never allowed to declare a
     problem "solved" — the benchmark's answer is the only answer, so the
     project could not accidentally grade its own homework;
  3. optionally, **property-based testing** — the input-space arm of the
     proposed trick, and a lovely idea. Instead of checking
     `add(2,3)==5`, you state a *property* ("appending an element and
     taking the maximum can never lower the maximum") and a tool, proptest,
     *generates its own random inputs* trying to break your claim —
     hundreds per second. And when it finds a violation, it **shrinks** it:
     it automatically minimizes the failing input, the way you'd hunt for
     the smallest configuration that makes an instrument glitch.
* **Honest counting.** A candidate that can't be turned into a runnable
  program counts as a **failure** — always. It is never quietly dropped
  from the sample. Every solve rate quoted below includes these failures.
  This single rule caused us real work, and we kept it anyway: sample
  sizes never shrink to just the friends who worked.

## 3. The methods, in the language you already know

* **Control runs at every fork.** (i) The plain bracket: sample several
  drafts per problem, with no feedback, at a deliberately *spread* of
  sampling-adventurousness settings — the photographer's exposure
  bracket: same shot taken safe and dull, again wild, so that something
  lands well in one of them — then pick by a fixed rule. (ii) The hint
  control: as before, but with the *exact missing trick* pasted into the
  instructions. (iii) The offline filter: take the plain bracket's drafts
  and apply the selection rule after the fact, with no live steering at
  all. If the live-steering apparatus is really what helps, it has to
  beat these — by amounts beyond chance.
* **Fixed random seeds and paired runs.** A random seed is a starting
  number that makes a given random run exactly reproducible, like the
  machine's dice being re-loaded identically. Whenever we compared two
  methods, both saw the same seeds and the same settings, and differed in
  exactly one deliberate way — the same-beam discipline. The seed
  configuration is written into every run record.
* **Conservative win statistics, with the weakness stated.** Our sample is
  25 problems, and problems are as non-exchangeable as nights on a
  telescope, so we never averaged results across problems as if they were
  identical. The primary test is the simplest one available: for each
  problem, record which of the two methods succeeded; if the methods were
  truly equal, the wins would split like a fair coin. Our rule — set in
  advance — was that a real effect must be unlikely for a fair coin: chance
  of one-in-twenty or less. A quick look at how big a win that demands at
  n=25 tells you a **5-to-0 sweep** is the smallest provable result. A
  3-to-0 tilt is *suggestive* but a coin produces that more than one time
  in ten, so a 3-to-0 must be reported as "not provable at this sample
  size," and below, we report it exactly that way. (We computed the
  pooled-average statistics too and printed them, but nothing was ever
  decided by them.)
* **Pre-registration, with a dated amendment log.** Cells, thresholds,
  ranking rules, and replication requirements were frozen in design
  documents *before* any data were taken. When a frozen threshold turned
  out to be structurally unreachable — defined, in one case, by ground
  truth that provably doesn't exist for four of our 25 problems — it was
  amended with a dated entry quoting the measurement that forced the
  change. Never silently, never after the fact.
* **Fixed budget accounting.** Every table below compares methods at equal
  budget. "Better" always means better per network-run.

## 4. What we found

**(a) The budget curve is a staircase — and it was still climbing when we
ran out.** We held everything fixed and gave each problem between 384 and
2048 network passes. The solve rate went **16% → 20% → 44% → 64%**. The
shape matters more than the growth: adding passes at the small budget did
*nothing at all* — flat — and then the next jump *quadrupled* the success
rate. That staircase shape refutes the two easy stories a person might
tell. Story one, "the model has a fixed talent" — wrong, because talent
clearly grows with budget. Story two, "double the compute, get double the
quality" — wrong too, because the response is a threshold: spend below the
threshold and you buy almost nothing; spend above it and it pays. And at
our maximum budget the curve was still heading upward — we never found the
top.

**(b) The most frequent discovery was our own equipment misbehaving — seven
times, all caught before a result could be faked.** Every measurement
apparatus lies; ours lied too, and this is the part of the program we are
proudest of. Three examples:
1. **A checker that broke good programs.** Our export code stripped what it
   thought were junk trailing tokens, and in doing so chopped the closing
   halves off perfectly well-formed programs — converting them into
   guaranteed syntax errors and dragging our solve rates down by
   two-thirds. We found it because the control twin — which has no
   cleanup-style generation to confuse us — suddenly started passing
   almost everything, which made us look.
2. **A one-word programming omission that changed the arithmetic.** A
   missing "don't track gradients" flag made the same mathematical
   operation run through a *different internal algorithm* in the GPU
   library, and the answers started differing in the last few decimal
   places — which, multiplied through a stochastic sampling process,
   changes which candidates you get entirely. We caught it with a
   **bit-identity gate**: we demanded that our steering engine, with all
   its feedback machinery switched off, reproduce the plain sampler
   *identically, byte for byte* — the computational version of demanding
   your control channel match the reference channel on every tick. It
   failed 0/4 at first; after the fix it passed 4/4. Every "the steering
   didn't help" claim below stands on that gate being green afterwards.
3. **A dictionary with 397 entries that mean nothing.** The model's
   vocabulary contains a band of 397 "words" that have no written form at
   all. A sampling process is free to draw them; the text decoder then
   crashes; and no "skip special characters" option can protect you,
   because as far as the software is concerned they aren't special. This
   killed one long run and is now handled, but it was pure luck it killed
   only one.

The general lesson is not about software. It is: **when the detector is
also something you built, your null channel has to match the reference
channel to the bit.** "Statistically identical" hides a different
algorithm.

**(c) Steering the writing with live toolchain feedback: no detectable
win, per the pre-registered gates.** The full proposal — sample eight
drafts per problem in an adventure-spread bracket (§3), check them with
the compiler as they form, let an engine act on that
feedback, then pick the best draft — produced the program's best single
result: **17 of 25 problems solved, versus 14 of 25 for the plain sampler**,
with no problem the plain sampler solved that the engine lost. Looks like
the trick works! The pre-registered analysis then dismantled the happy
story three ways:
1. *3 problems net gain over 25 is not provable at our sample size* — a
   coin produces a 3-0 or better tilt more than one time in ten.
2. *The engine's advantage evaporated at fixed drafts.* When we ran the
   selection rule offline, on exactly the same drafts the engine had seen,
   it matched the engine. The engine had drawn *different* drafts, and
   drawn slightly luckier — it fixed nothing about the *judging*.
3. *The one compelling data point failed replication.* A single problem
   that had *never* been solved by any other method got solved by the
   steered engine — exactly the kind of flip a causal story dreams of. We
   re-ran it with fresh random seeds, per the rule set in advance. It
   failed. One flip is one flip; the mechanism claim was dead with it.

So the verdict, in exactly the words of the pre-frozen gate: the
**capability is real** (17/25 is our best cell) but the **mechanism is
unproven** — and, more precisely, live steering is *indistinguishable* from
sampling more and selecting after, at equal budget.

(A delicious sub-result on the way: when we pasted the *exact missing
trick* into the instructions — literally handing over the idiom the model
kept failing to use — the model used the trick, *verbatim*, and still
failed the tests, four times out of four. Like a student who copies the key
formula into their notes and still can't solve the problem. The hard
problems were never about missing knowledge.)

**(d) The model has walls, and two instruments agree on where they are.**
Four of the 25 problems were *never* solved — not at any budget, not by any
method, including the best cell. The attempts aren't randomly failing;
they're failing *structured*. There's a dial ("temperature") controlling
how adventurous the sampling is — low means the model plays it safe and
repeats itself, high means it takes wild swings. At the tame setting, over
thirty-six attempts, exactly zero produced a working program; every solve
lives above a certain wildness threshold — a cliff, not a gradual slope.
And a month later, an independent measurement — scanning every draft we
had ever saved from any experiment — agreed: zero correct solutions exist
for those four problems in the entire corpus, at any cost. Two instruments,
built for different purposes, pointing at the same wall. That's what a
real boundary looks like.

**(e) The surprise: nobody in the model zoo can write the detectors.** The
proposal needs the model to write its own test suites at inference time —
that's the "input-space" arm of the joint search. Measured: the diffusion
model's test drafts *never compiled* against the target function — 0 of 12
sampled test files, and not near misses: unbalanced brackets, invented
modules. The strong control twin did better in spirit but still only
**16% of its test drafts compiled**. Both numbers are reproducible across
seeds — a talent wall, not bad luck. Worse, one of our carefully engineered
improvement attempts *broke the model itself*: a specific worked example
we added to the instructions made the twin's output scores collapse to an
exact zero vector from the third word on — deterministic, seed-invariant
hardware-plus-software degeneration, still unexplained (the forensic note
is committed). So at this model size — billions of parameters, running on
one laptop — **the proposed loop cannot be assembled, because its test
writer does not exist.**

**(f) Hand-written tests, though, judge perfectly.** We wrote the fifty
test files ourselves — from each problem's text alone, committed to the
record *before* ever being run against the model's drafts, so the commit
order itself proves we wrote to the problem statement and not to the drafts.
Each test was validated twice: it must accept at least one program known to
be correct, and — the interesting failure mode — twice our tests initially
*rejected* programs the benchmark accepted, and on inspection the benchmark
was wrong: one program computed a result that violated the problem's own
worked example. (The benchmark passes it anyway; file that one.) Then we
used the tests to pick among the 200 stored drafts of the plain sampler,
head-to-head with the previous selection rule — which ranks drafts by the
model's own confidence — at zero additional model cost. Result: **on the 64
drafts where our tests could give a verdict, they agreed with the official
benchmark 64 times out of 64**. Zero misjudgments. They recovered a
solution the confidence rule had fumbled and dropped it as a bonus. And one
small rule fix we diagnosed after the fact — when several drafts pass all
tests, break the tie by preferring drafts that reconstruct cleanly, never
by model confidence — takes the solve rate to **16 of 25: exactly the
"perfect hindsight" maximum** for that pile of drafts. (That fix is
labeled "after the fact" in the report and never counted as a result —
you only get to claim it by running it fresh.)

So the composite, in one sentence: **the judging machinery works
perfectly where it exists; the test files cannot be machine-written at this
scale; and where tests do exist, the expensive live-steering loop buys
nothing that sample-then-select doesn't buy at the same budget.**

## 5. What this teaches about doing experiments (the transferable part)

1. **Budget-shaped curves deserve budget-shaped analysis.** Ours was a
   staircase; a line fitted through it would have produced a confident,
   wrong extrapolation in either direction.
2. **The control run is the most valuable measurement in the program** —
   and ours had to be bit-for-bit, because "statistically identical" hides
   a different algorithm inside the box.
3. **Set thresholds before the data; let the data overrule them out loud.**
   Three of our pre-frozen gates turned out to be structurally unreachable
   (one depended on ground-truth solutions that provably don't exist for
   four of the twenty-five problems). All three are documented as dated
   amendments, and the pre-registered verdicts stand. Pre-registration is
   honesty, not stubbornness — and the difference is the dated entry.
4. **Replicate before believing a single flip.** Our one beautiful,
   mechanism-consistent success died on the first fresh-seed re-run. The
   write-up kept the corpse, and the standards for single-problem claims
   got stricter because of it.
5. **Count your failures as failures.** The sample never gets to shrink
   to the cases that worked. That's the difference between a solve rate
   and a press release.
6. **Never average away your variation.** Problems are not exchangeable;
   anyone who pools them is flattening exactly the structure that carries
   the mechanism. Per-problem wins and losses, always.

## 6. Where it goes next

The measured open problem is **supply**: at the model sizes we could run,
the test files can't be generated, so the (perfect) judging machinery
starves on coverage long before its judgment becomes the limit. The next
experimenter's shopping list, straight from these measurements: a test-file
generator that actually compiles, a design that puts coverage first, the
never-defer-to-confidence tie-break, and a bigger budget — the budget curve
was still climbing when we stopped.

For you, the reader: the full artifacts — instrument builds, freeze
records, per-run play books, and all seven bug hunts — are in the
repository's milestone reports; every number above is from a committed,
reproducible run. The skills this investigation exercised most are boring
and eternal: controls, replication, instrument debugging, and writing down
what would count as a null result *before* you take the data.
