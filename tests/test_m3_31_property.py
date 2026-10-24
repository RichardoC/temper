"""M3.1 PHASE 0 pins: R3 (the property rung), emission v3, and the loop wiring.

The property rung is the first rung in the program whose evidence comes from
EXECUTION, and the whole primary cell turns on it — so what is pinned here is
exactly what would silently turn P-prop-screen into a different experiment:

  * the ranking rule, including the undefined-pass_rate tiers and the
    degeneration to the screen rule when there is no evidence (design §2.3 +
    §7 Q1);
  * the abstention classes: timeout / abort / error / compile_error are NOT
    failures (§2.3), and R3's pass_rate is defined over measured calls only;
  * the loop wiring: R3 fires at the FINAL window only, once per R1-clean
    canvas, never before it, never on a dirty canvas, never twice;
  * R3 cannot kill, clone or move budget — a regression pin on the demoted
    role (design §2.3's "kill/clone remain OFF" and the §0 guardrail);
  * the pub normalization R3 needs to link at all (measured: the unmodified
    ghost program gives E0603), and that it changes nothing but visibility;
  * the freeze manifest's tamper detection (D7's second sha).

The cargo-executing pins are skipped when no rustc/cargo is on PATH; the
pure-logic pins always run.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from temper.steer.emission import (
    EMISSION_PROPERTY,
    EMISSION_SCREEN,
    property_rank_key,
    property_tier,
    rank_by_rule,
    rank_property,
    rank_screen,
    select_by_rule,
    select_property,
    select_screen,
)
from temper.steer.property import (
    ABSTAIN_STATUSES,
    HARNESS_FREEZE_SCHEMA,
    MEASURED_STATUSES,
    R3Call,
    R3Verdict,
    compute_harness_freeze_sha256,
    load_harness_freeze,
    make_pub_program,
    run_r3,
    strategy_hash_of,
)
from temper.steer.verdict import load_verdict_config
from temper.steer.view import SteerProblemView


def _have_cargo() -> bool:
    from temper.harness.runner import find_cargo
    try:
        find_cargo()
        return True
    except Exception:
        return False


@dataclass
class _Cand:
    """A canvas as the emission rules see it (v3 adds the property evidence)."""

    index: int
    temperature: float
    final_clean: bool
    mean_commit_conf: Optional[float]
    property_pass_rate: Optional[float] = None


# --------------------------------------------------------------------------
# the ranking rule
# --------------------------------------------------------------------------


class TestPropertyRanking(unittest.TestCase):
    """``property_pass_rate_then_mean_commit_conf_ties_lowest_temp`` (§2.3)."""

    def test_property_evidence_outranks_confidence(self):
        """THE primary claim: a proven-passing canvas beats a higher-confidence
        one. If this ever inverts, P-prop-screen measures nothing."""
        cands = [_Cand(0, 0.2, True, 0.95, 1.0),
                 _Cand(1, 1.0, True, 0.50, 1.0)]
        self.assertEqual(select_property(cands).index, 0)
        cands = [_Cand(0, 0.2, True, 0.20, 1.0),
                 _Cand(1, 1.0, True, 0.90, 0.0)]
        self.assertEqual(select_property(cands).index, 0,
                         "a measured-passing canvas must beat a measured-failing "
                         "one however confident the failing one looks")

    def test_higher_pass_rate_wins_within_the_measured_tier(self):
        cands = [_Cand(0, 0.2, True, 0.9, 1.0), _Cand(1, 0.8, True, 0.1, 0.5)]
        self.assertEqual(select_property(cands).index, 0)

    def test_a_canvas_with_no_evidence_does_not_beat_a_failing_one(self):
        """§2.3's asymmetry, in the direction that matters for the gate: the
        abstention tier sits BELOW proven-good but ABOVE proven-bad."""
        good = _Cand(0, 0.2, True, 0.1, 1.0)
        unknown = _Cand(1, 0.8, True, 0.9, None)
        bad = _Cand(2, 1.0, True, 0.95, 0.0)
        self.assertEqual([c.index for c in rank_property([bad, unknown, good])],
                         [0, 1, 2])

    def test_all_abstained_pool_degenerates_to_the_screen_rule(self):
        """With no property evidence anywhere, v3 MUST equal v2 exactly — that
        identity is what makes the primary a pure rule delta."""
        cands = [_Cand(0, 0.2, True, 0.30), _Cand(3, 1.1, True, 0.80),
                 _Cand(5, 0.8, True, 0.55)]
        self.assertEqual(select_property(cands).index,
                         select_screen(cands).index)
        self.assertEqual([c.index for c in rank_property(cands)],
                         [c.index for c in rank_screen(cands)])

    def test_no_clean_canvas_pool_still_uses_property_evidence(self):
        """The §3.3 item 2 fallback pool is the whole ensemble; the ranking still
        reads property evidence there (the all-dirty case P-prop-screen sees on
        the problems with zero R1-clean canvases)."""
        cands = [_Cand(0, 0.2, False, 0.9, 0.0), _Cand(1, 1.0, False, 0.1, 1.0)]
        self.assertEqual(select_property(cands).index, 1)

    def test_tie_order_is_the_screens(self):
        """Inside a tier: confidence desc, then temp asc, then index asc (§2.3
        keeps the screen's existing tie order)."""
        cands = [_Cand(6, 0.8, True, 0.55, 1.0), _Cand(2, 0.8, True, 0.55, 1.0),
                 _Cand(9, 1.2, True, 0.55, 1.0)]
        self.assertEqual([c.index for c in rank_property(cands)], [2, 6, 9])

    def test_tiers_are_the_three_pre_registered_ones(self):
        self.assertEqual(property_tier(_Cand(0, 0.2, True, 0.1, 1.0)), 0)
        self.assertEqual(property_tier(_Cand(0, 0.2, True, 0.1, None)), 1)
        self.assertEqual(property_tier(_Cand(0, 0.2, True, 0.1, 0.0)), 2)
        # a canvas the rung never touched is an abstention, not a failure
        class _Untouched:
            index, temperature, final_clean, mean_commit_conf = 0, 0.2, True, 0.5
        self.assertEqual(property_tier(_Untouched()), 1)

    def test_rank_key_orders_exactly_as_the_tiers_document(self):
        cands = [_Cand(0, 0.2, True, 0.9, 0.0), _Cand(1, 0.2, True, 0.5, None),
                 _Cand(2, 0.2, True, 0.1, 1.0)]
        self.assertEqual([c.index for c in sorted(cands, key=property_rank_key)],
                         [2, 1, 0])

    def test_empty_ensemble_selects_nothing(self):
        self.assertIsNone(select_property([]))

    def test_rule_dispatch_refuses_an_unknown_rule(self):
        """A cell may not run an emission rule the code does not implement, and
        a record cannot name one that did not run."""
        self.assertEqual(EMISSION_PROPERTY,
                         "property_pass_rate_then_mean_commit_conf_ties_lowest_temp")
        self.assertEqual(select_by_rule([_Cand(0, 0.2, True, 0.1, 1.0)],
                                        EMISSION_PROPERTY).index, 0)
        self.assertEqual(select_by_rule([_Cand(0, 0.2, True, 0.1, 1.0)],
                                        EMISSION_SCREEN).index, 0)
        with self.assertRaises(ValueError):
            select_by_rule([], "best_whatever")
        with self.assertRaises(ValueError):
            rank_by_rule([], "best_whatever")

    def test_the_config_names_the_rule_the_code_implements(self):
        self.assertEqual(load_verdict_config().emission, EMISSION_PROPERTY)

    def test_the_ladder_temperature_0_2_canvas_is_not_privileged(self):
        """v3 has no structural-temperature preference: the property rule must
        be able to PASS OVER the T=0.2 canvas, which is exactly the Power it
        needs on HumanEval_1 (the screen's measured miss)."""
        cands = [_Cand(0, 0.2, True, 0.10, 0.0), _Cand(1, 0.8, True, 0.05, 1.0)]
        self.assertEqual(select_property(cands).index, 1)


# --------------------------------------------------------------------------
# the verdict object + abstention semantics
# --------------------------------------------------------------------------


def _call(status: str, **kw) -> R3Call:
    return R3Call(problem="p", harness_sha="h", canvas="p0", status=status,
                  duration_s=0.1, **kw)


class TestR3VerdictSemantics(unittest.TestCase):
    def test_measured_statuses_are_pass_and_fail_only(self):
        self.assertEqual(MEASURED_STATUSES, ("pass", "fail"))
        self.assertEqual(ABSTAIN_STATUSES,
                         ("timeout", "abort", "error", "compile_error"))

    def test_timeout_abstains_and_never_counts_as_a_failure(self):
        """§2.3: 'timeout counts as abstention, never as failure'."""
        v = R3Verdict((_call("pass"), _call("timeout")))
        self.assertEqual(v.pass_rate, 1.0,
                         "a timeout must not be divided into the pass rate")
        self.assertEqual(v.n_measured, 1)
        self.assertEqual(v.n_abstained, 1)

    def test_all_timeout_leaves_pass_rate_undefined(self):
        v = R3Verdict((_call("timeout"), _call("timeout")))
        self.assertIsNone(v.pass_rate)
        self.assertTrue(v.all_abstained)
        self.assertEqual(property_tier(_wrap(v)), 1)

    def test_abort_and_harness_error_also_abstain(self):
        """Generalizing §2.3's sentence to every status the protocol can emit: a
        harness that gave up generating, or crashed, did not judge the
        candidate. Only pass/fail are evidence."""
        for status in ("abort", "error", "compile_error"):
            v = R3Verdict((_call("fail"), _call(status)))
            self.assertEqual(v.pass_rate, 0.0,
                             f"{status} must abstain, not add a second failure")
            self.assertEqual(v.n_measured, 1)

    def test_fail_is_evidence(self):
        v = R3Verdict((_call("fail"), _call("fail")))
        self.assertEqual(v.pass_rate, 0.0)
        self.assertFalse(v.all_abstained)
        self.assertEqual(property_tier(_wrap(v)), 2)

    def test_partial_pass_rate_is_the_measured_fraction(self):
        v = R3Verdict((_call("pass"), _call("fail"), _call("timeout")))
        self.assertEqual(v.pass_rate, 0.5)
        self.assertEqual(v.n_calls, 3)

    def test_minimal_inputs_survive_into_the_record(self):
        v = R3Verdict((_call("fail", minimal_input="[1]"), _call("pass")))
        rec = v.as_record()
        self.assertEqual(rec["minimal_inputs"], ["[1]"])
        self.assertEqual(rec["statuses"], ["fail", "pass"])

    def test_no_calls_is_not_all_abstained(self):
        """R3 never ran on this canvas — distinct from 'ran and abstained
        everywhere'. The tier is still 1 (no evidence), but the log distinguishes
        them via r3_fired."""
        v = R3Verdict(())
        self.assertIsNone(v.pass_rate)
        self.assertFalse(v.all_abstained)


def _wrap(v: R3Verdict) -> _Cand:
    return _Cand(0, 0.2, True, 0.5, v.pass_rate)


# --------------------------------------------------------------------------
# the pub normalization (R3 needs it to link at all)
# --------------------------------------------------------------------------


VIEW = SteerProblemView(name="HumanEval_1_separate_paren_groups",
                        prompt=("/// doc\n"
                                "fn separate_paren_groups(paren_string: String) "
                                "-> Vec<String> {\n"),
                        fn_name="separate_paren_groups",
                        signature_line="fn separate_paren_groups(paren_string: String) -> Vec<String> {")

CANVAS = ("```rust\n"
          "/// doc\n"
          "fn separate_paren_groups(paren_string: String) -> Vec<String> {\n"
          "    let mut out: Vec<String> = Vec::new();\n"
          "    out.push(paren_string);\n"
          "    out\n"
          "}\n"
          "```\n")


class TestPubNormalization(unittest.TestCase):
    def test_the_prompted_function_becomes_pub(self):
        program = make_pub_program(VIEW, CANVAS)
        self.assertIn("pub fn separate_paren_groups(", program.text)

    def test_nothing_but_visibility_changes(self):
        from temper.steer.verdict import ghost_close
        before = ghost_close(VIEW, CANVAS).text
        after = make_pub_program(VIEW, CANVAS).text
        self.assertEqual(after.replace("pub fn separate_paren_groups",
                                       "fn separate_paren_groups"), before,
                         "the R3 program must be the R1 program plus "
                         "visibility and nothing else")

    def test_already_pub_is_left_alone(self):
        canvas = CANVAS.replace("fn separate_paren_groups",
                                "pub fn separate_paren_groups")
        program = make_pub_program(VIEW, canvas)
        self.assertEqual(program.text.count("pub fn separate_paren_groups"), 1)

    def test_a_holed_canvas_is_refused(self):
        """R3 requires a completed, zero-hole canvas (§2.2) — a holed program
        would be R1's judgement re-run with execution noise on top."""
        holed = CANVAS.replace("out.push(paren_string);", "<|mask|>")
        with self.assertRaises(ValueError):
            make_pub_program(VIEW, holed)

    def test_ghost_program_metadata_is_preserved(self):
        from temper.steer.verdict import ghost_close
        before = ghost_close(VIEW, CANVAS)
        after = make_pub_program(VIEW, CANVAS)
        self.assertEqual(after.prompt_lines, before.prompt_lines)
        self.assertEqual(after.hole_lines, before.hole_lines)
        self.assertEqual(after.tail_lines, before.tail_lines)
        self.assertEqual(after.fragment_exact, before.fragment_exact)


# --------------------------------------------------------------------------
# the freeze manifest (D7's second sha)
# --------------------------------------------------------------------------


class TestHarnessFreeze(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        self.src = self.tmp / "harnesses"
        self.src.mkdir()
        self.harness_text = "pub fn run_property() {}\n"
        (self.src / "HumanEval_1__h0.rs").write_text(self.harness_text,
                                                     encoding="utf-8")
        self.payload = {
            "schema": HARNESS_FREEZE_SCHEMA,
            "version": "test-v1",
            "problems": [{"name": "HumanEval_1_separate_paren_groups",
                          "harnesses": [{
                              "slot": 0, "seed": 0,
                              "strategy_hash": strategy_hash_of(self.harness_text),
                              "source_path": "harnesses/HumanEval_1__h0.rs",
                              "replaced": False}]}],
        }
        body = dict(self.payload)
        self.manifest = dict(body, sha256=compute_harness_freeze_sha256(body))
        self.path = self.tmp / "harness_freeze.json"
        self.path.write_text(json.dumps(self.manifest), encoding="utf-8")

    def test_a_committed_manifest_loads(self):
        frozen = load_harness_freeze(self.path)
        self.assertEqual(len(frozen["HumanEval_1_separate_paren_groups"]), 1)
        h = frozen["HumanEval_1_separate_paren_groups"][0]
        self.assertEqual(h.strategy_hash, strategy_hash_of(self.harness_text))
        self.assertEqual(h.source, self.harness_text)

    def test_editing_the_manifest_is_detected(self):
        tampered = dict(self.manifest)
        tampered["problems"] = [{"name": "x", "harnesses": []}]
        p = self.tmp / "tampered.json"
        p.write_text(json.dumps(tampered), encoding="utf-8")
        with self.assertRaises(Exception) as ctx:
            load_harness_freeze(p)
        self.assertIn("TAMPERED", str(ctx.exception))

    def test_editing_a_harness_case_file_is_detected(self):
        """The manifest's sha covers the manifest; each harness's own sha must
        cover its TEXT, or a post-freeze edit to a case file would pass."""
        (self.src / "HumanEval_1__h0.rs").write_text("pub fn run_property() {}\n\
// edited after the freeze\n", encoding="utf-8")
        with self.assertRaises(Exception) as ctx:
            load_harness_freeze(self.path)
        self.assertIn("strategy_hash", str(ctx.exception))

    def test_a_missing_manifest_is_loud(self):
        with self.assertRaises(Exception):
            load_harness_freeze(self.tmp / "nope.json")

    def test_unstamped_manifest_is_refused(self):
        p = self.tmp / "unstamped.json"
        p.write_text(json.dumps(self.payload), encoding="utf-8")
        with self.assertRaises(Exception) as ctx:
            load_harness_freeze(p)
        self.assertIn("never frozen", str(ctx.exception))


# --------------------------------------------------------------------------
# cargo-executing pins (R3 end to end)
# --------------------------------------------------------------------------

_PROP_PROPERTIES = """\
use crate::candidate::separate_paren_groups;
use proptest::prelude::*;
use proptest::test_runner::{TestCaseError, TestCaseResult, TestError, TestRunner};

pub const STRATEGY_DESC: &str = "space-joined balanced paren groups";

// Generate only WELL-FORMED input (groups already balanced, joined by spaces),
// because the contract only constrains balanced input: judging an unbalanced
// string would be testing something the documentation never promises.
fn strategy() -> impl Strategy<Value = String> {
    proptest::collection::vec(
        proptest::sample::select(vec!["()", "(())", "()()", "(()())", "(())()"]),
        0..4usize,
    )
    .prop_map(|groups| groups.join(" "))
}

fn property(s: String) -> TestCaseResult {
    let out = separate_paren_groups(s.clone());
    // the contract: each returned group is balanced, and every parenthesis of
    // the input appears in exactly one returned group, in order
    for part in &out {
        let mut depth = 0i64;
        for ch in part.chars() {
            match ch {
                '(' => depth += 1,
                ')' => depth -= 1,
                _ => {}
            }
            if depth < 0 {
                return Err(TestCaseError::fail(format!("negative depth in {:?}", part)));
            }
        }
        if depth != 0 {
            return Err(TestCaseError::fail(format!("unbalanced group {:?}", part)));
        }
    }
    let strip = |t: &str| -> String { t.chars().filter(|c| *c == '(' || *c == ')').collect() };
    if strip(&out.concat()) != strip(&s) {
        return Err(TestCaseError::fail(format!(
            "parentheses not preserved: input {:?} -> output {:?}", s, out)));
    }
    Ok(())
}

pub fn run_property(runner: &mut TestRunner) -> Result<(), TestError<String>> {
    runner.run(&strategy(), property)
}
"""

# A harness that returns Ok unconditionally: it compiles and passes against
# ANY candidate, including the todo!() stub — the "measures nothing" shape G4
# has to be able to see.
_VACUOUS_PROPERTIES = """\
use crate::candidate::separate_paren_groups;
use proptest::prelude::*;
use proptest::test_runner::{TestCaseError, TestCaseResult, TestError, TestRunner};

pub const STRATEGY_DESC: &str = "anything";

fn strategy() -> impl Strategy<Value = String> {
    "[() ]{0,4}"
}

fn property(s: String) -> TestCaseResult {
    let _ = separate_paren_groups(s);
    Ok(())
}

pub fn run_property(runner: &mut TestRunner) -> Result<(), TestError<String>> {
    runner.run(&strategy(), property)
}
"""

# A harness that violates the prompt: it calls a helper that does not exist.
_BROKEN_PROPERTIES = """\
use crate::candidate::separate_paren_groups;
use proptest::prelude::*;
use proptest::test_runner::{TestCaseError, TestCaseResult, TestError, TestRunner};

fn strategy() -> impl Strategy<Value = String> { ".*" }

fn property(s: String) -> TestCaseResult {
    let _ = separate_paren_groups(s);
    let _ = crate::candidate::a_helper_that_does_not_exist();
    Ok(())
}

pub fn run_property(runner: &mut TestRunner) -> Result<(), TestError<String>> {
    runner.run(&strategy(), property)
}
"""

_CORRECT_CANDIDATE = """\
/// doc
fn separate_paren_groups(paren_string: String) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut depth = 0usize;
    for ch in paren_string.chars() {
        if ch == ' ' { continue; }
        cur.push(ch);
        if ch == '(' { depth += 1; }
        if ch == ')' { depth = depth.saturating_sub(1); }
        if depth == 0 && !cur.is_empty() {
            out.push(cur.clone());
            cur.clear();
        }
    }
    if !cur.is_empty() { out.push(cur); }
    out
}
"""

# Wrong-but-plausible: it forgets to ACCUMULATE a group and emits one entry per
# parenthesis character. It compiles, it R1-passes, and it is wrong for every
# non-empty input — the "compile-clean-but-semantically-wrong" shape the screen
# mis-ranks and the property rung exists to catch.
_WRONG_CANDIDATE = """\
/// doc
fn separate_paren_groups(paren_string: String) -> Vec<String> {
    paren_string
        .chars()
        .filter(|c| !c.is_whitespace())
        .map(|c| c.to_string())
        .collect()
}
"""


def _harness(text: str, slot: int = 0):
    from temper.steer.property import Harness
    return Harness(problem=VIEW.name, slot=slot, seed=slot, source=text,
                   strategy_hash=strategy_hash_of(text))


@unittest.skipUnless(_have_cargo(), "cargo not available on this host")
class TestR3OnCargo(unittest.TestCase):
    """R3 end to end on the real toolchain, one problem, real cargo."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        from temper.harness.runner import run_candidate
        # prime the shared warm cache once, so the pins stay fast
        run_candidate("pub fn separate_paren_groups(p: String) -> Vec<String> { vec![p] }\n",
                      _VACUOUS_PROPERTIES, cases=1, timeout_ms=200,
                      workdir=cls.tmp / "warm")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _run(self, canvas: str, texts: List[str]) -> R3Verdict:
        program = make_pub_program(VIEW, "```rust\n" + canvas + "```\n")
        return run_r3(program.text, [_harness(t, i) for i, t in enumerate(texts)],
                      canvas_ref="p0", problem=VIEW.name, cases=16,
                      timeout_ms=500, workdir=self.tmp / "wd")

    def test_a_correct_canvas_passes_and_a_wrong_one_fails(self):
        """The measurement the primary cell depends on: property evidence
        SEPARATES a container-OK canvas from a wrong-but-compiling one."""
        good = self._run(_CORRECT_CANDIDATE, [_PROP_PROPERTIES])
        bad = self._run(_WRONG_CANDIDATE, [_PROP_PROPERTIES])
        self.assertEqual(good.statuses, ("pass",), "correct canvas must pass")
        self.assertEqual(bad.statuses, ("fail",),
                         "wrong-but-compiling canvas must fail")
        self.assertEqual(good.pass_rate, 1.0)
        self.assertEqual(bad.pass_rate, 0.0)

    def test_the_failure_carries_a_shrunk_minimal_input(self):
        """D2's R3 schema: the failing call records proptest's shrunk input, so
        a property verdict is auditable after the fact."""
        bad = self._run(_WRONG_CANDIDATE, [_PROP_PROPERTIES])
        self.assertTrue(bad.calls[0].minimal_input,
                        "a fail must carry minimal_input")
        self.assertIn("minimal_input", bad.calls[0].as_event())

    def test_a_harness_that_cannot_link_abstains_rather_than_failing(self):
        """The code never ran, so it said nothing about the canvas."""
        v = self._run(_CORRECT_CANDIDATE, [_BROKEN_PROPERTIES])
        self.assertEqual(v.statuses, ("compile_error",))
        self.assertIsNone(v.pass_rate)
        self.assertTrue(v.all_abstained)

    def test_a_vacuous_harness_passes_anything_and_that_is_visible(self):
        """A harness that asserts nothing passes even the wrong canvas — the
        shape G4 must be able to detect, and the reason a passing harness is
        not by itself proof of a good property."""
        v = self._run(_WRONG_CANDIDATE, [_VACUOUS_PROPERTIES])
        self.assertEqual(v.statuses, ("pass",))

    def test_every_call_records_the_harness_sha(self):
        """D2: each R3 event names the exact property text that produced it."""
        v = self._run(_CORRECT_CANDIDATE,
                      [_PROP_PROPERTIES, _VACUOUS_PROPERTIES])
        self.assertEqual(len(v.calls), 2)
        for call in v.calls:
            self.assertEqual(len(call.harness_sha), 64)
        self.assertNotEqual(v.calls[0].harness_sha, v.calls[1].harness_sha)
        events = [c.as_event() for c in v.calls]
        for ev in events:
            for key in ("problem", "harness_sha", "canvas", "status",
                        "duration_s", "minimal_input"):
                self.assertIn(key, ev)

    def test_the_two_harness_pass_rate_aggregates(self):
        """§2.3's pass_rate is over the harness SET, not one harness."""
        v = self._run(_WRONG_CANDIDATE,
                      [_PROP_PROPERTIES, _VACUOUS_PROPERTIES])
        self.assertEqual(sorted(v.statuses), ["fail", "pass"])
        self.assertEqual(v.pass_rate, 0.5)


if __name__ == "__main__":
    unittest.main()
