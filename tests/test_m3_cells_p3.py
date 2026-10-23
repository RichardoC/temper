"""M3.0 P3 cell machinery: the two emission rules and the C-full driver seam.

The emission rule is part of a cell's DEFINITION (design §4.1), so both the
in-loop engine and the offline counterfactual must use the same one — otherwise
C-full vs C-screen measures two implementations of a rule instead of the
mechanism between them. Pinned here:

  * ``select_structural`` (C-null) prefers the T=0.2 canvas when it is clean,
    falls back to the lowest-T clean canvas, then to max confidence — the three
    branches design rev1 §4.1 names;
  * ``select_screen`` (C-screen, C-full) picks best clean by mean commit
    confidence with the (confidence, lowest temp, index) tie-break;
  * the C-full ladder is the frozen rev2 prefix and a non-integral envelope is
    refused rather than silently re-composed;
  * the steer engine returns exactly ONE row per problem (the game's emitted
    canvas) so a C-full cell is commensurable with the per-problem binary the
    unsteered anchors report.
"""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from temper.steer.emission import (
    STRUCTURAL_TEMP,
    rank_screen,
    select_screen,
    select_structural,
)
from temper.steer.engine import REV2_LADDER, ladder_for_envelope


@dataclass
class _Cand:
    """A stand-in for a sufficiently scored canvas."""

    index: int
    temperature: float
    final_clean: bool
    mean_commit_conf: Optional[float]


class TestStructuralEmission(unittest.TestCase):
    """(i) C-null's rule: T=0.2 if clean, else lowest-T clean."""

    def test_structural_canvas_wins_when_clean(self):
        cands = [_Cand(0, 0.2, True, 0.10), _Cand(1, 0.8, True, 0.90)]
        got = select_structural(cands)
        self.assertEqual(got.index, 0,
                         "a clean T=0.2 canvas must be emitted even when a "
                         "hotter canvas has higher confidence — C-null is the "
                         "ladder-only control, not a confidence selector")

    def test_lowest_temperature_clean_canvas_is_the_fallback(self):
        cands = [_Cand(0, 0.2, False, 0.90),
                 _Cand(1, 1.0, True, 0.20),
                 _Cand(2, 0.8, True, 0.10)]
        self.assertEqual(select_structural(cands).index, 2)

    def test_no_clean_canvas_falls_back_to_max_confidence(self):
        cands = [_Cand(0, 0.2, False, 0.10), _Cand(1, 0.8, False, 0.70)]
        self.assertEqual(select_structural(cands).index, 1,
                         "design 3.3 item 2: the game always emits an answer")

    def test_structural_temp_matches_the_rev2_ladder(self):
        self.assertAlmostEqual(STRUCTURAL_TEMP, REV2_LADDER[0])

    def test_duplicate_temperature_ties_go_to_the_lowest_index(self):
        # the rev2 ladder places 0.8 twice; the fallback must be deterministic
        cands = [_Cand(0, 0.2, False, 0.9), _Cand(6, 0.8, True, 0.5),
                 _Cand(1, 0.8, True, 0.4)]
        self.assertEqual(select_structural(cands).index, 1)


class TestScreenEmission(unittest.TestCase):
    """(ii) C-screen / C-full's rule: best clean by confidence, ties lowest T."""

    def test_best_confidence_clean_canvas_wins(self):
        cands = [_Cand(0, 0.2, True, 0.30), _Cand(3, 1.1, True, 0.80)]
        self.assertEqual(select_screen(cands).index, 3,
                         "the screen ranks clean canvases by confidence, so "
                         "C-null and C-screen differ exactly here")

    def test_dirty_canvases_are_excluded_when_any_clean_exists(self):
        cands = [_Cand(0, 0.2, False, 0.99), _Cand(1, 0.8, True, 0.10)]
        self.assertEqual(select_screen(cands).index, 1)

    def test_tie_on_confidence_goes_to_the_lowest_temperature(self):
        cands = [_Cand(5, 1.2, True, 0.55), _Cand(2, 0.8, True, 0.55)]
        self.assertEqual(select_screen(cands).index, 2)

    def test_tie_on_confidence_and_temperature_goes_to_the_lowest_index(self):
        cands = [_Cand(6, 0.8, True, 0.55), _Cand(1, 0.8, True, 0.55)]
        self.assertEqual(select_screen(cands).index, 1)

    def test_no_clean_canvas_falls_back_to_max_confidence(self):
        cands = [_Cand(0, 0.2, False, 0.10), _Cand(1, 1.0, False, 0.66)]
        self.assertEqual(select_screen(cands).index, 1)

    def test_missing_confidence_ranks_last_among_clean(self):
        cands = [_Cand(0, 0.2, True, None), _Cand(1, 1.2, True, 0.01)]
        self.assertEqual(select_screen(cands).index, 1)

    def test_ranking_is_the_same_order_the_selection_uses(self):
        cands = [_Cand(0, 0.2, True, 0.30), _Cand(1, 0.8, True, 0.90),
                 _Cand(2, 1.0, False, 0.99)]
        self.assertEqual([c.index for c in rank_screen(cands)], [1, 0])

    def test_empty_ensemble_selects_nothing(self):
        self.assertIsNone(select_screen([]))
        self.assertIsNone(select_structural([]))


class TestEnvelopeLadder(unittest.TestCase):
    """The rev2 ladder is frozen; the envelope decides how much of it is bought."""

    def test_full_envelope_buys_the_whole_rev2_ladder(self):
        self.assertEqual(ladder_for_envelope(2048, 256), list(REV2_LADDER))

    def test_rev2_ladder_is_the_preregistered_grid(self):
        # T=0.2 structural + round-robin {0.8, 0.95, 1.0, 1.1, 1.2}, padded
        self.assertEqual(REV2_LADDER,
                         (0.2, 0.8, 0.95, 1.0, 1.1, 1.2, 0.8, 0.95))

    def test_partial_envelope_takes_a_prefix_in_ladder_order(self):
        self.assertEqual(ladder_for_envelope(1024, 256), [0.2, 0.8, 0.95, 1.0])

    def test_non_integral_envelope_is_refused(self):
        # 2048 NFE at 300 steps is 6.8 canvases: the ladder's composition
        # (how many draws each temperature gets) would be ambiguous
        with self.assertRaises(ValueError):
            ladder_for_envelope(2048, 300)

    def test_envelope_larger_than_the_ladder_is_refused(self):
        with self.assertRaises(ValueError):
            ladder_for_envelope(256 * 9, 256)


class TestSteerEngineSeam(unittest.TestCase):
    """The bench driver's ``sample_fn`` contract: ONE row per problem."""

    def test_ladder_reaches_the_engine_from_the_cli_shape(self):
        """The CLI passes --envelope-nfe; the engine derives the ladder from it
        and the frozen rev2 prefix, so a mismatched envelope is caught before
        any GPU time is spent."""
        from temper.steer.engine import make_steer_sample_fn
        fn = make_steer_sample_fn(
            out=Path(tempfile.mkdtemp()) / "c-full.jsonl",
            envelope_nfe=2048, steps=256, canvas_len=384)
        self.assertTrue(callable(fn))

    def test_engine_refuses_more_than_one_particle(self):
        from temper.steer.engine import make_steer_sample_fn
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(tmp))
        fn = make_steer_sample_fn(out=tmp / "c-full.jsonl", envelope_nfe=2048,
                                  steps=256, canvas_len=384)
        with self.assertRaises(ValueError) as ctx:
            fn("some prompt", canvas_len=384, steps=256, particles=4)
        self.assertIn("-K 1", str(ctx.exception))

    def test_engine_refuses_an_unknown_prompt(self):
        from temper.steer.engine import make_steer_sample_fn
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(tmp))
        fn = make_steer_sample_fn(out=tmp / "c-full.jsonl", envelope_nfe=2048,
                                  steps=256, canvas_len=384)
        with self.assertRaises(ValueError) as ctx:
            fn("a prompt no vendored problem produced", canvas_len=384,
               steps=256, particles=1)
        self.assertIn("does not match", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()


def _analysis_tool():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "m3_p3_analysis",
        Path(__file__).resolve().parents[1] / "scripts" / "m3_p3_analysis.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _replication_tool():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "m3_p3_replication",
        Path(__file__).resolve().parents[1] / "scripts" / "m3_p3_replication.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestCohortTable(unittest.TestCase):
    """The design §6.1 cohorts are named by HumanEval NUMBER, and the P1
    reachability tags overlays them — a flip inside either is not mechanism
    evidence."""

    def setUp(self):
        self.a = _analysis_tool()

    def test_design_cohorts_resolve_by_number(self):
        self.assertEqual(self.a.cohort_of("HumanEval_0_has_close_elements"),
                         "type_quartet")
        self.assertEqual(self.a.cohort_of("HumanEval_24_largest_divisor"),
                         "semantic_trio")
        self.assertEqual(self.a.cohort_of("HumanEval_17_parse_music"),
                         "exclusion_pair")
        self.assertEqual(self.a.cohort_of("HumanEval_23_strlen"),
                         "structural_base")

    def test_p1_reachability_is_an_overlay_not_a_bucket(self):
        # 17 is BOTH an exclusion problem and P1-reachable — the report's
        # stricter reading of the attribution gate depends on seeing both
        tags = self.a.cohort_tags("HumanEval_17_parse_music")
        self.assertIn("exclusion_pair", tags)
        self.assertIn("p1_reachable", tags)

    def test_p1_unreachable_set_is_the_probes_four(self):
        self.assertEqual(set(self.a.P1_UNREACHABLE), {0, 18, 19, 21})

    def test_s256_passer_set_is_the_banked_eleven(self):
        self.assertEqual(sorted(self.a.S256_PASSERS),
                         [1, 2, 3, 5, 7, 8, 13, 14, 15, 16, 23])
        self.assertEqual(len(self.a.S256_PASSERS), 11)


class TestGates(unittest.TestCase):
    """The gates must FAIL in the ways design §4.2 says they must — a gate that
    cannot fail is not a gate."""

    def setUp(self):
        self.a = _analysis_tool()

    def _cell(self, label, passers_by_number):
        per = {}
        for n in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                  17, 18, 19, 20, 21, 23, 24, 25]:
            ok = n in passers_by_number
            name = f"HumanEval_{n}_p"
            per[name] = {"statuses": ["OK"] if ok else ["SyntaxError"],
                         "n_ok": 1 if ok else 0, "n_scored": 1, "n_excluded": 0,
                         "pass": ok, "emitted": 0, "emitted_temp": 0.2}
        return {"label": label, "per_problem": per, "n_problems": len(per),
                "n_passers": len(passers_by_number), "honest_p1": None}

    def test_ship_gate_needs_fifteen(self):
        base = set(self.a.S256_PASSERS)
        fourteen = base | {0, 4, 20}
        g = self.a.gate_ship(self._cell("cfull", fourteen),
                             self._cell("anchor1024", base),
                             self._cell("cnull", base), None)
        self.assertFalse(g["checks"]["passers_ge_15"])
        self.assertFalse(g["pass"])
        fifteen = base | {0, 4, 20, 21}
        g = self.a.gate_ship(self._cell("cfull", fifteen),
                             self._cell("anchor1024", base),
                             self._cell("cnull", base), None)
        self.assertTrue(g["checks"]["passers_ge_15"])

    def test_ship_gate_fails_on_a_regression_of_an_s256_passer(self):
        base = set(self.a.S256_PASSERS)
        # 16 passers but problem 3 (an s256 passer) was LOST
        with_regression = (base - {3}) | {0, 4, 11, 17, 18, 20}
        self.assertEqual(len(with_regression), 16)
        g = self.a.gate_ship(self._cell("cfull", with_regression),
                             self._cell("anchor1024", base),
                             self._cell("cnull", base), None)
        self.assertEqual(len(g["regressions"]), 1)
        self.assertFalse(g["checks"]["zero_net_regressions_on_11_s256_passers"])
        self.assertFalse(g["pass"], "the ship gate demands zero net regressions")

    def test_ship_gate_needs_confirmed_attributed_flips(self):
        base = set(self.a.S256_PASSERS)
        strong = base | {0, 4, 11, 17, 20}
        g = self.a.gate_ship(self._cell("cfull", strong),
                             self._cell("anchor1024", base),
                             self._cell("cnull", base), None)
        self.assertFalse(g["checks"]["confirmed_attributed_flips_ge_2"])
        repl = {"all_green": True, "problems": [
            {"problem": "HumanEval_0_p", "confirmed_attributed": True},
            {"problem": "HumanEval_11_p", "confirmed_attributed": True}]}
        g = self.a.gate_ship(self._cell("cfull", strong),
                             self._cell("anchor1024", base),
                             self._cell("cnull", base), repl)
        self.assertTrue(g["checks"]["confirmed_attributed_flips_ge_2"])

    def test_attribution_gate_fails_when_cfull_equals_the_nulls(self):
        base = set(self.a.S256_PASSERS)
        same = base | {0, 4, 11}
        g = self.a.gate_attribution(self._cell("cfull", same),
                                    self._cell("cnull", same),
                                    self._cell("cscreen", same), None)
        self.assertFalse(g["checks"]["margin_vs_cnull_ge_2"])
        self.assertFalse(g["pass"],
                         "C-full == C-null is a measured null, not a result")

    def test_attribution_gate_needs_replication_green(self):
        base = set(self.a.S256_PASSERS)
        cfull = base | {0, 4, 11, 17}
        cnull = base
        g = self.a.gate_attribution(self._cell("cfull", cfull),
                                    self._cell("cnull", cnull),
                                    self._cell("cscreen", cnull), None)
        self.assertFalse(g["checks"]["flip_set_replication_green"])
        self.assertFalse(g["pass"])

    def test_attribution_gate_rejects_flips_confined_to_the_reachable_set(self):
        """A flip on the type quartet or on a P1-reachable problem could be
        diversity or knowledge delivery — the design text's outside-cohort
        requirement, made explicit here."""
        base = set(self.a.S256_PASSERS)
        cfull = base | {0, 4}          # both in the type quartet
        repl = {"all_green": True, "problems": [
            {"problem": "HumanEval_0_p", "confirmed_attributed": False},
            {"problem": "HumanEval_4_p", "confirmed_attributed": False}]}
        g = self.a.gate_attribution(self._cell("cfull", cfull),
                                    self._cell("cnull", base),
                                    self._cell("cscreen", base), repl)
        self.assertEqual(g["outside_cohort_flips"], [])
        self.assertFalse(
            g["checks"]["at_least_one_flip_outside_ladder_reachable"])

    def test_strong_gate_needs_eighteen(self):
        base = set(self.a.S256_PASSERS)
        self.assertFalse(self.a.gate_strong(
            self._cell("cfull", base | {0, 4, 11, 17}))["pass"])
        self.assertTrue(self.a.gate_strong(
            self._cell("cfull", base | {0, 4, 11, 17, 18, 19, 20}))["pass"])


class TestFlipReplication(unittest.TestCase):
    """A flip is one Bernoulli draw; confirmation needs the second game."""

    def _run(self, *args):
        import io
        import contextlib
        from scripts.m3_p3_replication import main
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = main(list(args))
        return rc, buf.getvalue()

    def test_a_flip_that_does_not_reproduce_is_not_confirmed(self):
        import json
        import tempfile
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(tmp))

        # REAL vendored names: the tool refuses unknown problems rather than
        # guessing which problem a flip belongs to (its cohort tags depend on
        # the HumanEval number).
        names = {17: "HumanEval_17_parse_music",
                 19: "HumanEval_19_sort_numbers",
                 21: "HumanEval_21_rescale_to_unit"}

        def cell(label, passes):
            rows = []
            for n in [17, 19, 21]:
                rows.append({"name": names[n],
                             "statuses_official": ["OK"] if n in passes
                             else ["SyntaxError"],
                             "n_scored": 1, "n_excluded": 0})
            p = tmp / f"{label}.json"
            p.write_text(json.dumps({"per_problem": rows,
                                     "aggregate": {}}), encoding="utf-8")
            return str(p)
        prim = cell("prim", {19})
        repl = cell("repl", set())      # the flip did not survive
        out = tmp / "r.json"
        rc, text = self._run("--primary", prim, "--replication", repl,
                             "--flips", names[19], "--json", str(out))
        self.assertEqual(rc, 0)
        self.assertIn("NOT-REPRODUCED", text)
        payload = json.loads(out.read_text(encoding="utf-8"))
        self.assertFalse(payload["all_green"])
        self.assertEqual(payload["n_confirmed_attributed"], 0)

    def test_unknown_problem_name_is_refused(self):
        import tempfile
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(tmp))
        p = tmp / "x.json"
        p.write_text('{"per_problem": [], "aggregate": {}}', encoding="utf-8")
        with self.assertRaises(SystemExit):
            self._run("--primary", str(p), "--replication", str(p),
                      "--flips", "HumanEval_999_nope")
