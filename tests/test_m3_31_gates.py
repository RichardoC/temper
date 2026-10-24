"""M3.1 gate pins: every gate must be able to FAIL, and fail for the right reason.

P3's rule, carried forward: a gate that cannot fail is not a gate. Each pin
builds the smallest artifact set that trips ONE clause of ONE gate and asserts
that the gate reports that clause and not the others — so a later refactor
cannot quietly soften a gate into vacuity.

Also pinned: the primary comparison's DIRECTION (design §3 pre-registers
P-prop-screen vs C-screen, and a positive delta must mean the property cell is
better), and the honest-power statement every gate line carries.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import mkdtemp
import shutil

from scripts.m3_31_analysis import (
    detectable,
    gate_g1,
    gate_g2,
    gate_g3,
    gate_g4,
    gate_g5,
    load_cell,
    paired_report,
    sign_test_one_sided,
)


def _cell(label: str, passing, n: int = 25) -> dict:
    """A synthetic cell: the problems whose number is in ``passing`` pass."""
    per = {}
    for i in range(n):
        name = f"HumanEval_{i}_p"
        ok = i in passing
        per[name] = {"statuses": ["OK"] if ok else ["SyntaxError"],
                     "n_ok": 1 if ok else 0, "pass": ok, "emitted": 0,
                     "emitted_temp": 0.2, "property_pass_rate": None}
    return {"label": label, "results_path": "", "per_problem": per,
            "n_problems": n, "n_passers": len(passing),
            "honest_p1": len(passing) / n, "run_path": None,
            "emission_cell": label}


def _detail_row(problem: str, *, up=True, same_pool=True, screen_ok=False,
                any_ok=True, prop_ok=True) -> dict:
    return {
        "problem": problem,
        "flip_direction": "up" if up else "down",
        "g2_a_same_pool": same_pool,
        "g2_b_screen_emitted_rejected": not screen_ok,
        "g2_c_ok_canvas_existed": any_ok,
        "g2_d_property_emitted_ok": prop_ok,
        "g2_attributed": bool(up and same_pool and not screen_ok and any_ok),
    }


class TestSignTest(unittest.TestCase):
    def test_the_alpha_05_threshold_is_five_one_directional_flips(self):
        """The honest-power claim on every gate line, pinned numerically: at
        n=25 paired draws alpha=0.05 one-sided needs FIVE one-directional
        discordants (1/32 = 0.03125); four (1/16 = 0.0625) cannot clear it."""
        self.assertLess(sign_test_one_sided(5, 0), 0.05)
        self.assertEqual(sign_test_one_sided(5, 0), 1 / 32)
        self.assertGreater(sign_test_one_sided(4, 0), 0.05)
        self.assertEqual(sign_test_one_sided(4, 0), 1 / 16)
        self.assertEqual(sign_test_one_sided(3, 0), 0.125)
        self.assertEqual(sign_test_one_sided(0, 0), 1.0)

    def test_detectable_names_the_five_flips_for_n25(self):
        self.assertIn("5", detectable(25))


class TestG1CanFail(unittest.TestCase):
    def test_below_16_fails(self):
        g = gate_g1(_cell("propscreen", {0, 1, 2}), _cell("cscreen", {0}))
        self.assertFalse(g["checks"]["passers_ge_16"])
        self.assertFalse(g["pass"])

    def test_16_passers_but_a_single_flip_fails_the_margin_clause(self):
        """The +2 net-flip margin is a SEPARATE clause: a cell can reach the
        pass line on the strength of the v2 rule's own picks and still fail."""
        screen = {i for i in range(16)}       # the v2 rule's own 16 picks
        prop = (screen - {0}) | {20}          # one up, one down = net 0
        g = gate_g1(_cell("propscreen", prop), _cell("cscreen", screen))
        self.assertTrue(g["checks"]["passers_ge_16"])
        self.assertFalse(g["checks"]["net_flips_up_ge_2"])
        self.assertFalse(g["pass"])

    def test_the_pre_registered_pass_case(self):
        screen = {1, 2, 3, 5, 7, 8, 13, 14, 15, 16, 23, 24, 25, 4}
        prop = (screen - {1, 9}) | {1, 9, 17, 18}
        g = gate_g1(_cell("propscreen", prop), _cell("cscreen", screen))
        self.assertGreaterEqual(g["passers"], 16)
        self.assertGreaterEqual(g["margin"], 2)


class TestG2CanFail(unittest.TestCase):
    def test_a_flip_without_an_ok_canvas_is_not_attributed(self):
        detail = {"attribution": [
            _detail_row("HumanEval_1_p", any_ok=False),
            _detail_row("HumanEval_9_p", any_ok=False)]}
        g = gate_g2(detail)
        self.assertEqual(g["n_attributed"], 0)
        self.assertFalse(g["pass"])

    def test_a_flip_down_fails_the_gate(self):
        detail = {"attribution": [
            _detail_row("HumanEval_1_p"), _detail_row("HumanEval_9_p"),
            _detail_row("HumanEval_4_p", up=False)]}
        g = gate_g2(detail)
        self.assertFalse(g["checks"]["no_flips_down"])
        self.assertFalse(g["pass"])

    def test_a_different_pool_breaks_attribution(self):
        detail = {"attribution": [
            _detail_row("HumanEval_1_p"), _detail_row("HumanEval_9_p"),
            _detail_row("HumanEval_4_p", same_pool=False)]}
        g = gate_g2(detail)
        self.assertEqual(g["n_attributed"], 2)
        self.assertIn("HumanEval_4_p", g["unattributed_ups"])
        self.assertFalse(g["pass"])

    def test_the_measured_targets_are_reported(self):
        detail = {"attribution": [_detail_row("HumanEval_1_p"),
                                  _detail_row("HumanEval_9_p")]}
        g = gate_g2(detail)
        self.assertEqual(g["measured_targets"], [1, 9])
        self.assertEqual(g["measured_targets_hit"], [1, 9])


class TestG3CanFail(unittest.TestCase):
    def test_16_fails(self):
        g = gate_g3(_cell("ppropengine", {0, 1, 2}), _cell("cfull", set()),
                    None)
        self.assertFalse(g["checks"]["passers_ge_17"])
        self.assertFalse(g["pass"])

    def test_a_s256_regression_fails(self):
        """Problem 1 is in the banked 11-passer cohort: losing it is a
        regression even at 17 passers."""
        # exactly 17 passers, but problem 1 (in the banked 11-passer cohort) is
        # gone — its slot is filled by problem 20, so the COUNT clears 17 while
        # the regression clause fires
        # the 10 surviving banked cohort members plus 7 outside-cohort wins:
        # exactly 17 passers, with problem 1 (a banked passer) LOST
        passing = {2, 3, 5, 7, 8, 13, 14, 15, 16, 23} | {0, 4, 6, 9, 10, 11, 12}
        g = gate_g3(_cell("ppropengine", passing), _cell("cfull", set()), None)
        self.assertEqual(g["passers"], 17)
        self.assertEqual(g["s256_regressions"], ["HumanEval_1_p"])
        self.assertFalse(g["pass"])

    def test_the_clarification_is_attached(self):
        g = gate_g3(_cell("ppropengine", set()), None, None)
        self.assertIn("Q3", g["clarification"])
        self.assertIn("UPPER BOUND", g["clarification"])


class TestG4CanFail(unittest.TestCase):
    """G4 AS AMENDED (§7 A5 + A8): the supply gate is >=19/21 control-available
    and the 90% compile bar is GONE. These pins make the amendment itself
    testable: the old bar must not silently return, and the new gate must be able
    to fail."""

    def _detail(self, agree, disagree):
        return {"gates": {"G4": {"n_measured": agree + disagree,
                                 "property_agrees_with_container": agree,
                                 "property_disagrees_with_container": disagree,
                                 "n_abstained": 0}}}

    def _freeze(self, covered, n_ctrl=21, first_rate=0.16):
        return {"compile_rate": {"first_draft_rate": first_rate},
                "coverage": {"problems_with_ge1_harness": covered,
                             "n_control_available": n_ctrl}}

    def test_the_removed_90pct_bar_cannot_sink_the_gate(self):
        """A5: the compile rate is reported, never gated. A 16% rate — the
        measured reality — must NOT fail G4 when supply and correlation hold."""
        g = gate_g4(self._detail(64, 0), self._freeze(19, first_rate=0.16))
        self.assertEqual(g["first_draft_compile_rate_reported"], 0.16)
        self.assertNotIn("harness_first_draft_compile_rate_ge_90pct", g["checks"])
        self.assertTrue(g["pass"])

    def test_supply_below_19_fails(self):
        g = gate_g4(self._detail(64, 0), self._freeze(18))
        self.assertFalse(g["checks"]["G4a_supply_ge_19_of_21_control_available"])
        self.assertFalse(g["pass"])

    def test_correlation_failing_still_fails_the_gate(self):
        g = gate_g4(self._detail(2, 5), self._freeze(19))
        self.assertFalse(g["checks"]["G4b_property_right_more_often_than_wrong"])
        self.assertFalse(g["pass"])

    def test_the_supply_gate_tracks_the_measured_ceiling(self):
        """A8: the ceiling is the CONTROL-AVAILABLE count, not 25. A cell whose
        ceiling is 21 is judged out of 21."""
        g = gate_g4(self._detail(64, 0), self._freeze(19, n_ctrl=21))
        self.assertEqual(g["n_control_available"], 21)
        self.assertTrue(g["pass"])
        # the gate is RELATIVE to the ceiling as well as absolute: with a full
        # pool of 25 control-available problems, 19 covered is NOT enough
        # (n_ctrl - 2 = 23), which is exactly the original strict behaviour
        # returning for a cell that could actually supply it.
        g2 = gate_g4(self._detail(64, 0), self._freeze(19, n_ctrl=25))
        self.assertFalse(g2["pass"],
                         "with 25 control-available problems the ceiling clause "
                         "must still demand near-full coverage")
        g3 = gate_g4(self._detail(64, 0), self._freeze(23, n_ctrl=25))
        self.assertTrue(g3["pass"])


class TestG5CanFail(unittest.TestCase):
    def test_three_of_four_vs_none_is_engine_specific(self):
        g = gate_g5({"engine_passes": 3, "unsteered_passes": 0, "n_games": 4})
        self.assertEqual(g["classification"], "engine_specific")
        self.assertTrue(g["pass"])

    def test_three_of_four_vs_two_is_draw_luck(self):
        g = gate_g5({"engine_passes": 3, "unsteered_passes": 2, "n_games": 4})
        self.assertEqual(g["classification"], "draw_luck")
        self.assertFalse(g["pass"])

    def test_two_of_four_is_draw_luck(self):
        g = gate_g5({"engine_passes": 2, "unsteered_passes": 0, "n_games": 4})
        self.assertEqual(g["classification"], "draw_luck")

    def test_a_skipped_gate_is_not_a_pass(self):
        self.assertTrue(gate_g5(None).get("skipped"))
        self.assertIsNone(gate_g5(None)["pass"])


class TestPrimaryDirection(unittest.TestCase):
    def test_a_positive_delta_means_the_second_cell_is_better(self):
        """Design §3 pre-registers P-prop-screen vs C-screen; scripts take
        (A, B) and report delta = B - A, so 'screen propscreen' must yield a
        POSITIVE delta when the property rule recovers problems."""
        screen = _cell("cscreen", {0, 1, 2})
        prop = _cell("propscreen", {0, 1, 2, 8, 9})
        rep = paired_report(screen, prop)
        self.assertEqual(rep["delta"], 2)
        self.assertEqual(rep["flips_up"], ["HumanEval_8_p", "HumanEval_9_p"])

    def test_within_ensemble_comparisons_have_zero_draw_variance(self):
        """A rule delta over ONE ensemble cannot change the pass->fail direction
        unless it emits a rejected canvas where the other emitted an OK one —
        which is exactly what G2 clause (b) tracks."""
        screen = _cell("cscreen", {0})
        prop = _cell("propscreen", {0})
        rep = paired_report(screen, prop)
        self.assertEqual(rep["n_discordant"], 0)
        self.assertEqual(rep["sign_test_p_one_sided"], 1.0)


if __name__ == "__main__":
    unittest.main()
