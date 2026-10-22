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
