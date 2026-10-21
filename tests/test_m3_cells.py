"""D6: paired migration + sign test / McNemar + Jeffreys (reporting-only).

Tolerances and roles under test:
  * the sign test and exact McNemar are the PRIMARY decision quantities
    (rev1 §4.2 R4) — pinned against hand-computed values on toy matrices;
  * the per-problem migration matrix reproduces the M2.3/M2.4a one;
  * Jeffreys is a REPORTING convenience and says so in its own output;
  * the §4.2 reference values reproduce from the committed corrected anchors
    (D6 acceptance: exact).
"""

from __future__ import annotations

import importlib.util
import json
import math
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _load_tool():
    spec = importlib.util.spec_from_file_location(
        "m3_cells", REPO / "scripts" / "m3_cells.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


m3 = _load_tool()


def _cell(label, passes, n_problems=25):
    """A synthetic cell: the first len(passes) problems with the given flags."""
    per = {}
    for i in range(n_problems):
        name = f"P{i:02d}"
        ok = passes[i] if i < len(passes) else False
        per[name] = {"statuses": ["OK"] * (4 if ok else 0) +
                                ["SyntaxError"] * (0 if ok else 4),
                     "n_scored": 4, "n_excluded": 0,
                     "n_ok": 4 if ok else 0, "pass": ok,
                     "all_pass": bool(ok)}
    return {"label": label, "results_path": str(label), "run_path": None,
            "per_problem": per, "n_problems": n_problems,
            "honest_p1": sum(passes) / n_problems, "official_p1": None,
            "n_excluded_rows": 0, "nfe_per_problem": None, "gen_wall_h": None}


class TestSignTest(unittest.TestCase):
    def test_no_discordant_pairs_is_p_one(self):
        self.assertEqual(m3.sign_test_one_sided(0, 0), 1.0)

    def test_known_exact_values(self):
        # P(X >= k | n, 1/2)
        self.assertAlmostEqual(m3.sign_test_one_sided(1, 0), 0.5)
        self.assertAlmostEqual(m3.sign_test_one_sided(2, 0), 0.25)
        self.assertAlmostEqual(m3.sign_test_one_sided(5, 0), 1 / 32)
        self.assertAlmostEqual(m3.sign_test_one_sided(6, 0), 1 / 64)
        # 6 discordant, one of them against: P(X>=5 of 6) = 7/64
        self.assertAlmostEqual(m3.sign_test_one_sided(5, 1), 7 / 64)
        # the other direction: P(X>=1 of 6) = 63/64
        self.assertAlmostEqual(m3.sign_test_one_sided(1, 5), 63 / 64)

    def test_one_sided_alpha_05_needs_five_clean_flips(self):
        self.assertGreater(m3.sign_test_one_sided(4, 0), 0.05)   # 1/16
        self.assertLess(m3.sign_test_one_sided(5, 0), 0.05)      # 1/32

    def test_mcnemar_two_sided(self):
        self.assertAlmostEqual(m3.mcnemar_exact(6, 0), 2 / 64)
        self.assertAlmostEqual(m3.mcnemar_exact(0, 6), 2 / 64)
        self.assertAlmostEqual(m3.mcnemar_exact(3, 3), 1.0)
        self.assertEqual(m3.mcnemar_exact(0, 0), 1.0)


class TestMigration(unittest.TestCase):
    def test_matrix_counts_and_discordant_set(self):
        a = _cell("a", [True, True, False, False], n_problems=4)
        b = _cell("b", [True, False, True, False], n_problems=4)
        m = m3.migration_matrix(a, b)
        self.assertEqual(m["table"], {"pass_pass": 1, "pass_fail": 1,
                                      "fail_pass": 1, "fail_fail": 1})
        flips = sorted(r["problem"] for r in m["rows"]
                       if r["transition"] not in ("pass_pass", "fail_fail"))
        self.assertEqual(flips, ["P01", "P02"])

    def test_paired_report_carries_detectable_effect_note(self):
        a = _cell("a", [False] * 5)
        b = _cell("b", [True] * 5)
        rep = m3.print_paired(a, b, primary=True)
        self.assertIn("discordant", rep["detectable_effect"])
        self.assertAlmostEqual(rep["sign_test_p_one_sided"], 1 / 32)


class TestJeffreysIsReportingOnly(unittest.TestCase):
    def test_caveat_present(self):
        j = m3.jeffreys_posterior(14)
        self.assertIn("reporting-only", j["caveat"])
        self.assertAlmostEqual(j["P>0.44"], 0.886, delta=0.006)

    def test_reference_values_reproduce(self):
        self.assertAlmostEqual(m3.jeffreys_posterior(14)["P>0.44"], 0.886, delta=0.006)
        self.assertAlmostEqual(m3.jeffreys_posterior(16)["P>0.44"], 0.978, delta=0.006)
        self.assertAlmostEqual(m3.jeffreys_posterior(18)["P>0.64"], 0.795, delta=0.006)
        self.assertAlmostEqual(m3.jeffreys_posterior(13)["P>0.44"], 0.79, delta=0.006)

    def test_deterministic_across_calls(self):
        self.assertEqual(m3.jeffreys_posterior(15), m3.jeffreys_posterior(15))


class TestCorrectedAnchorsReproduce(unittest.TestCase):
    """D6 acceptance: the corrected-basis anchors reproduce exactly."""

    def test_anchor_table(self):
        expected = {"s96": 0.16, "s128": 0.20, "s256": 0.44, "s512": 0.64}
        for label, hon in expected.items():
            results, run = m3.CORRECTED[label]
            cell = m3.load_cell(f"{label}={results}:{run}")
            self.assertAlmostEqual(cell["honest_p1"], hon, places=6,
                                   msg=f"{label}: honest p@1")
            self.assertEqual(cell["n_problems"], 25)
            self.assertEqual(sum(1 for v in cell["per_problem"].values()
                                 if v["pass"]),
                             int(round(hon * 25)),
                             f"{label}: passer count")

    def test_nfe_and_wall_reproduce_the_m2_4b_table(self):
        expected = {"s96": (384, 6.18), "s128": (512, 8.82),
                    "s256": (1024, 7.21), "s512": (2048, 14.48)}
        for label, (nfe, wall) in expected.items():
            results, run = m3.CORRECTED[label]
            cell = m3.load_cell(f"{label}={results}:{run}")
            self.assertAlmostEqual(cell["nfe_per_problem"], nfe, places=1)
            self.assertAlmostEqual(cell["gen_wall_h"], wall, places=2)

    def test_exclusion_counting_is_honest(self):
        results, run = m3.CORRECTED["s256"]
        cell = m3.load_cell(f"s256={results}:{run}")
        # the 17/19 exclusion pair is 4 rows each and must NOT count as a pass
        self.assertFalse(cell["per_problem"]["HumanEval_17_parse_music"]["pass"])
        self.assertFalse(cell["per_problem"]["HumanEval_19_sort_numbers"]["pass"])
        self.assertEqual(cell["n_excluded_rows"], 8)

    def test_reproduce_cli_passes(self):
        self.assertEqual(m3.reproduce_references(), 0)


if __name__ == "__main__":
    unittest.main()


class TestHintQuartetTable(unittest.TestCase):
    """P0.5 reporting tool: the quartet table + the design's decision rule."""

    @classmethod
    def setUpClass(cls):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "m3_p05", REPO / "scripts" / "m3_p05_hint_table.py")
        cls.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.mod)

    def test_status_classification(self):
        self.assertEqual(self.mod.classify({"statuses_official": ["OK"] * 4,
                                            "n_excluded": 0}), "PASS4")
        self.assertEqual(self.mod.classify({"statuses_official": ["SyntaxError"] * 4,
                                            "n_excluded": 0}), "S4")
        self.assertEqual(self.mod.classify({"statuses_official": [],
                                            "n_excluded": 4}), "EXCL4")
        self.assertEqual(self.mod.classify({"statuses_official":
                                            ["OK", "SyntaxError", "OK", "Exception"],
                                            "n_excluded": 0}), "E1P2S1")
        self.assertEqual(self.mod.classify(None), "NOT-RUN")

    def test_pass_requires_at_least_one_ok(self):
        self.assertTrue(self.mod.passes({"statuses_official": ["OK", "SyntaxError"]}))
        self.assertFalse(self.mod.passes({"statuses_official": ["SyntaxError"] * 4}))
        self.assertFalse(self.mod.passes({"statuses_official": [], "n_excluded": 4}))
        self.assertFalse(self.mod.passes(None))
