"""Unit tests for temper.multipl_e.score (M2.0 item 3) — pure parts only.

The full container round trip is exercised by the M2.0 smoke test
(docs/m2.0-report.md); these tests pin the estimator math and the
manifest/input parsing.
"""

from __future__ import annotations

import json
import math
import shutil
import tempfile
import unittest
from pathlib import Path

from temper.multipl_e.score import _parse_input_dir, pass_at_k


def _comb_pass_at_k(n, c, k):
    """Ground truth from the closed form 1 - C(n-c,k)/C(n,k)."""
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


class TestPassAtK(unittest.TestCase):
    def test_estimator_matches_closed_form(self):
        for n in (1, 2, 4, 10, 37):
            for c in range(0, n + 1):
                for k in (1, 2, 10):
                    if k > n:
                        continue
                    self.assertAlmostEqual(
                        pass_at_k(n, c, k), _comb_pass_at_k(n, c, k),
                        places=10,
                        msg=f"n={n} c={c} k={k}")

    def test_saturation(self):
        self.assertEqual(pass_at_k(1, 1, 1), 1.0)
        self.assertEqual(pass_at_k(10, 1, 10), 1.0)  # n - c < k
        self.assertEqual(pass_at_k(4, 0, 1), 0.0)

    def test_known_value(self):
        # n=4, c=2, k=1 -> 0.5
        self.assertAlmostEqual(pass_at_k(4, 2, 1), 0.5)
        # n=100, c=10, k=10 (Codex paper canonical example shape)
        self.assertAlmostEqual(pass_at_k(100, 10, 10),
                               _comb_pass_at_k(100, 10, 10), places=10)


class TestParseInputDir(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_manifest_excluded_from_problems(self):
        (self.tmp / "temper-sort.json").write_text(json.dumps({
            "name": "temper-sort", "language": "rs", "prompt": "",
            "tests": "fn main() {}", "completions": ["fn f() {}"]}))
        (self.tmp / "manifest.json").write_text(json.dumps({
            "problems": [{"file": "temper-sort.json", "name": "temper-sort",
                          "completions": 1}],
            "excluded": [{"run": "r.jsonl", "i": 1,
                          "problem": "temper-sort", "reason": "x"}],
        }))
        info = _parse_input_dir(self.tmp)
        self.assertEqual(set(info["problems"]), {"temper-sort"})
        self.assertEqual(info["problems"]["temper-sort"]["n_scored"], 1)
        self.assertEqual(info["excluded_by_problem"], {"temper-sort": 1})
        self.assertEqual(len(info["excluded"]), 1)

    def test_no_manifest(self):
        (self.tmp / "p.json").write_text(json.dumps({
            "name": "p", "completions": ["a", "b"]}))
        info = _parse_input_dir(self.tmp)
        self.assertEqual(info["problems"]["p"]["n_scored"], 2)
        self.assertEqual(info["excluded_by_problem"], {})


if __name__ == "__main__":
    unittest.main(verbosity=2)
