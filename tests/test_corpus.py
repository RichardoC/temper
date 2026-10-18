"""Unit tests for the M1.5 corpus writer changes in temper.steer.l0:
seed dedup within a strategy_hash + lineage metadata in corpus JSONs.
Stdlib unittest, no model, no harness.
"""

from __future__ import annotations

import json
import tempfile
import types
import unittest
from pathlib import Path

from temper.steer.l0 import _persist_failing_seed


def _task(corpus_root: Path, strategy_hash: str = "abc123"):
    return types.SimpleNamespace(
        dir=corpus_root,
        name="unittest-task",
        strategy_hash=strategy_hash,
    )


def _row(i=0, seed=12345, temperature=0.4, alg="topk_margin", failing_seed=12345):
    return types.SimpleNamespace(
        i=i,
        temperature=temperature,
        alg=alg,
        seed=seed,
        minimal_input="[3, 1, 2]",
        reason="prop_fail",
        failing_seed=failing_seed,
    )


class TestCorpusDedupAndLineage(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.task = _task(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def _corpus(self):
        return self.root / "corpus" / self.task.strategy_hash

    def test_first_persist_writes_lineage_metadata(self):
        path = _persist_failing_seed(self.task, _row(i=2), "20260101T000000Z",
                                     steps=128)
        payload = json.loads((self.root / path).read_text())
        self.assertEqual(payload["candidate_index"], 2)
        self.assertEqual(payload["particle_config"],
                         {"temperature": 0.4, "alg": "topk_margin", "steps": 128})
        self.assertEqual(payload["run_id"], "20260101T000000Z")
        self.assertEqual(payload["seed"], 12345)
        # legacy aliases still present for M1-era tooling
        self.assertEqual(payload["particle"], 2)
        self.assertEqual(payload["run"], "20260101T000000Z")

    def test_identical_seed_deduped_keeps_first_seen(self):
        p1 = _persist_failing_seed(self.task, _row(i=0), "20260101T000000Z",
                                   steps=96)
        p2 = _persist_failing_seed(self.task, _row(i=3), "20260102T000000Z",
                                   steps=128)
        self.assertEqual(p1, p2)                      # same file referenced
        files = list(self._corpus().glob("*.json"))
        self.assertEqual(len(files), 1)               # not rewritten
        # first-seen timestamp survives in the filename
        self.assertIn("20260101T000000Z", files[0].name)
        payload = json.loads(files[0].read_text())
        self.assertEqual(payload["run_id"], "20260101T000000Z")

    def test_different_seed_not_deduped(self):
        _persist_failing_seed(self.task, _row(i=0, failing_seed=1),
                              "20260101T000000Z", steps=96)
        p2 = _persist_failing_seed(self.task, _row(i=1, failing_seed=2),
                                   "20260101T000000Z", steps=96)
        files = sorted(self._corpus().glob("*.json"))
        self.assertEqual(len(files), 2)
        self.assertNotEqual(str(self.root / p2), str(files[0]))

    def test_dedup_scoped_to_strategy_hash(self):
        _persist_failing_seed(self.task, _row(i=0), "20260101T000000Z", steps=96)
        other = _task(self.root, strategy_hash="different-strategy")
        p2 = _persist_failing_seed(other, _row(i=1), "20260102T000000Z",
                                   steps=128)
        self.assertIn("different-strategy", p2)       # own corpus dir

    def test_no_failing_seed_persists_nothing(self):
        r = _row()
        r.failing_seed = None
        self.assertIsNone(_persist_failing_seed(self.task, r,
                                                "20260101T000000Z", steps=96))
        self.assertFalse(self._corpus().exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
