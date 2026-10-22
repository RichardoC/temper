"""D2 (steer-decision replay, schema v1.3) and D3 (NFE/verifier metering).

D2 acceptance: the replayer re-derives every decision from the log alone, and
a POISONED verdict breaks it. D3 acceptance: meters agree with hand-built fake
games, the envelope assert fires, and the v1.2 record shapes are preserved so
a steered record still exports/scores through the unchanged instrument.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from temper.model.dream import ParticleResult
from temper.steer.log import (
    SCHEMA_VERSION,
    ReplayMismatch,
    SteerGameLog,
    decide,
    persistence_of,
    read_game_log,
    replay_game,
)
from temper.steer.metering import (
    BudgetExceeded,
    CellMeters,
    GameMeters,
    aggregate_nfe,
)
from temper.steer.verdict import load_verdict_config


def _checkpoint(particle, step, *, informed=(), r0=(), stubs=(),
                is_final=False, live=4, remaining=100, ok=None):
    r1_ok = (not informed) if ok is None else ok
    return {
        "particle": particle, "resume_epoch": 0,
        "step": step, "is_final": is_final, "nfe_at": step,
        "committed_token_count": step, "canvas_prefix_sha256": f"{particle}-{step}",
        "mask_share": 0.0, "expected_mask_share": 0.0,
        "r0_flags": list(r0), "r2_stubs": list(stubs),
        "r1": {"ok": r1_ok, "stderr_sha256": "x", "error_codes": list(informed),
               "informed_codes": list(informed), "fragment_exact": True},
        "live_particles": live, "budget_remaining_nfe": remaining,
        "envelope_nfe": 1024,
    }


def _log_events(decisions):
    """A synthetic game log whose decision events come from `decide`."""
    cfg = load_verdict_config()
    events = [{"kind": "meta", "schema_version": SCHEMA_VERSION,
               "steps": 256, "verdict_config_sha256": cfg.sha256}]
    history = []
    for rec in decisions:
        history.append(rec)
        d = decide(rec, config=cfg,
                   persistence=persistence_of(history, rec["particle"],
                                              rec["resume_epoch"]))
        events.append(dict(rec, kind="checkpoint"))
        events.append({"kind": "decision", "particle": rec["particle"],
                       "resume_epoch": rec["resume_epoch"], "step": rec["step"],
                       "action": d.action, "reason": d.reason,
                       "nfe_at": rec["nfe_at"]})
    return events


class TestReplay(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_verdict_config()

    def test_schema_version_is_1_3(self):
        self.assertEqual(SCHEMA_VERSION, "1.3")

    def test_clean_game_replays(self):
        recs = [_checkpoint(0, 64), _checkpoint(0, 128),
                _checkpoint(0, 256, is_final=True, ok=True)]
        events = _log_events(recs)
        got = replay_game(events, self.cfg)
        self.assertEqual([d.action for d in got], ["continue"] * 3)

    def test_positive_evidence_kills_after_patience(self):
        """Informed evidence must persist ``cfg.patience_checkpoints``
        consecutive checkpoints before it kills — read from the FROZEN config,
        not hardcoded (D4 repinned it 2 -> 3)."""
        p = self.cfg.patience_checkpoints
        recs = [_checkpoint(0, 32 * (i + 1), informed=("E0277",))
                for i in range(p)]
        events = _log_events(recs)
        actions = [d.action for d in replay_game(events, self.cfg)]
        for i in range(p - 1):
            self.assertEqual(actions[i], "continue",
                             f"checkpoint {i}: patience not yet reached")
        self.assertEqual(actions[p - 1], "kill")

    def test_single_checkpoint_evidence_never_kills(self):
        recs = [_checkpoint(0, 64, informed=("E0277",)),
                _checkpoint(0, 128)]
        events = _log_events(recs)
        self.assertEqual([d.action for d in replay_game(events, self.cfg)],
                         ["continue", "continue"])

    def test_interrupted_evidence_run_resets_patience(self):
        """A clean checkpoint between two informed ones resets the run, so the
        kill never fires however long the log is."""
        recs = [_checkpoint(0, 64, informed=("E0277",)),
                _checkpoint(0, 128),
                _checkpoint(0, 192, informed=("E0277",)),
                _checkpoint(0, 256)]
        actions = [d.action for d in replay_game(_log_events(recs), self.cfg)]
        self.assertEqual(set(actions), {"continue"})

    def test_r0_fatal_flag_kills_immediately(self):
        recs = [_checkpoint(0, 64, r0=("prose_after_fence",))]
        got = replay_game(_log_events(recs), self.cfg)
        self.assertEqual(got[0].action, "kill")
        self.assertIn("r0_fatal", got[0].reason)

    def test_r2_committed_stub_kills_immediately(self):
        recs = [_checkpoint(0, 64, stubs=("todo!(",))]
        got = replay_game(_log_events(recs), self.cfg)
        self.assertEqual(got[0].action, "kill")
        self.assertIn("r2_committed_stub", got[0].reason)

    def test_final_checkpoint_never_kills(self):
        # the final screen makes a canvas INELIGIBLE, it does not kill it:
        # the game still has to emit something (design §3.3 item 2)
        recs = [_checkpoint(0, 256, is_final=True, informed=("E0277",))]
        got = replay_game(_log_events(recs), self.cfg)
        self.assertEqual(got[0].action, "continue")
        self.assertEqual(got[0].reason, "final_r1_dirty_ineligible")

    def test_poisoned_verdict_breaks_replay(self):
        """D2 acceptance: tamper a verdict -> replay mismatch.

        Case 1: a clean checkpoint is poisoned with an R0 fatal flag (an
        immediate kill), while the recorded decision still says continue.
        Case 2: two clean checkpoints are poisoned with informed evidence so
        the patience rule fires at the second one.
        """
        recs = [_checkpoint(0, 64)]
        events = _log_events(recs)
        for ev in events:
            if ev.get("kind") == "checkpoint":
                ev["r0_flags"] = ["prose_after_fence"]
        with self.assertRaises(ReplayMismatch):
            replay_game(events, self.cfg)

        # enough consecutive informed checkpoints to exceed the frozen
        # patience, so poisoning them MUST flip the decision to kill
        p = self.cfg.patience_checkpoints
        recs = [_checkpoint(0, 32 * (i + 1)) for i in range(p)]
        events = _log_events(recs)
        for ev in events:
            if ev.get("kind") == "checkpoint":
                ev["r1"]["ok"] = False
                ev["r1"]["informed_codes"] = ["E0277"]
                ev["r1"]["error_codes"] = ["E0277"]
        with self.assertRaises(ReplayMismatch):
            replay_game(events, self.cfg)

    def test_poisoned_decision_breaks_replay(self):
        recs = [_checkpoint(0, 64)]
        events = _log_events(recs)
        for ev in events:
            if ev.get("kind") == "decision":
                ev["action"] = "kill"
        with self.assertRaises(ReplayMismatch):
            replay_game(events, self.cfg)

    def test_dropped_decision_event_breaks_replay(self):
        recs = [_checkpoint(0, 64), _checkpoint(0, 128)]
        events = [e for e in _log_events(recs)
                  if not (e.get("kind") == "decision" and e.get("step") == 128)]
        with self.assertRaises(ReplayMismatch):
            replay_game(events, self.cfg)

    def test_persistence_is_recomputed_not_stored(self):
        recs = [_checkpoint(0, 64, informed=("E0277",)),
                _checkpoint(0, 128, informed=("E0277",))]
        self.assertEqual(persistence_of(recs, 0, 0), 2)
        self.assertEqual(persistence_of(recs[:1], 0, 0), 1)
        # a clean checkpoint in between resets the run
        mixed = recs[:1] + [_checkpoint(0, 96)] + recs[1:]
        self.assertEqual(persistence_of(mixed, 0, 0), 1)


class TestGameLogRoundTrip(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_log_writes_schema_and_replays_from_disk(self):
        cfg = load_verdict_config()
        path = self.tmp / "HumanEval_0.jsonl"
        log = SteerGameLog(path)
        log.meta(game="g", problem="HumanEval_0", steps=4, K=1,
                 verdict_config_sha256=cfg.sha256)
        recs = [_checkpoint(0, 2, informed=("E0277",)),
                _checkpoint(0, 4, informed=("E0277",), is_final=True,
                            ok=False)]
        for rec in recs:
            log.emit("checkpoint", **rec)
            d = decide(rec, config=cfg,
                       persistence=persistence_of(recs[:recs.index(rec) + 1],
                                                  0, 0))
            log.emit("decision", particle=0, resume_epoch=0, step=rec["step"],
                     action=d.action, reason=d.reason, nfe_at=rec["nfe_at"])
        log.done(nfe_used=4)
        events = read_game_log(path)
        self.assertEqual(events[0]["kind"], "meta")
        self.assertEqual(events[0]["schema_version"], SCHEMA_VERSION)
        self.assertEqual(events[-1]["kind"], "done")
        got = replay_game(events, cfg)
        self.assertEqual([d.action for d in got], ["continue", "continue"])

    def test_log_is_cleared_between_games(self):
        path = self.tmp / "g.jsonl"
        SteerGameLog(path).meta(game="a")
        self.assertEqual(len(read_game_log(path)), 1)
        SteerGameLog(path).meta(game="b")
        self.assertEqual(len(read_game_log(path)), 1)


class TestMetering(unittest.TestCase):
    def test_game_meter_accrues_and_caps(self):
        g = GameMeters(envelope_nfe=10, k=2)
        for _ in range(10):
            g.spend_model(0)
        self.assertEqual(g.nfe_model, 10)
        self.assertEqual(g.remaining_nfe, 0)
        with self.assertRaises(BudgetExceeded):
            g.spend_model(0)

    def test_killed_particle_stops_accruing(self):
        g = GameMeters(envelope_nfe=100, k=2)
        g.spend_model(0, 30)
        g.spend_model(1, 20)
        self.assertEqual(g.per_particle_nfe, {0: 30, 1: 20})
        self.assertEqual(g.nfe_model, 50)
        self.assertEqual(g.remaining_nfe, 50)

    def test_verifier_wall_is_not_nfe(self):
        g = GameMeters(envelope_nfe=100, k=1)
        g.spend_model(0, 10)
        g.spend_verifier(0.35, calls=4)
        self.assertEqual(g.nfe_model, 10)
        self.assertEqual(g.nfe_verifier_calls, 4)
        self.assertAlmostEqual(g.verifier_wall_s, 0.35)

    def test_verifier_fraction_uses_nominal_canvas_wall(self):
        g = GameMeters(envelope_nfe=1024, k=1)
        g.spend_verifier(10.0)
        frac = g.verifier_fraction_of_budget_wall
        self.assertLess(frac, 0.011)   # 10s of ~1060s
        self.assertGreater(frac, 0.009)

    def test_cell_envelope_assert_fires(self):
        cell = CellMeters(n_problems=2, envelope_nfe=10)
        for _ in range(2):
            g = GameMeters(envelope_nfe=10, k=1)
            g.spend_model(0, 10)
            cell.add(g)
        cell.assert_within_envelope()          # exactly at the ceiling: legal
        g3 = GameMeters(envelope_nfe=20, k=1)
        g3.spend_model(0, 5)
        cell.add(g3)
        with self.assertRaises(BudgetExceeded):
            cell.assert_within_envelope()

    def test_cell_record_is_v1_2_compatible(self):
        cell = CellMeters(n_problems=1, envelope_nfe=1024)
        g = GameMeters(envelope_nfe=1024, k=4)
        g.spend_model(0, 1024)
        g.spend_verifier(0.42, calls=8)
        cell.add(g)
        rec = cell.as_record()
        # the shapes scripts/m2_3_cells.py reads on a done line
        for key in ("nfe_total", "verifier_wall_s_total"):
            self.assertIn(key, rec)
        self.assertEqual(rec["nfe_total"], 1024)
        self.assertEqual(rec["nfe_envelope_ceiling"], 1024)

    def test_particle_result_carries_the_new_metering_fields(self):
        p = ParticleResult(index=0, nfe=7)
        self.assertEqual(p.nfe_total, 7)          # v1.2 nfe still wins
        self.assertEqual(p.nfe_model, None)       # not set by the unsteered path
        self.assertEqual(p.nfe_verifier_calls, 0)
        self.assertEqual(p.verifier_wall_s, 0.0)
        p.nfe_model = 9                            # naive path sets it explicitly
        self.assertEqual(p.nfe_total, 7)          # nfe is authoritative

    def test_aggregate_nfe_accepts_engine_particles(self):
        parts = [ParticleResult(index=i, nfe=4) for i in range(2)]
        parts[0].nfe_verifier_calls = 4
        parts[0].verifier_wall_s = 0.14
        agg = aggregate_nfe(parts)
        self.assertEqual(agg["nfe_total"], 8)
        self.assertEqual(agg["nfe_verifier_calls_total"], 4)
        self.assertAlmostEqual(agg["verifier_wall_s_total"], 0.14)

    def test_aggregate_reads_ar_particle_forwards(self):
        ar = SimpleNamespace(nfe=83, nfe_prefill=1, n_forward_total=83)
        agg = aggregate_nfe([ar])
        self.assertEqual(agg["nfe_total"], 83)   # AR nfe == generated tokens


if __name__ == "__main__":
    unittest.main()
