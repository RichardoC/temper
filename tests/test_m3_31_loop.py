"""M3.1 loop wiring: R3 fires at the final window and nothing else changes.

The engine cell (P-prop-engine) is 15 h of GPU, and the ONE thing that makes it
M3.1 rather than a re-run of C-full is that the emission ranking reads R3
evidence. These pins fix the wiring at the granularity the cell depends on:

  * R3 fires at the FINAL window only — once per R1-clean canvas, never once per
    checkpoint, never before every particle finished (design §2.3, D4's
    negative);
  * the R3 call count is exactly ``harnesses x R1-clean canvases`` — the
    pre-flight gate's arithmetic (design §3);
  * R3 never kills, clones or moves budget: a canvas the property rung judges
    BADLY still emits, and a game whose every canvas fails properties still
    completes its envelope (the demoted role, extended to the new rung);
  * R3 evidence actually reaches the emission decision — the property rule
    picks a canvas the confidence rule would not have;
  * a verifier crash leaves an abstention, never a scored failure.

The fake model/tokenizer are the ones the P2 equivalence gate uses, so no GPU
is touched; the R3 harnesses here are stubs whose *status* is injected, because
what is under test is the wiring, not cargo (``test_m3_31_property`` covers
cargo end to end).
"""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from temper.model.dream import MASK_TOKEN_ID
from temper.steer.log import read_game_log, replay_game
from temper.steer.loop import run_game
from temper.steer.emission import property_tier
from temper.steer.property import Harness, R3Call, R3Verdict, strategy_hash_of
from temper.steer.verdict import load_verdict_config
from temper.steer.view import SteerProblemView

V, PROMPT, CANVAS, STEPS = 24, 4, 6, 8


class _FakeTokenizer:
    eos_token_id = 2
    pad_token_id = 0

    def apply_chat_template(self, messages, tokenize=True,
                            add_generation_prompt=True):
        return [10, 11, 12, 13]

    def decode(self, ids, skip_special_tokens=True):
        table = {i: chr(ord("a") + i % 26) for i in range(V)}
        table[MASK_TOKEN_ID % V] = ""
        return "```rust\nfn f() -> i32 {\n" + "".join(
            table.get(int(i), "") for i in ids) + "\n}\n```"


class _FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x, attention_mask, tok_idx):
        B, L = x.shape
        shift = float(int(x.sum().item()) % 13) * 0.1
        return type("O", (), {
            "logits": torch.randn(B, L, V) * 4.0 + shift})()


VIEW = SteerProblemView(name="p", prompt="/// doc\nfn f() -> i32 {\n",
                        fn_name="f", signature_line="fn f() -> i32 {")


def _harness(slot: int) -> Harness:
    src = f"// stub harness {slot}\n"
    return Harness(problem="p", slot=slot, seed=slot, source=src,
                   strategy_hash=strategy_hash_of(src))


def _verdict(statuses) -> R3Verdict:
    return R3Verdict(tuple(
        R3Call(problem="p", harness_sha=f"h{i}", canvas="p", status=s,
               duration_s=0.01) for i, s in enumerate(statuses)))


class _R3Stub:
    """Stand-in for ``run_r3``: records every call, returns scripted statuses."""

    def __init__(self, statuses_for_canvas=None):
        self.calls = []
        self.statuses_for_canvas = statuses_for_canvas or (lambda ref: ("pass",))

    def __call__(self, program_text, harnesses, *, canvas_ref="", problem="",
                 cases=256, timeout_ms=2000, sandbox=None, workdir=None):
        self.calls.append({"canvas": canvas_ref, "n_harnesses": len(harnesses),
                           "cases": cases, "timeout_ms": timeout_ms})
        statuses = self.statuses_for_canvas(canvas_ref)
        # one call per harness, cycling the scripted statuses: the stub models
        # the SHAPE of a real verdict (n calls for n harnesses), so a pin on the
        # call count means the same thing it means on the cargo path
        return _verdict([statuses[i % len(statuses)]
                         for i in range(len(harnesses))])


class _Base(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        self.cfg = load_verdict_config()

    def _run(self, *, k=2, steps=STEPS, harnesses=None, path_name="g.jsonl",
             policy=True, r3=None, tokenizer=None, clean=None):
        """One game with R3's harness runner stubbed.

        The fake model emits nonsense, so R1 would never call any canvas clean
        and R3's pool would be empty by construction — which is correct
        behaviour but useless for testing the wiring. ``clean`` therefore stands
        in for the R1 screen: a bool (all canvases clean / none) or a callable
        ``(view, cfg, particle) -> bool``. Default: all clean.
        """
        path = self.tmp / path_name
        screen = (clean if clean is not None else True)
        screen_fn = (screen if callable(screen)
                     else (lambda view, cfg, p: bool(screen)))
        ctx = (mock.patch("temper.steer.loop.run_r3", r3) if r3 is not None
               else _nullcontext())
        with ctx, mock.patch("temper.steer.loop._final_is_clean",
                             side_effect=screen_fn):
            res = run_game(
                VIEW, "x", tokenizer=tokenizer or _FakeTokenizer(),
                model=_FakeModel(), envelope_nfe=steps * k, k=k, steps=steps,
                canvas_len=CANVAS, base_seed=0, policy_enabled=policy,
                config=self.cfg, harnesses=harnesses, log_path=path,
                log=lambda s: None)
        return res, read_game_log(path)

    def _all_clean(self, eligible):
        """Force every canvas R1-clean so R3 has a pool to fire on."""
        return True


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


class TestR3FiresAtTheFinalWindowOnly(_Base):
    def test_r3_fires_once_per_r1_clean_canvas_after_all_particles_finish(self):
        r3 = _R3Stub()
        res, events = self._run(k=3, harnesses=[_harness(0), _harness(1)], r3=r3)
        emit = next(e for e in events if e["kind"] == "emit")
        n_clean = len(emit["final_clean"])
        self.assertEqual(len(r3.calls), n_clean,
                         "R3 fires once per R1-clean canvas")
        self.assertEqual(sorted(r3.calls[0].keys()) if r3.calls else [],
                         ["canvas", "cases", "n_harnesses", "timeout_ms"])

    def test_the_pre_flight_arithmetic_holds(self):
        """The §3 pre-flight gate's quantity: R3 calls == harnesses x clean."""
        r3 = _R3Stub()
        res, events = self._run(k=4, harnesses=[_harness(0), _harness(1)], r3=r3)
        emit = next(e for e in events if e["kind"] == "emit")
        expected = 2 * len(emit["final_clean"])
        self.assertEqual(emit["r3_calls"], expected,
                         "logged r3_calls must equal harnesses x clean canvases")

    def test_no_r3_before_the_final_checkpoint(self):
        """R3 must never appear at a mid-canvas checkpoint (D4's negative).

        The demoted role fires the rung ladder at the final checkpoint only, so
        "no R3 before the last step" is implied by the step set — asserted
        directly so a future re-enable of mid-canvas rungs cannot carry R3
        along silently.
        """
        r3 = _R3Stub()
        _, events = self._run(k=2, harnesses=[_harness(0)], r3=r3)
        steps_seen = {e["step"] for e in events if e["kind"] == "checkpoint"}
        self.assertEqual(steps_seen, {STEPS})
        # every R3 call happened after the LAST checkpoint event
        kinds = [e["kind"] for e in events]
        last_checkpoint = max(i for i, k in enumerate(kinds) if k == "checkpoint")
        first_r3 = min((i for i, k in enumerate(kinds) if k == "r3"), default=None)
        if first_r3 is not None:
            self.assertGreater(first_r3, last_checkpoint,
                               "R3 must fire after the final checkpoint, during "
                               "the emission window")

    def test_a_dirty_canvas_is_never_r3ed(self):
        """§2.3: R3 requires a completed, zero-hole, R1-clean canvas. Mock the
        R1 screen so only one particle is clean and check the pool."""
        r3 = _R3Stub()
        res, events = self._run(k=3, harnesses=[_harness(0)], r3=r3,
                                clean=lambda view, cfg, p: p.index == 1)
        emit = next(e for e in events if e["kind"] == "emit")
        self.assertEqual(emit["final_clean"], [1])
        self.assertEqual(len(r3.calls), 1)
        self.assertEqual(r3.calls[0]["canvas"], "p1")

    def test_the_configured_case_count_reaches_r3(self):
        r3 = _R3Stub()
        self._run(k=1, harnesses=[_harness(0)], r3=r3)
        self.assertEqual(r3.calls[0]["cases"], 256,
                         "the frozen 256-case count must reach the runner")

    def test_no_harnesses_means_no_r3_and_says_so(self):
        r3 = _R3Stub()
        _, events = self._run(k=2, harnesses=None, r3=r3)
        emit = next(e for e in events if e["kind"] == "emit")
        self.assertFalse(emit["r3_available"])
        self.assertEqual(emit["r3_calls"], 0)
        self.assertEqual(r3.calls, [])


class TestR3CannotKillCloneOrMoveBudget(_Base):
    """Regression pin: R3 is an EMISSION-ONLY rung (design §2.3/§0)."""

    def test_a_canvas_that_fails_every_property_still_emits_and_completes(self):
        r3 = _R3Stub(statuses_for_canvas=lambda ref: ("fail", "fail"))
        res, events = self._run(k=3, harnesses=[_harness(0), _harness(1)], r3=r3)
        kinds = [e["kind"] for e in events]
        self.assertNotIn("kill", kinds,
                         "R3 evidence must never execute a kill")
        self.assertEqual(res.meters.nfe_model, STEPS * 3,
                         "a property failure must not truncate the envelope")
        self.assertTrue(all(p.alive for p in res.particles))
        self.assertIsNotNone(res.emitted,
                             "the game still emits one answer (§3.3 item 2)")

    def test_property_evidence_never_touches_the_decision_log(self):
        """The D2 decisions must be identical with and without R3 — R3 is not an
        input to the policy, only to the emission ranking."""
        r3 = _R3Stub(statuses_for_canvas=lambda ref: ("fail",))
        _, with_r3 = self._run(k=2, harnesses=[_harness(0)], r3=r3,
                               path_name="a.jsonl")
        _, without = self._run(k=2, harnesses=None, path_name="b.jsonl")
        dec = lambda evs: [(e["particle"], e["step"], e["action"], e["reason"])
                           for e in evs if e["kind"] == "decision"]
        self.assertEqual(dec(with_r3), dec(without))

    def test_budget_meters_are_untouched_by_r3(self):
        """R3 is verifier work, never model NFE (D3)."""
        r3 = _R3Stub()
        res, events = self._run(k=2, harnesses=[_harness(0)], r3=r3)
        done = next(e for e in events if e["kind"] == "done")
        meters = done["meters"]
        self.assertEqual(meters["nfe_model"], STEPS * 2)
        self.assertGreater(meters["nfe_verifier_calls"], 0,
                           "R3 calls are metered on the verifier axis")


class TestR3DrivesTheEmission(_Base):
    def test_the_property_rule_picks_a_canvas_confidence_would_not(self):
        """THE wiring claim: R3 evidence re-ranks the ensemble.

        The fake model's confidences are not controllable, so the pin states the
        claim in the form that holds for EVERY draw: the emitted canvas is the
        property rule's own argmin over the logged evidence, and any divergence
        from the v2 screen's pick is property-attributed (the canvas that was
        emitted passed its properties and the one the screen wanted did not).
        That is exactly the flip G2 counts.
        """
        def statuses(ref):
            return ("fail",) if ref == "p0" else ("pass",)

        r3 = _R3Stub(statuses_for_canvas=statuses)
        res, events = self._run(k=2, harnesses=[_harness(0)], r3=r3)
        emit = next(e for e in events if e["kind"] == "emit")
        self.assertEqual(emit["emission_rule"],
                         "property_pass_rate_then_mean_commit_conf_ties_lowest_temp")
        # the emitted canvas is the property rule's pick over the logged evidence
        logged = [{"index": int(i),
                   "temperature": emit["temperatures"][i],
                   "mean_commit_conf": emit["mean_commit_conf"][i],
                   "property_pass_rate": emit["property_pass_rate"].get(i)}
                  for i in emit["final_clean"] and
                  [str(j) for j in emit["final_clean"]]]
        ranked = sorted(
            logged,
            key=lambda c: (property_tier(type("C", (), c)()),
                           -(c["property_pass_rate"] or 0.0),
                           -(c["mean_commit_conf"] if c["mean_commit_conf"]
                             is not None else -1e9),
                           c["temperature"], c["index"]))
        self.assertEqual(emit["particle"], ranked[0]["index"])
        # and it passes properties: the rule never emits a canvas it measured bad
        # while a passing one existed
        self.assertEqual(emit["property_pass_rate"][str(emit["particle"])], 1.0)
        # any divergence from the screen is property-attributed by construction
        if emit["screen_selected"] != emit["particle"]:
            self.assertEqual(
                emit["property_pass_rate"][str(emit["screen_selected"])], 0.0,
                "a divergence from the v2 screen must be the property verdict's "
                "doing — the screen's canvas was measured BAD")

    def test_the_emit_event_names_the_rule_and_both_picks(self):
        r3 = _R3Stub()
        _, events = self._run(k=2, harnesses=[_harness(0)], r3=r3)
        emit = next(e for e in events if e["kind"] == "emit")
        self.assertEqual(emit["emission_rule"], self.cfg.emission)
        self.assertIn("screen_selected", emit)
        self.assertIn("emission_ranking", emit)
        self.assertIn("property_pass_rate", emit)
        self.assertTrue(emit["r3_fired"])

    def test_r3_events_carry_the_d2_schema_fields(self):
        """Design §2.4: D2 logs every R3 call (problem, harness sha, canvas
        ref, status, duration, minimal_input)."""
        r3 = _R3Stub()
        _, events = self._run(k=1, harnesses=[_harness(0), _harness(1)], r3=r3)
        r3_events = [e for e in events if e["kind"] == "r3"]
        self.assertTrue(r3_events)
        for ev in r3_events:
            for key in ("problem", "harness_sha", "canvas", "status",
                        "duration_s", "minimal_input"):
                self.assertIn(key, ev)

    def test_an_all_abstained_canvas_is_logged_as_its_own_class(self):
        """§7 Q1: abstention stays measurable, so G4 can weigh it."""
        r3 = _R3Stub(statuses_for_canvas=lambda ref: ("timeout", "timeout"))
        _, events = self._run(k=1, harnesses=[_harness(0), _harness(1)], r3=r3)
        self.assertIn("r3_abstain", [e["kind"] for e in events])

    def test_a_verifier_crash_is_an_abstention_not_a_failure(self):
        def boom(*a, **kw):
            raise RuntimeError("cargo vanished")

        r3 = boom
        res, events = self._run(k=1, harnesses=[_harness(0)], r3=r3)
        kinds = [e["kind"] for e in events]
        self.assertIn("r3_error", kinds)
        self.assertIsNone(res.particles[0].property_pass_rate,
                          "a verifier crash leaves no evidence")
        self.assertIsNotNone(res.emitted)


class TestEquivalenceModeUnaffected(_Base):
    def test_all_off_never_fires_r3(self):
        r3 = _R3Stub()
        res, events = self._run(k=1, harnesses=[_harness(0)], r3=r3,
                                policy=False)
        self.assertEqual(r3.calls, [],
                         "the equivalence mode must remain rung-free")
        self.assertNotIn("r3", [e["kind"] for e in events])

    def test_a_steered_r3_game_still_replays(self):
        r3 = _R3Stub()
        _, events = self._run(k=2, harnesses=[_harness(0)], r3=r3)
        decisions = replay_game(events, self.cfg)
        recorded = [e["action"] for e in events if e["kind"] == "decision"]
        self.assertEqual([d.action for d in decisions], recorded)


if __name__ == "__main__":
    unittest.main()
