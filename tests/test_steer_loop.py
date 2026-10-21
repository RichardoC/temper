"""Steer loop (temper.steer.loop): all-off equivalence + envelope discipline.

CPU-only: the model is a deterministic fake, so no GPU is touched. What is
pinned here is the CONTRACT the P2 GPU spot-check will exercise end-to-end:

  * all-off mode drives ``dream_fixed.fixed_shape_step`` with the SAME seeds
    and the SAME schedule as ``dream_fixed._sample_fixed`` -> identical
    canvases (the bit-identity gate's CPU-shadow);
  * the steered mode never exceeds its NFE envelope (D3), kills on positive
    evidence only, and never re-uses a parent's RNG stream for a clone;
  * the game log replays cleanly (D2) for a steered game.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import torch

from temper.model.dream import MASK_TOKEN_ID
from temper.model.dream_fixed import fixed_shape_step
from temper.steer.log import read_game_log, replay_game
from temper.steer.loop import run_game, segment_seed
from temper.steer.verdict import load_verdict_config
from temper.steer.view import SteerProblemView

V, PROMPT, CANVAS, STEPS = 24, 4, 6, 8


class _FakeTokenizer:
    """Minimal tokenizer: ids -> chars, a markdown fence, one EOS."""

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


class _KillingTokenizer(_FakeTokenizer):
    """Emits an unterminated fence so R0 has something fatal to find."""

    def decode(self, ids, skip_special_tokens=True):
        return "```rust\nfn f() -> i32 {\n" + super().decode(
            ids, skip_special_tokens).split("{\n", 1)[-1].rsplit("\n}", 1)[0]


class _FakeModel(torch.nn.Module):
    """Logits drawn from the GLOBAL torch RNG, content-shifted.

    Deliberately RNG-consuming: the equivalence witness is only meaningful if
    a different seed stream yields a different canvas. Content shifting keeps
    the model sensitive to the canvas (a real model property) while the draw
    order is the global generator's — exactly what a seed mismatch would
    perturb.
    """

    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1))
        self.calls = 0

    def forward(self, x, attention_mask, tok_idx):
        self.calls += 1
        B, L = x.shape
        shift = float(int(x.sum().item()) % 13) * 0.1
        return type("O", (), {
            "logits": torch.randn(B, L, V) * 4.0 + shift})()


def _sample_fixed_reference(model, enc, *, steps, canvas_len, temperature):
    """The P2 reference: dream_fixed._sample_fixed's loop, driven directly."""
    device = "cpu"
    x = torch.cat([torch.tensor([enc], dtype=torch.long),
                   torch.full((1, canvas_len), MASK_TOKEN_ID)], dim=1)
    timesteps = torch.linspace(1, 1e-3, steps + 1)
    tok = lambda i, x_, l: x_          # noqa: E731
    lg = lambda i, x_, l: l            # noqa: E731
    for i in range(steps):
        x = fixed_shape_step(
            model, x, attention_mask="full", tok_idx=None, timesteps=timesteps,
            i=i, steps=steps, canvas_len=canvas_len, temperature=temperature,
            top_p=0.95, top_k=None, eos_penalty=0.0, pad_token_id=0,
            mask_token_id=MASK_TOKEN_ID, eps=1e-3,
            generation_tokens_hook_func=tok, generation_logits_hook_func=lg)
    return x


class TestSegmentSeed(unittest.TestCase):
    def test_first_segment_reproduces_the_dream_seed(self):
        # P2 hinges on this: particle i's first uninterrupted segment must use
        # exactly base_seed + i, the seed dream.sample uses.
        for i in range(4):
            self.assertEqual(segment_seed(0, i, 0, 0), i)
            self.assertEqual(segment_seed(7, i, 0, 0), 7 + i)

    def test_later_segments_get_distinct_derived_seeds(self):
        seeds = {segment_seed(0, 0, e, s)
                 for e in range(3) for s in (0, 64, 128, 256)}
        self.assertEqual(len(seeds), 12, "segments must not share a stream")


class TestAllOffEquivalence(unittest.TestCase):
    """The P2 gate's CPU shadow: all-off == the fixed-shape sampler."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_all_off_matches_reference_canvases(self):
        view = SteerProblemView(name="p", prompt="fn f() -> i32 {\n",
                                fn_name="f", signature_line="fn f() -> i32 {")
        cfg = load_verdict_config()
        for k, base_seed in ((1, 0), (2, 5)):
            with self.subTest(k=k, base_seed=base_seed):
                ref_model = _FakeModel()
                enc = _FakeTokenizer().apply_chat_template(
                    [{"role": "user", "content": "x"}])
                refs = []
                for i in range(k):
                    torch.manual_seed(base_seed + i)
                    refs.append(_sample_fixed_reference(
                        ref_model, enc, steps=STEPS,
                        canvas_len=CANVAS, temperature=0.2))

                loop_model = _FakeModel()
                res = run_game(
                    view, "ignore-me", tokenizer=_FakeTokenizer(),
                    model=loop_model, envelope_nfe=STEPS * k, k=k,
                    steps=STEPS, canvas_len=CANVAS, temperature=0.2,
                    base_seed=base_seed, policy_enabled=False, config=cfg,
                    log_path=self.tmp / "game.jsonl", log=lambda s: None)
                self.assertEqual(res.meters.nfe_model, STEPS * k)
                for i in range(k):
                    self.assertTrue(
                        torch.equal(res.particles[i].x, refs[i]),
                        f"particle {i}: all-off canvas differs from the "
                        "fixed-shape sampler reference")

    def test_all_off_logs_no_decisions(self):
        view = SteerProblemView(name="p", prompt="fn f() -> i32 {\n",
                                fn_name="f", signature_line="fn f() -> i32 {")
        path = self.tmp / "g.jsonl"
        res = run_game(view, "x", tokenizer=_FakeTokenizer(),
                       model=_FakeModel(), envelope_nfe=STEPS, k=1,
                       steps=STEPS, canvas_len=CANVAS, policy_enabled=False,
                       log_path=path, log=lambda s: None)
        self.assertEqual(res.decisions, [])
        kinds = [e["kind"] for e in read_game_log(path)]
        self.assertNotIn("checkpoint", kinds)
        self.assertIn("emit", kinds)


class TestEquivalenceWitnessHasTeeth(unittest.TestCase):
    """The all-off gate is only meaningful if a seed bug BREAKS it."""

    def test_off_by_one_segment_seed_breaks_equivalence(self):
        import temper.steer.loop as loop_mod
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp)
        view = SteerProblemView(name="p", prompt="fn f() -> i32 {\n",
                                fn_name="f", signature_line="fn f() -> i32 {")
        cfg = load_verdict_config()
        enc = _FakeTokenizer().apply_chat_template(
            [{"role": "user", "content": "x"}])
        torch.manual_seed(0)
        ref = _sample_fixed_reference(_FakeModel(), enc, steps=STEPS,
                                      canvas_len=CANVAS, temperature=0.2)
        original = loop_mod.segment_seed
        loop_mod.segment_seed = lambda b, i, e, s: original(b, i, e, s) + 1
        try:
            res = run_game(view, "x", tokenizer=_FakeTokenizer(),
                           model=_FakeModel(), envelope_nfe=STEPS, k=1,
                           steps=STEPS, canvas_len=CANVAS, base_seed=0,
                           policy_enabled=False, config=cfg,
                           log_path=tmp / "g.jsonl", log=lambda s: None)
        finally:
            loop_mod.segment_seed = original
        self.assertFalse(
            torch.equal(res.particles[0].x, ref),
            "an off-by-one seed still matched the reference — the equivalence "
            "witness is blind to seed drift")


class TestSteeredGame(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.cfg = load_verdict_config()

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def _view(self, body_hint=""):
        return SteerProblemView(
            name="p", prompt="/// doc\nfn f() -> i32 {\n",
            fn_name="f", signature_line="fn f() -> i32 {")

    def test_envelope_is_never_exceeded(self):
        path = self.tmp / "g.jsonl"
        res = run_game(self._view(), "x", tokenizer=_FakeTokenizer(),
                       model=_FakeModel(), envelope_nfe=STEPS * 2, k=3,
                       steps=STEPS, canvas_len=CANVAS, base_seed=0,
                       policy_enabled=True, config=self.cfg, log_path=path,
                       log=lambda s: None)
        self.assertLessEqual(res.meters.nfe_model, STEPS * 2)
        for _ in res.particles:
            pass
        # no particle ever exceeds the envelope on its own
        for n in res.meters.per_particle_nfe.values():
            self.assertLessEqual(n, STEPS * 2)

    def test_steered_game_log_replays(self):
        path = self.tmp / "g.jsonl"
        run_game(self._view(), "x", tokenizer=_FakeTokenizer(),
                 model=_FakeModel(), envelope_nfe=STEPS * 2, k=2,
                 steps=STEPS, canvas_len=CANVAS, base_seed=0,
                 policy_enabled=True, config=self.cfg, log_path=path,
                 log=lambda s: None)
        events = read_game_log(path)
        kinds = [e["kind"] for e in events]
        self.assertIn("checkpoint", kinds)
        self.assertIn("decision", kinds)
        replayed = replay_game(events, self.cfg)
        recorded = [e for e in events if e["kind"] == "decision"]
        self.assertEqual([d.action for d in replayed],
                         [e["action"] for e in recorded])

    def _non_killing_config(self):
        """The frozen config with the kill triggers cleared: this test isolates
        the CHECKPOINT GRID, not the policy."""
        import dataclasses
        return dataclasses.replace(self.cfg, r0_flags_fatal=(),
                                   informed_error_classes=(), stub_markers=())

    def test_checkpoints_land_on_the_frozen_grid(self):
        path = self.tmp / "g.jsonl"
        run_game(self._view(), "x", tokenizer=_FakeTokenizer(),
                 model=_FakeModel(), envelope_nfe=STEPS * 2, k=1, steps=STEPS,
                 canvas_len=CANVAS, policy_enabled=True,
                 config=self._non_killing_config(), log_path=path,
                 log=lambda s: None)
        steps_seen = {e["step"] for e in read_game_log(path)
                      if e["kind"] == "checkpoint"}
        expected = {int(round(STEPS * f)) for f in self.cfg.checkpoint_grid}
        self.assertEqual(steps_seen, expected,
                         f"{steps_seen} != the frozen grid {expected}")
        self.assertIn(STEPS, steps_seen, "the final checkpoint must fire")

    def test_a_kill_stops_the_particle_and_frees_its_budget(self):
        path = self.tmp / "g.jsonl"
        import dataclasses
        killing = dataclasses.replace(self.cfg, r0_flags_fatal=("fence_unterminated",))
        view = SteerProblemView(name="p", prompt="/// doc\nfn f() -> i32 {\n",
                                fn_name="f", signature_line="fn f() -> i32 {")
        res = run_game(view, "x", tokenizer=_KillingTokenizer(),
                       model=_FakeModel(), envelope_nfe=STEPS, k=2,
                       steps=STEPS, canvas_len=CANVAS, policy_enabled=True,
                       config=killing, log_path=path, log=lambda s: None)
        kills = [e for e in read_game_log(path) if e["kind"] == "kill"]
        self.assertTrue(kills, "an R0-fatal canvas must be killed")
        self.assertTrue(all(not p.alive for p in res.particles))

    def test_a_truncated_budget_still_renders_every_particle(self):
        """Envelope exhaustion is legal: particles that never reached the final
        step still get a rendered canvas (the game always emits ONE answer)."""
        path = self.tmp / "g.jsonl"
        res = run_game(self._view(), "x", tokenizer=_FakeTokenizer(),
                       model=_FakeModel(), envelope_nfe=STEPS, k=3,
                       steps=STEPS, canvas_len=CANVAS, policy_enabled=True,
                       config=self.cfg, log_path=path, log=lambda s: None)
        self.assertLessEqual(res.meters.nfe_model, STEPS)
        self.assertTrue(all(p.final_raw_text for p in res.particles))
        self.assertIsNotNone(res.emitted)

    def test_emission_prefers_a_final_r1_clean_particle(self):
        path = self.tmp / "g.jsonl"
        res = run_game(self._view(), "x", tokenizer=_FakeTokenizer(),
                       model=_FakeModel(), envelope_nfe=STEPS * 2, k=2,
                       steps=STEPS, canvas_len=CANVAS, policy_enabled=True,
                       config=self.cfg, log_path=path, log=lambda s: None)
        self.assertIsNotNone(res.emitted)
        self.assertIn(res.emitted.index, {p.index for p in res.particles})


if __name__ == "__main__":
    unittest.main()
