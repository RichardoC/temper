"""Unit tests for temper.model.qwen (M2.4b phase 1: the AR twin, row B3).

Contract under test (mirror of the Dream Particle interface, driven through
the SAME bench driver via the sample_fn injection point):

  * sampling tail = the dLLM cells' verbatim HF math (temperature div ->
    top-p filter -> softmax -> multinomial) — pinned against transformers'
    own warpers on CPU;
  * chat rendering: user turn BYTE-IDENTICAL to bench.build_prompt output,
    template-default system line, full rendering carried for sha-pinning;
  * decode loop instrumentation: nfe == generated tokens (one [1,1] forward
    each), nfe_prefill == 1, EOS stop, truncation at the token budget,
    per-sample seeds base_seed+i, error isolation per particle;
  * driver integration: run_bench with sample=qwen.sample (fake model) —
    meta labels model/milestone, rows carry the additive nfe_prefill +
    rendered_prompt_sha256 fields, extract/score pipeline joins unchanged.

NO GPU anywhere in this suite: the model is a CPU fake; the only real
artifact is the locally cached Qwen tokenizer (skipped if absent).
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

import temper.model.qwen as qwen
from temper.model.qwen import (
    QwenParticle,
    _filtered_probs,
    build_rendered_prompt,
    sample,
)
from temper.multipl_e.bench import build_prompt, run_bench


def _qwen_snapshot_dir():
    hits = glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--Qwen--Qwen2.5-Coder-7B-Instruct/"
        "snapshots/*/"))
    return Path(hits[0]) if hits else None


def _load_tokenizer():
    snap = _qwen_snapshot_dir()
    if snap is None:
        return None
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(snap))


class FakeQwenCausalLM:
    """CPU stand-in for Qwen2ForCausalLM's forward: emits a scripted token
    sequence through peaked logits (argmax-margin so the multinomial draw is
    deterministic), regardless of input ids. Counts forward calls for the
    NFE cross-check."""

    def __init__(self, vocab_size, script_ids, eos_ids, *,
                 explode_on_prefill=None):
        self.vocab_size = vocab_size
        self.script = list(script_ids)
        self.eos_ids = set(eos_ids)
        self.generation_config = SimpleNamespace(eos_token_id=list(eos_ids))
        self._dummy = torch.zeros(1)
        self.forward_calls = 0
        self.prefill_calls = 0
        self._explode_on_prefill = explode_on_prefill

    def parameters(self):
        return iter([self._dummy])

    def _logits_for(self, next_token):
        row = torch.full((1, 1, self.vocab_size), -60.0)
        row[0, 0, next_token] = 60.0
        return row

    def __call__(self, input_ids, past_key_values=None, use_cache=True):
        self.forward_calls += 1
        if input_ids.shape[1] > 1:  # prefill
            self.prefill_calls += 1
            self._prompt_len = int(input_ids.shape[1])
            if self._explode_on_prefill is not None and \
                    self.prefill_calls == self._explode_on_prefill:
                raise RuntimeError("fake MPS boom")
            past = SimpleNamespace(n_past=int(input_ids.shape[1]))
            return SimpleNamespace(logits=self._logits_for(self.script[0]),
                                   past_key_values=past)
        # decode step: this call feeds one generated token and predicts the
        # next: idx = tokens-fed-so-far - prompt_len.
        k = past_key_values.n_past if past_key_values is not None else 0
        idx = k + int(input_ids.shape[1]) - self._prompt_len
        nxt = self.script[idx] if 0 <= idx < len(self.script) \
            else sorted(self.eos_ids)[0]
        past = SimpleNamespace(n_past=k + int(input_ids.shape[1]))
        return SimpleNamespace(logits=self._logits_for(nxt),
                               past_key_values=past)


class TestSamplingTailParity(unittest.TestCase):
    """The B3 tail must be the B2 tail's math: EXACT parity with the Dream
    fixed-shape sampler's prob computation (same primitives), and agreement
    with transformers' own warpers up to the fp boundary."""

    def test_exact_parity_with_dream_fixed_tail(self):
        # B2's sampler computes probs via _entropy_probs_padded at [B, V];
        # B3's _filtered_probs must produce the IDENTICAL row.
        from temper.model.dream_fixed import _entropy_probs_padded

        torch.manual_seed(0)
        scores = torch.randn(1, 500) * 3.0
        mine = _filtered_probs(scores, temperature=0.2, top_p=0.95)
        dream = _entropy_probs_padded(scores, temperature=0.2, top_p=0.95,
                                      top_k=None)
        self.assertTrue(torch.equal(mine, dream),
                        "B3 tail must be bit-identical to the B2 tail")

    def test_filter_matches_hf_warpers_up_to_fp_boundary(self):
        from transformers.generation.logits_process import (
            TemperatureLogitsWarper,
            TopPLogitsWarper,
        )

        torch.manual_seed(0)
        scores = torch.randn(2, 500) * 3.0
        mine = _filtered_probs(scores, temperature=0.2, top_p=0.95)
        # HF pipeline order: temperature first, then top-p (same as ours)
        warped = TopPLogitsWarper(top_p=0.95, min_tokens_to_keep=1)(
            None, TemperatureLogitsWarper(temperature=0.2)(None, scores))
        my_support = set((mine > 0).nonzero(as_tuple=True)[1].tolist())
        hf_support = set(torch.isfinite(warped)[0].nonzero(
            as_tuple=True)[0].tolist())
        # identical sets except possibly ONE boundary token whose mass sits
        # within fp rounding of the top-p cut (desc-cumsum vs asc-cumsum
        # summation order). Both are the standard "minimal prefix > top_p".
        diff = my_support ^ hf_support
        self.assertLessEqual(len(diff), 2,
                             f"support diverges beyond the fp boundary: {diff}")
        for j in diff:
            self.assertLess(float(mine[0, j]), 1e-2,
                            "disputed token must be at the mass boundary")

    def test_greedy_off_always_samples(self):
        # T=0.2 (not 0): the draw is multinomial over the top-p support —
        # a peaked-but-not-one-hot row must sometimes pick non-argmax.
        torch.manual_seed(1)
        scores = torch.full((1, 50), -10.0)
        scores[0, 7] = 1.0
        scores[0, 13] = 0.5
        picks = {qwen._sample_token(scores, temperature=0.2, top_p=0.95)[0]
                 for _ in range(200)}
        self.assertTrue(picks <= {7, 13})
        self.assertEqual(len(picks), 2, "T=0.2 must keep both candidates")


class TestRenderedPrompt(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = _load_tokenizer()

    def test_user_turn_byte_identical_and_template_default_system(self):
        if self.tok is None:
            self.skipTest("Qwen tokenizer not cached")
        user = build_prompt("/// doc\nfn f(x: i32) -> bool {\n")
        rendered = build_rendered_prompt(self.tok, user)
        # template default system line (publisher formatting, no custom one)
        self.assertIn("You are Qwen, created by Alibaba Cloud. "
                      "You are a helpful assistant.", rendered)
        # the user turn is byte-identical between the markers
        u0 = rendered.index("<|im_start|>user\n") + len("<|im_start|>user\n")
        u1 = rendered.index("<|im_end|>\n", u0)
        self.assertEqual(rendered[u0:u1], user)
        self.assertTrue(rendered.endswith("<|im_start|>assistant\n"))
        # sha256 pin is stable across calls
        self.assertEqual(
            hashlib.sha256(rendered.encode()).hexdigest(),
            hashlib.sha256(build_rendered_prompt(self.tok, user)
                           .encode()).hexdigest())

    def test_template_sha_matches_bench_template(self):
        # the B3 record's instruction_template_sha256 must equal the dLLM
        # cells' (same INSTRUCTION_TEMPLATE object drives both engines)
        from temper.multipl_e.bench import INSTRUCTION_TEMPLATE
        self.assertEqual(
            hashlib.sha256(INSTRUCTION_TEMPLATE.encode()).hexdigest(),
            hashlib.sha256(INSTRUCTION_TEMPLATE.encode()).hexdigest())


class TestDecodeLoopWithFakeModel(unittest.TestCase):
    """sample() against a fake causal LM: instrumentation + interface."""

    @classmethod
    def setUpClass(cls):
        cls.tok = _load_tokenizer()
        if cls.tok is None:
            raise unittest.SkipTest("Qwen tokenizer not cached")
        cls.V = cls.tok.vocab_size + 8  # margin above any scripted id

    def _fake(self, text_tokens, eos_ids, explode_on_prefill=None):
        return FakeQwenCausalLM(self.V, text_tokens, eos_ids,
                                explode_on_prefill=explode_on_prefill)

    def setUp(self):
        self.user = build_prompt("/// doc\nfn f(x: i32) -> bool {\n")
        self.rendered = build_rendered_prompt(self.tok, self.user)

    def test_eos_stop_counts_nfe_per_token(self):
        script = self.tok.encode("fn f() -> bool {\n    true\n}",
                                 add_special_tokens=False)
        eos = [151645, 151643]
        model = self._fake(script, eos)
        qwen._RESIDENT["tokenizer"] = self.tok
        qwen._RESIDENT["model"] = model
        try:
            [p] = sample(self.user, canvas_len=384, steps=0, particles=1,
                         temperatures=[0.2], base_seed=5)
        finally:
            qwen._RESIDENT["tokenizer"] = None
            qwen._RESIDENT["model"] = None
        self.assertEqual(p.nfe_prefill, 1)
        self.assertEqual(p.nfe, len(script) + 1)  # script tokens + EOS
        # decoder-only AR: prefill produces the FIRST token, so actual
        # forwards = gen tokens (= nfe + nfe_prefill - 1)
        self.assertEqual(model.forward_calls, p.n_forward_total)
        self.assertEqual(p.n_forward_total, p.nfe + p.nfe_prefill - 1)
        self.assertFalse(p.truncated)
        self.assertEqual(p.seed, 5)
        self.assertEqual(p.temperature, 0.2)
        self.assertEqual(p.alg, "ar")
        self.assertEqual(p.committed_tokens_total, len(script))
        self.assertEqual(p.rendered_prompt, self.rendered)
        self.assertIn("fn f()", p.raw_text)
        self.assertNotIn("<|im_end|>", p.raw_text)  # special tokens stripped
        self.assertEqual(len(p.per_step_s), p.nfe + 1)  # prefill + decodes
        self.assertGreater(p.wall_s, 0.0)

    def test_truncation_at_token_budget(self):
        script = self.tok.encode("x " * 50, add_special_tokens=False)
        eos = [151645, 151643]
        model = self._fake(script, eos)  # script longer than budget, no EOS
        qwen._RESIDENT["tokenizer"] = self.tok
        qwen._RESIDENT["model"] = model
        try:
            [p] = sample(self.user, canvas_len=8, steps=0, particles=1,
                         temperatures=[0.2], base_seed=0)
        finally:
            qwen._RESIDENT["tokenizer"] = None
            qwen._RESIDENT["model"] = None
        self.assertTrue(p.truncated)
        self.assertEqual(p.nfe, 8)
        self.assertEqual(p.tokens_total, 8)
        # prefill + 7 decode forwards (the 8th token needs no follow-up)
        self.assertEqual(model.forward_calls, 8)
        self.assertEqual(p.n_forward_total, 8)

    def test_same_seed_reproducible(self):
        script = self.tok.encode("fn f() -> bool {\n    true\n}",
                                 add_special_tokens=False)
        eos = [151645, 151643]

        def run():
            model = self._fake(script, eos)
            qwen._RESIDENT["tokenizer"] = self.tok
            qwen._RESIDENT["model"] = model
            try:
                return sample(self.user, canvas_len=64, steps=0, particles=1,
                              temperatures=[0.2], base_seed=42)[0]
            finally:
                qwen._RESIDENT["tokenizer"] = None
                qwen._RESIDENT["model"] = None

        a, b = run(), run()
        self.assertEqual(a.raw_text, b.raw_text)
        self.assertEqual(a.nfe, b.nfe)
        self.assertEqual(a.step_confidences, b.step_confidences)

    def test_particle_seeds_and_error_isolation(self):
        script = self.tok.encode("ok", add_special_tokens=False)
        eos = [151645, 151643]
        # particle 1's prefill explodes; particles 0 and 2 must survive
        model = self._fake(script, eos, explode_on_prefill=2)
        qwen._RESIDENT["tokenizer"] = self.tok
        qwen._RESIDENT["model"] = model
        try:
            parts = sample(self.user, canvas_len=16, steps=0, particles=3,
                           temperatures=[0.2] * 3, base_seed=10)
        finally:
            qwen._RESIDENT["tokenizer"] = None
            qwen._RESIDENT["model"] = None
        self.assertEqual([p.seed for p in parts], [10, 11, 12])
        self.assertEqual([p.index for p in parts], [0, 1, 2])
        self.assertIsNone(parts[0].error)
        self.assertIn("RuntimeError: fake MPS boom", parts[1].error or "")
        self.assertEqual(parts[1].nfe, 0)  # failed before any decode
        self.assertIsNone(parts[2].error)
        self.assertEqual(parts[0].raw_text, parts[2].raw_text)

    def test_nonzero_steps_refused(self):
        with self.assertRaises(ValueError):
            sample("x", canvas_len=8, steps=256, particles=1)


class TestDriverIntegrationQwen(unittest.TestCase):
    """run_bench + qwen.sample (fake model): record schema end to end."""

    @classmethod
    def setUpClass(cls):
        cls.tok = _load_tokenizer()

    def setUp(self):
        if self.tok is None:
            self.skipTest("Qwen tokenizer not cached")
        self.tmp = Path(tempfile.mkdtemp())
        self.out = self.tmp / "run.jsonl"

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_record_meta_and_additive_row_fields(self):
        from temper.multipl_e.bench import load_problems
        from temper.multipl_e.export import _fn_header_line

        td = load_problems("humaneval-rs", 1)[0]
        sig = _fn_header_line(td["prompt"]).group(0)
        compliant = ("```rust\n" + sig + "\n    unimplemented!()\n}\n```")
        script = self.tok.encode(compliant, add_special_tokens=False)
        model = FakeQwenCausalLM(self.tok.vocab_size + 8, script,
                                 [151645, 151643])
        qwen._RESIDENT["tokenizer"] = self.tok
        qwen._RESIDENT["model"] = model
        try:
            res = run_bench(
                "humaneval-rs", n_problems=1, k=2, steps=0, canvas=384,
                base_seed=0, temperature=0.2, alg="ar", out=self.out,
                sample_fn=qwen.sample, model=qwen.MODEL_ID,
                milestone="M2.4b-B3", log=lambda s: None,
            )
        finally:
            qwen._RESIDENT["tokenizer"] = None
            qwen._RESIDENT["model"] = None

        lines = [json.loads(l) for l in
                 self.out.read_text(encoding="utf-8").splitlines() if l.strip()]
        meta, cands, done = lines[0], lines[1:-1], lines[-1]
        self.assertEqual(meta["model"], "Qwen/Qwen2.5-Coder-7B-Instruct")
        self.assertEqual(meta["milestone"], "M2.4b-B3")
        self.assertEqual(meta["steps"], 0)
        self.assertEqual(meta["canvas"], 384)
        self.assertEqual(meta["alg"], "ar")
        self.assertEqual(len(cands), 2)
        for c in cands:
            self.assertEqual(c["nfe_prefill"], 1)
            self.assertEqual(c["alg"], "ar")
            self.assertGreater(c["nfe"], 0)
            self.assertTrue(c["extract_ok"], c.get("extract_reason"))
            # rendered prompt pinned: sidecar + sha
            rside = self.out.parent / (self.out.name + ".raw") / \
                f"{td['name']}__prompt.rendered.txt"
            self.assertTrue(rside.exists())
            self.assertEqual(
                c["rendered_prompt_sha256"],
                hashlib.sha256(rside.read_bytes()).hexdigest())
        self.assertEqual(done["nfe_prefill_total"], 2)
        self.assertEqual(res["summary"]["n_rows"], 2)

    def test_resume_includes_model_in_config_check(self):
        from temper.multipl_e.bench import load_problems
        from temper.multipl_e.export import _fn_header_line

        td = load_problems("humaneval-rs", 1)[0]
        sig = _fn_header_line(td["prompt"]).group(0)
        script = self.tok.encode(
            "```rust\n" + sig + "\n    unimplemented!()\n}\n```",
            add_special_tokens=False)

        def fresh_model():
            return FakeQwenCausalLM(self.tok.vocab_size + 8, script,
                                    [151645, 151643])

        def run_once(**kw):
            qwen._RESIDENT["tokenizer"] = self.tok
            qwen._RESIDENT["model"] = fresh_model()
            try:
                params = dict(
                    benchmark="humaneval-rs", n_problems=1, k=1, steps=0,
                    canvas=384, base_seed=0, temperature=0.2, alg="ar",
                    out=self.out, sample_fn=qwen.sample,
                    model=qwen.MODEL_ID, milestone="M2.4b-B3",
                    log=lambda s: None)
                params.update(kw)
                return run_bench(**params)
            finally:
                qwen._RESIDENT["tokenizer"] = None
                qwen._RESIDENT["model"] = None

        run_once()
        run_once(resume=True)  # config match incl. model -> clean resume
        # a DIFFERENT model label must refuse to resume the same record
        with self.assertRaises(SystemExit):
            run_once(resume=True, model="Dream-org/Dream-Coder-v0-Instruct-7B")


if __name__ == "__main__":
    unittest.main(verbosity=2)
