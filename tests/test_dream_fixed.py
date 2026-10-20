"""Tests for temper.model.dream_fixed (M2.3 phase-3 MPS leak fix).

CPU-only: verifies the fixed-shape (padded) entropy pipeline is
generation-behavior-neutral vs the native gathered-rows formula — same probs,
same confidence, same x0 draws for real rows (multinomial row-prefix property),
plus the tie-free order construction and the dispatch/kill-switch behavior.
The MPS end-to-end gate is the s128 bit-identity spot-check (see
docs/m2.3-phase3-forensics.md); these unit tests pin the math and scoping.
"""
from __future__ import annotations

import os
import types
import unittest
from unittest import mock

import torch
import torch.distributions as dists

from temper.model.dream_fixed import (
    _entropy_probs_padded,
    _entropy_sample_tokens_padded,
    _top_k_logits,
    _top_p_logits,
    apply_fixed_shape_sampler,
)

V = 64
L = 24
B = 12


def _masked_rows_logits(seed: int, n_mask: int):
    """Random bf16-scale logits [1, L, V] with the first n_mask canvas rows masked."""
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(1, L, V, generator=g).bfloat16().float()
    is_mask = torch.zeros(L, dtype=torch.bool)
    is_mask[2 : 2 + n_mask] = True  # canvas region, ascending positions
    return logits, is_mask


def _native_gather_formula(logits: torch.Tensor, is_mask: torch.Tensor,
                           temperature: float, top_p: float):
    """The vendored sample_tokens path evaluated exactly as native: gather masked
    rows (ascending), then the op sequence."""
    mask_rows = is_mask.nonzero(as_tuple=True)[0]
    mask_logits = logits[0, mask_rows]  # [n, V]
    original_dtype = mask_logits.dtype
    x = mask_logits.to(torch.float32)
    if temperature > 0:
        x = x / temperature
    x = _top_p_logits(x, top_p)
    probs = torch.softmax(x, dim=-1)
    log_probs = torch.log(probs + 1e-10)
    confidence = torch.sum(probs * log_probs, dim=-1)
    return probs, confidence.to(original_dtype)


class TestFixedShapeValueEquivalence(unittest.TestCase):
    def _order_and_mask(self, is_mask: torch.Tensor):
        key = is_mask.long() * L - torch.arange(L)
        order_B = torch.topk(key, B).indices
        mask_sel = is_mask.long().gather(0, order_B).bool()
        return order_B, mask_sel

    def test_probs_confidence_x0_identical(self):
        # n_mask <= B always in production (masks only exist in the canvas
        # region and B == canvas); the sampler raises on n_mask > B.
        for n_mask in (1, 5, B):
            with self.subTest(n_mask=n_mask):
                logits, is_mask = _masked_rows_logits(7, n_mask)
                order_B, mask_sel = self._order_and_mask(is_mask)
                padded = logits[0].index_select(0, order_B)  # [B, V]

                probs_pad = _entropy_probs_padded(
                    padded, temperature=0.2, top_p=0.95, top_k=None)
                conf_pad, x0_pad = _entropy_sample_tokens_padded(
                    padded, temperature=0.2, top_p=0.95, top_k=None)

                probs_native, conf_native = _native_gather_formula(
                    logits, is_mask, 0.2, 0.95)
                torch.manual_seed(99)
                x0_native = dists.Categorical(probs=probs_native).sample()
                torch.manual_seed(99)
                x0_pad_draw = dists.Categorical(probs=probs_pad).sample()

                # real rows: identical per-row values (row-wise ops)
                self.assertTrue(torch.equal(probs_pad[mask_sel], probs_native))
                self.assertTrue(torch.equal(conf_pad[mask_sel], conf_native))
                # row-prefix property: draws for the real rows are identical
                self.assertTrue(torch.equal(x0_pad_draw[mask_sel], x0_native))
                # the mirror's own x0 (drawn over padded probs) matches too
                torch.manual_seed(99)
                conf2, x0_mirror = _entropy_sample_tokens_padded(
                    padded, temperature=0.2, top_p=0.95, top_k=None)
                self.assertTrue(torch.equal(x0_mirror[mask_sel], x0_native))

    def test_order_b_is_native_gather_order(self):
        _, is_mask = _masked_rows_logits(3, 5)
        order_B, mask_sel = self._order_and_mask(is_mask)
        native_rows = is_mask.nonzero(as_tuple=True)[0]
        self.assertTrue(torch.equal(order_B[:5], native_rows))  # ascending, complete
        self.assertTrue(torch.equal(
            mask_sel,
            torch.cat([torch.ones(5, dtype=torch.bool),
                       torch.zeros(B - 5, dtype=torch.bool)])))

    def test_top_k_mask_rowwise_identical(self):
        logits, is_mask = _masked_rows_logits(5, 6)
        mask_rows = is_mask.nonzero(as_tuple=True)[0]
        gathered = logits[0, mask_rows].to(torch.float32)
        padded_full = logits[0].to(torch.float32)
        # top_k applied per-row: gathered rows vs full padded tensor
        self.assertTrue(torch.equal(
            _top_k_logits(gathered, 10), _top_k_logits(padded_full, 10)[mask_rows]))

    def test_temperature_zero_branch(self):
        logits, is_mask = _masked_rows_logits(13, 4)
        order_B, mask_sel = self._order_and_mask(is_mask)
        padded = logits[0].index_select(0, order_B)
        probs_pad = _entropy_probs_padded(
            padded, temperature=0.0, top_p=0.95, top_k=None)
        conf_pad, x0_pad = _entropy_sample_tokens_padded(
            padded, temperature=0.0, top_p=0.95, top_k=None)
        probs_native, conf_native = _native_gather_formula(logits, is_mask, 0.0, 0.95)
        self.assertTrue(torch.equal(probs_pad[mask_sel], probs_native))
        self.assertTrue(torch.equal(conf_pad[mask_sel], conf_native))
        self.assertTrue(torch.equal(x0_pad[mask_sel], probs_native.argmax(dim=-1)))


class _FakeMixin:
    def _sample(self, *a, **k):  # sentinel native
        return "native"


class _FakeModel(_FakeMixin):
    pass


class TestDispatchAndScoping(unittest.TestCase):
    def test_override_is_instance_scoped_and_active(self):
        m1, m2 = _FakeModel(), _FakeModel()
        self.assertIs(apply_fixed_shape_sampler(m1), True)
        self.assertIs(m1._temper_fixed_sample_installed, True)
        # instance-scoped: m2 untouched
        self.assertFalse(hasattr(m2, "_temper_fixed_sample_installed"))
        # idempotent
        self.assertIs(apply_fixed_shape_sampler(m1), True)
        # installed attribute is a bound method of the fixed builder
        self.assertIsInstance(m1._sample, types.MethodType)
        # native sentinel still reachable on m2
        self.assertEqual(m2._sample(1), "native")

    def test_native_kill_switch(self):
        m = _FakeModel()
        with mock.patch.dict(os.environ, {"TEMPER_NATIVE_DREAM_SAMPLE": "1"}):
            self.assertIs(apply_fixed_shape_sampler(m), False)
        self.assertFalse(hasattr(m, "_temper_fixed_sample_installed"))


if __name__ == "__main__":
    unittest.main()
