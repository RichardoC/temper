"""Fixed-shape (padded) entropy sampling tail for Dream-Coder on MPS — M2.3 leak fix.

ROOT CAUSE (M2.3 phase-3 forensics, docs/m2.3-phase3-forensics.md): the vendored
Dream ``_sample`` loop gathers the masked rows every denoise step
(``logits[mask_index]`` -> ``[n_mask, V]``) and runs the whole sampling tail
(fp32 cast, temperature div, top-p sort/cumsum, top-k mask, softmax,
multinomial) at THAT shape. ``n_mask`` changes every step, so MPSGraph
recompiles ~20 graph executables per step and each compiled executable pins
driver memory. Measured on this host: MPS driver-allocated memory grows
~1-2 GB/step WITHIN one canvas (torch-side alloc flat at 14.8 GB, Python RSS
~1 GB — our hooks exonerated by a no-hooks probe), and macOS SIGKILLs the
process at a hard ~335 GB driver-allocation cap. steps=128 ends a canvas at
~250 GB (survives; shapes repeat across samples so the compile cache is
reused); steps=256 hits the cap at ~step 205. The 96/128 cells ran at 60-75%
of the cap (retroactively explains the M2.2-era system-wide memory incident).

FIX (this module): run the whole entropy tail at a FIXED padded shape
``[B, V]`` with B = canvas length (max possible masked rows). Every op then
compiles ONCE per canvas and the driver counter plateaus. Two invariants make
this generation-behavior-neutral (same seeds -> same tokens):

1. Every reshaped op is RNG-FREE and ROW-WISE (fp32 cast, temperature div,
   top-p sort/cumsum/mask, top-k mask, softmax, neg-entropy confidence):
   per-row results are identical to the gathered-rows computation by
   construction. The single RNG-consuming op (``Categorical.sample`` ->
   ``torch.multinomial(replacement=True)``) still runs — at the padded width.
   Its draws for real rows are preserved because torch.multinomial on MPS has
   the ROW-PREFIX property, empirically verified on this host for V=152064 and
   (n, B) in {(100,384), (383,384), (50,384), (384,512)}: with the same
   generator state, ``multinomial([B, V])[:n] == multinomial([n, V])``.
   Dummy padding rows are discarded before any write-back.

2. The [n]-shaped bookkeeping (which rows are real) is built with FIXED-shape
   ops: ``key = is_mask*L - arange(L)`` is strictly decreasing on masked
   positions and strictly smaller on unmasked ones, so ``topk(key, B)`` returns
   the masked positions in ascending order (the native gather order) followed
   by dummy positions — no ties, no stable-sort requirement, one compile.

Known accepted deviation: the per-step confidence STATISTIC (instrumentation
only — it never feeds sampling) is computed over padded rows; per-row values
are identical, so recorded confidences match up to fp32 summation-order ulps.
Tokens are unaffected; the s128 bit-identity spot-check is the gate.

Escaped valves: ``TEMPER_NATIVE_DREAM_SAMPLE=1`` restores the native path;
non-entropy algorithms, batch>1 calls and alg_temp-random calls delegate to
the native ``_sample`` unchanged. The override is applied per model INSTANCE
(never class-level).
"""
from __future__ import annotations

import os
import types
from typing import Callable, Optional

import torch
import torch.distributions as dists

__all__ = ["apply_fixed_shape_sampler", "fixed_shape_step"]


def _entropy_probs_padded(
    padded_logits: torch.Tensor,
    *,
    temperature: float,
    top_p: Optional[float],
    top_k: Optional[int],
) -> torch.Tensor:
    """Verbatim mirror of the vendored ``sample_tokens`` pre-sampling math
    (fp32 cast, temperature, top-p, top-k, softmax) evaluated at the FIXED
    padded shape [B, V]. Ops are row-wise, so per-row results equal the native
    gathered-rows computation. Returns probs [B, V] fp32.
    """
    logits = padded_logits.to(torch.float32)
    if temperature > 0:
        logits = logits / temperature
    if top_p is not None and top_p < 1:
        logits = _top_p_logits(logits, top_p)
    if top_k is not None:
        logits = _top_k_logits(logits, top_k)
    return torch.softmax(logits, dim=-1)


def _entropy_sample_tokens_padded(
    padded_logits: torch.Tensor,
    *,
    temperature: float,
    top_p: Optional[float],
    top_k: Optional[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Verbatim mirror of the vendored ``sample_tokens`` (generation_utils.py)
    with neg_entropy=True semantics, evaluated at the FIXED padded shape
    [B, V]. Returns (confidence[B] in input dtype, x0[B] long).
    """
    original_dtype = padded_logits.dtype
    probs = _entropy_probs_padded(
        padded_logits, temperature=temperature, top_p=top_p, top_k=top_k
    )

    if temperature > 0:
        x0 = dists.Categorical(probs=probs).sample()
        confidence = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)
    else:
        confidence, x0 = probs.max(dim=-1)

    # neg_entropy branch (the bench's alg): overwrites confidence, keeps x0.
    epsilon = 1e-10
    log_probs = torch.log(probs + epsilon)
    confidence = torch.sum(probs * log_probs, dim=-1)

    return confidence.to(original_dtype), x0


def _top_p_logits(logits, top_p=None):
    # Verbatim port of generation_utils.top_p_logits (RNG-free, row-wise).
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
    mask = mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)
    logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
    return logits


def _top_k_logits(logits, top_k=None):
    # Verbatim port of generation_utils.top_k_logits (RNG-free, row-wise).
    top_k = min(top_k, logits.size(-1))
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits = logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)
    return logits


def _make_fixed_sample(native_sample: Callable):
    """Build the fixed-shape ``_sample`` closing over the native fallback and
    the vendored module namespace (for DreamModelOutput)."""
    native_globals = native_sample.__globals__
    DreamModelOutput = native_globals.get("DreamModelOutput")  # may be None in
    # tests; only required when return_dict_in_generate=True (checked at call)

    def _sample_fixed(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor],
        generation_config,
        generation_tokens_hook_func,
        generation_logits_hook_func,
    ):
        output_history = generation_config.output_history
        return_dict_in_generate = generation_config.return_dict_in_generate
        max_length = generation_config.max_length
        mask_token_id = generation_config.mask_token_id
        pad_token_id = generation_config.pad_token_id
        steps = generation_config.steps
        eps = generation_config.eps
        alg = generation_config.alg
        alg_temp = generation_config.alg_temp
        temperature = generation_config.temperature
        top_p = generation_config.top_p
        top_k = generation_config.top_k
        eos_penalty = generation_config.eos_penalty

        # Fixed-shape path applies ONLY to: batch 1, entropy alg, no alg_temp
        # randomness (alg_temp 0/None). Everything else: native path unchanged.
        eligible = (
            alg == "entropy"
            and input_ids.shape[0] == 1
            and (alg_temp is None or alg_temp == 0)
        )
        if not eligible:
            return native_sample(
                self,
                input_ids,
                attention_mask=attention_mask,
                generation_config=generation_config,
                generation_tokens_hook_func=generation_tokens_hook_func,
                generation_logits_hook_func=generation_logits_hook_func,
            )

        histories = [] if (return_dict_in_generate and output_history) else None

        B = max_length - input_ids.shape[-1]  # canvas length = max masked rows
        device = input_ids.device

        # pad input_ids to max_length (native verbatim)
        x = torch.nn.functional.pad(
            input_ids, (0, max_length - input_ids.shape[1]), value=mask_token_id
        )

        if attention_mask is not None and torch.any(attention_mask == 0.0):
            attention_mask = torch.nn.functional.pad(
                attention_mask, (0, max_length - attention_mask.shape[1]), value=1.0
            )
            tok_idx = attention_mask.long().cumsum(-1) - 1
            tok_idx.masked_fill_(attention_mask == 0, 1)
            attention_mask = torch.logical_and(
                attention_mask.unsqueeze(1).unsqueeze(-2),
                attention_mask.unsqueeze(1).unsqueeze(-1),
            )
        else:
            tok_idx = None
            attention_mask = "full"

        timesteps = torch.linspace(1, eps, steps + 1, device=device)

        x = generation_tokens_hook_func(None, x, None)
        for i in range(steps):
            x = fixed_shape_step(
                self,
                x,
                attention_mask=attention_mask,
                tok_idx=tok_idx,
                timesteps=timesteps,
                i=i,
                steps=steps,
                canvas_len=B,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                eos_penalty=eos_penalty,
                pad_token_id=pad_token_id,
                mask_token_id=mask_token_id,
                eps=eps,
                generation_tokens_hook_func=generation_tokens_hook_func,
                generation_logits_hook_func=generation_logits_hook_func,
            )
            if histories is not None:
                histories.append(x.clone())

        if return_dict_in_generate:
            if DreamModelOutput is None:
                raise RuntimeError(
                    "DreamModelOutput not found in vendored generation module")
            return DreamModelOutput(
                sequences=x,
                history=histories,
            )
        return x

    return _sample_fixed


def fixed_shape_step(
    model,
    x: torch.Tensor,
    *,
    attention_mask,
    tok_idx,
    timesteps,
    i: int,
    steps: int,
    canvas_len: int,
    temperature: float,
    top_p: Optional[float],
    top_k: Optional[int],
    eos_penalty: float,
    pad_token_id: int,
    mask_token_id: int,
    eps: float,
    generation_tokens_hook_func,
    generation_logits_hook_func,
) -> torch.Tensor:
    """One full-canvas denoise step at FIXED padded shape.

    This is the shared body of ``_sample_fixed`` AND of the M3 steer loop
    (``temper.steer.loop``): the ops, their order, the tie-free masked-row
    selection and the hook call sites are exactly the ones the fixed-shape
    sampler always ran — extracted so the steer loop can drive ONE step of ONE
    particle without re-implementing (and silently drifting from) the tail.
    ``canvas_len`` is B, the maximum number of masked rows (= canvas length).
    """
    mask_index = x == mask_token_id  # [1, L]
    logits = model(x, attention_mask, tok_idx).logits
    logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

    logits = generation_logits_hook_func(i, x, logits)

    t = timesteps[i]
    s = timesteps[i + 1]

    B = canvas_len
    L = x.shape[1]
    arange_L = torch.arange(L, device=x.device)
    neg_inf = torch.tensor(-torch.inf, device=x.device)

    is_mask = mask_index[0]  # [L]
    n_mask = int(is_mask.sum())  # CPU sync (native has the same one)
    if n_mask > B:  # cannot happen for Dream prompts (masks only in canvas)
        raise RuntimeError(
            f"fixed-shape sampler: masked rows {n_mask} > canvas {B}"
        )

    # FIXED-shape row selection: masked positions in ascending order (native
    # gather order), then dummy positions. Tie-free by keys.
    key = is_mask.long() * L - arange_L
    order_B = torch.topk(key, B).indices  # [B]
    mask_sel = is_mask.long().gather(0, order_B).bool()  # [B]

    padded = logits[0].index_select(0, order_B)  # [B, V] FIXED shape
    # native adds the eos penalty to the gathered rows pre-sampling;
    # row-wise add, so applying it on padded rows matches on real rows.
    padded[:, pad_token_id] = padded[:, pad_token_id] + (
        eos_penalty * torch.log(1 - t + eps)
    )

    confidence_padded, x0_padded = _entropy_sample_tokens_padded(
        padded,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )

    full_confidence = torch.full_like(
        x, -torch.inf, device=x.device, dtype=logits.dtype
    )
    conf_write = torch.where(
        mask_sel, confidence_padded, neg_inf.to(confidence_padded.dtype)
    )
    full_confidence[0].index_put_(
        (order_B,), conf_write.to(full_confidence.dtype)
    )

    num_mask_token = mask_index.sum() / mask_index.shape[0]
    number_transfer_tokens = (
        int(num_mask_token * (1 - s / t))
        if i < steps - 1
        else int(num_mask_token)
    )
    if number_transfer_tokens > 0:
        # alg_temp randomness is unreachable on this path: _sample_fixed only
        # routes here for alg_temp 0/None (see its eligibility gate).
        _, transfer_index = torch.topk(full_confidence, number_transfer_tokens)
        x_ = torch.zeros_like(x, device=x.device) + mask_token_id
        x0_write = torch.where(mask_sel, x0_padded, mask_token_id)
        x_[0].index_put_((order_B,), x0_write)
        row_indices = (
            torch.arange(x.size(0), device=x.device)
            .unsqueeze(1)
            .expand_as(transfer_index)
        )
        x[row_indices, transfer_index] = x_[row_indices, transfer_index]

    x = generation_tokens_hook_func(i, x, logits)
    return x


def apply_fixed_shape_sampler(model) -> bool:
    """Install the fixed-shape entropy sampler on THIS model instance.

    Idempotent; no-op when TEMPER_NATIVE_DREAM_SAMPLE=1. Returns True when the
    override is active.
    """
    if os.environ.get("TEMPER_NATIVE_DREAM_SAMPLE") == "1":
        return False
    if getattr(model, "_temper_fixed_sample_installed", False):
        return True

    native_sample = None
    for klass in type(model).__mro__:
        if "_sample" in klass.__dict__:
            native_sample = klass.__dict__["_sample"]
            break
    if native_sample is None:
        raise RuntimeError("DreamGenerationMixin._sample not found in MRO")

    model._sample = types.MethodType(_make_fixed_sample(native_sample), model)
    model._temper_fixed_sample_installed = True
    return True
