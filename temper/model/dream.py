"""Dream-Coder sampling wrapper: K=1 sequential canvases over the model's
own ``diffusion_generate`` loop (no fork, no monkeypatch — decision from
spike REPORT.md §6/§7).

Per-particle flow (design §0/§4):
  chat-template the prompt -> one canvas of `canvas_len` mask tokens ->
  ``model.diffusion_generate(..., output_history=True,
  return_dict_in_generate=True)`` with two native hooks:

  * generation_tokens_hook_func(step, x, logits):
      - torch.mps.empty_cache() EVERY step (mandatory MPS fix, spike §4),
      - record committed-token counts and MPS memory peaks.
  * generation_logits_hook_func(step, x, logits):
      - mean top-1 softmax confidence over currently-masked positions,
        computed in fp32 in bounded chunks (stats only — NEVER clones or
        retains the full [B, L, 152064] logits, ~89 MB/step, spike §0).

K particles = K SEQUENTIAL diffusion_generate calls (batched K>1 is NO-GO:
padded-row bf16 NaNs on MPS + zero throughput gain, spike §0/§3).

Python 3.9 compatible.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import torch

from temper.model.runtime import get_model, mps_memory_snapshot

__all__ = ["ParticleResult", "sample"]

# Dream-Coder mask token id (DreamGenerationConfig default; verified in spike).
MASK_TOKEN_ID = 151666

# Design §4: per-particle temperature jitter set (L0's only steering knob).
DEFAULT_TEMPERATURE_CYCLE = (0.2, 0.4, 0.6, 0.8)

# Design §4: alternate remasking algorithms across particles.
DEFAULT_ALG_CYCLE = ("entropy", "topk_margin")

# fp32 confidence chunk (rows) — bounds the softmax transient to
# chunk * vocab * 4B ~= 39 MB at chunk=64 (vs 233 MB unchunked at canvas 384).
_CONF_CHUNK_ROWS = 64


@dataclass
class ParticleResult:
    """One sequential canvas: raw text + instrumentation stats."""

    index: int
    raw_text: str = ""
    temperature: float = 0.0
    alg: str = "entropy"
    seed: int = 0
    prompt_len: int = 0
    canvas_len: int = 0
    steps: int = 0
    nfe: Optional[int] = None   # M2.0 schema v1.2: forward passes executed
    wall_s: float = 0.0
    per_step_s: List[float] = field(default_factory=list)
    committed_tokens_per_step: List[int] = field(default_factory=list)
    committed_tokens_total: int = 0
    step_confidences: List[float] = field(default_factory=list)
    mean_commit_conf: Optional[float] = None  # NaN-safe mean of step confidences
    truncated: bool = False     # mask tokens still present at end of canvas
    peak_mps_allocated_MB: Optional[float] = None
    peak_mps_driver_MB: Optional[float] = None
    error: Optional[str] = None


def _normalize_particles(
    temperature: Optional[float],
    alg: Optional[str],
    particles: int,
    temperatures: Optional[List[float]],
    algs: Optional[List[str]],
):
    """Resolve per-particle (temperature, alg) lists from the design cycles."""
    if temperatures is None:
        base = temperature if temperature is not None else DEFAULT_TEMPERATURE_CYCLE[0]
        temps = [base] * particles if particles == 1 else [
            DEFAULT_TEMPERATURE_CYCLE[i % len(DEFAULT_TEMPERATURE_CYCLE)]
            for i in range(particles)
        ]
        if particles == 1:
            temps = [base]
    else:
        temps = list(temperatures)
        if len(temps) < particles:
            temps = temps + [temps[-1]] * (particles - len(temps))
    if algs is None:
        algs_out = (
            [alg] * particles if alg else
            [DEFAULT_ALG_CYCLE[i % len(DEFAULT_ALG_CYCLE)] for i in range(particles)]
        )
    else:
        algs_out = list(algs)
        if len(algs_out) < particles:
            algs_out = algs_out + [algs_out[-1]] * (particles - len(algs_out))
    return temps[:particles], algs_out[:particles]


def _make_hooks(state: dict):
    """Build the two native hooks with per-step instrumentation.

    state keys filled: step_times, step_confs, peak_alloc_MB, peak_driver_MB.
    """

    def tokens_hook(step, x, logits):
        now = time.perf_counter()
        if step is None:
            state["t0"] = now           # pre-loop call: (None, x, None)
        else:
            state["step_times"].append(now - state["t0"])
            state["t0"] = now
        # MANDATORY MPS fix (spike §4): purge Metal driver cache each step,
        # otherwise freed 20-25 GB/step sampling transients accumulate -> OOM.
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
            alloc = torch.mps.current_allocated_memory() / 2**20
            drv = torch.mps.driver_allocated_memory() / 2**20
            state["peak_alloc_MB"] = max(state["peak_alloc_MB"] or 0.0, alloc)
            state["peak_driver_MB"] = max(state["peak_driver_MB"] or 0.0, drv)
        return x

    def logits_hook(step, x, logits):
        # Stats-only read of the raw [B, L, V] logits: mean top-1 softmax
        # confidence over currently-masked positions, fp32, chunked.
        try:
            if logits is None or step is None:
                return logits
            mask_rows = (x[0] == MASK_TOKEN_ID).nonzero(as_tuple=True)[0]
            n = int(mask_rows.numel())
            if n == 0:
                return logits
            total = 0.0
            count = 0
            for i in range(0, n, _CONF_CHUNK_ROWS):
                sel = logits[0, mask_rows[i : i + _CONF_CHUNK_ROWS]].to(torch.float32)
                top1 = torch.softmax(sel, dim=-1).max(dim=-1).values
                total += float(top1.sum())
                count += int(top1.numel())
                del sel, top1
            conf = total / max(count, 1)
            if math.isfinite(conf):
                state["step_confs"].append(conf)
            else:
                state["step_confs"].append(float("nan"))
        except Exception as exc:  # instrumentation must never kill generation
            state.setdefault("hook_errors", []).append(repr(exc))
        return logits

    return tokens_hook, logits_hook


def _generate_one(
    tokenizer,
    model,
    prompt: str,
    *,
    canvas_len: int,
    steps: int,
    temperature: float,
    alg: str,
    seed: int,
    device: str,
) -> ParticleResult:
    torch.manual_seed(seed)
    enc = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
    )
    input_ids = torch.tensor([enc], dtype=torch.long).to(device)
    attention_mask = torch.ones_like(input_ids)
    prompt_len = int(input_ids.shape[1])

    state: dict = {
        "t0": None,
        "step_times": [],
        "step_confs": [],
        "peak_alloc_MB": None,
        "peak_driver_MB": None,
        "hook_errors": [],
    }
    tokens_hook, logits_hook = _make_hooks(state)

    t_gen = time.perf_counter()
    out = model.diffusion_generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=canvas_len,
        steps=steps,
        temperature=temperature,
        top_p=0.95,
        alg=alg,
        alg_temp=0.0,
        output_history=True,
        return_dict_in_generate=True,
        generation_tokens_hook_func=tokens_hook,
        generation_logits_hook_func=logits_hook,
    )
    wall_s = time.perf_counter() - t_gen

    gen_ids = out.sequences[0][prompt_len:].tolist()
    truncated = MASK_TOKEN_ID in gen_ids
    eos = tokenizer.eos_token_id
    if eos is not None and eos in gen_ids:
        gen_ids = gen_ids[: gen_ids.index(eos)]
    raw_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

    # committed tokens per step from output_history (x clone after each step);
    # history holds CUMULATIVE committed-so-far -> convert to per-step deltas
    committed_cumulative = []
    for h in out.history or []:
        committed_cumulative.append(
            max(0, int((h[0] != MASK_TOKEN_ID).sum()) - prompt_len)
        )
    committed = []
    prev = 0
    for c in committed_cumulative:
        committed.append(c - prev)
        prev = c

    confs = state["step_confs"]
    finite = [c for c in confs if math.isfinite(c)]
    mean_conf = sum(finite) / len(finite) if finite else None

    res = ParticleResult(
        index=0,
        raw_text=raw_text,
        temperature=temperature,
        alg=alg,
        seed=seed,
        prompt_len=prompt_len,
        canvas_len=canvas_len,
        steps=steps,
        # M2.0 schema v1.2: one forward pass per denoise step actually
        # executed (per_step_s carries exactly one timing entry per step).
        nfe=len(state["step_times"]),
        wall_s=round(wall_s, 2),
        per_step_s=[round(s, 3) for s in state["step_times"]],
        committed_tokens_per_step=committed,
        committed_tokens_total=(committed_cumulative[-1] if committed_cumulative else 0),
        step_confidences=[round(c, 4) if math.isfinite(c) else None for c in confs],
        mean_commit_conf=round(mean_conf, 4) if mean_conf is not None else None,
        truncated=truncated,
        peak_mps_allocated_MB=(
            round(state["peak_alloc_MB"], 1)
            if state["peak_alloc_MB"] is not None else None
        ),
        peak_mps_driver_MB=(
            round(state["peak_driver_MB"], 1)
            if state["peak_driver_MB"] is not None else None
        ),
    )
    if state["hook_errors"]:
        res.error = "instrumentation warnings: " + "; ".join(state["hook_errors"][:3])
    return res


def sample(
    prompt: str,
    *,
    canvas_len: int,
    steps: int,
    temperature: Optional[float] = None,
    alg: Optional[str] = None,
    particles: int = 1,
    temperatures: Optional[List[float]] = None,
    algs: Optional[List[str]] = None,
    base_seed: int = 0,
    progress: Optional[Callable[[int, "ParticleResult"], None]] = None,
) -> List[ParticleResult]:
    """Sample `particles` canvases SEQUENTIALLY (K=1 per forward, spike §0).

    Per-particle config: temperature cycled through the design §4 set
    {0.2, 0.4, 0.6, 0.8} (or explicit `temperatures`), remasking algorithm
    alternating entropy/topk_margin (or explicit `algs`). Determinism: each
    particle seeds torch with base_seed + i.

    Returns one ParticleResult per particle; a failing particle carries
    `error` and does not abort the remaining ones.
    """
    if particles < 1:
        raise ValueError("particles must be >= 1")
    tokenizer, model = get_model()
    device = str(next(model.parameters()).device)
    temps, algs_out = _normalize_particles(
        temperature, alg, particles, temperatures, algs
    )

    results: List[ParticleResult] = []
    for i in range(particles):
        r = ParticleResult(index=i)
        try:
            r = _generate_one(
                tokenizer,
                model,
                prompt,
                canvas_len=canvas_len,
                steps=steps,
                temperature=temps[i],
                alg=algs_out[i],
                seed=base_seed + i,
                device=device,
            )
            r.index = i
        except Exception as exc:  # keep remaining particles alive
            r.index = i
            r.temperature = temps[i]
            r.alg = algs_out[i]
            r.seed = base_seed + i
            r.error = f"{type(exc).__name__}: {exc}"
        if progress is not None:
            progress(i, r)
        results.append(r)
    # final purge so the next phase (harness) sees a quiet GPU
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return results


def memory_peak_note() -> dict:
    """Process-level memory counters for run records (non-fatal on CPU)."""
    return mps_memory_snapshot()
