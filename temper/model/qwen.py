"""Qwen2.5-Coder-7B-Instruct AR sampling wrapper — M2.4b, baseline row B3.

The AR twin of the Dream-Coder dLLM cells (docs/benchmarks-decision.md §2:
"B3: same-harness AR control, self-normalizing — isolates diffusion-search
effect from model quality"). Same instruction prompt (byte-identical user
turn from temper.multipl_e.bench.INSTRUCTION_TEMPLATE), same seeds
(base_seed+i per sample), same T=0.2 / top_p=0.95 / K=4 / 384-token budget —
but standard autoregressive decoding with KV cache instead of a denoising
canvas.

Interface parity: `sample` mirrors temper.model.dream.sample — the
bench driver's sample_fn injection point (run_bench) accepts either engine
unchanged. Per-sample results subclass dream.ParticleResult, so every field
the driver reads (index/raw_text/wall_s/nfe/truncated/error/seed/...) exists
with the same meaning; AR additions are additive: `nfe_prefill` (prefill
forward passes; gen forwards stay in `nfe`) and `rendered_prompt` (full chat
rendering; its sha256 pins the exact model-visible prompt in the record).

NFE semantics (recorded honestly, NOT unit-matched — see M2.4b report):
  * dLLM cell: one NFE per denoise step = one FULL-canvas forward (prompt +
    384 positions re-encoded every step).
  * AR row:   `nfe` = one forward per GENERATED token ([1,1] with KV cache),
              `nfe_prefill` = prompt-encoding forwards (= 1 here).
  Same "NFE" label, wildly different work per unit — wall-clock is the
  second comparison axis, and both are reported.

Config parity with the dLLM cells (this is a matched-config twin, NOT the
Qwen model card's recommended sampling): temperature 0.2, top_p 0.95, no
repetition penalty, no top_k, greedy off. The sampling tail is the VERBATIM
HF math already ported + leak-fixed in temper.model.dream_fixed
(_top_p_logits) — B2 and B3 share the literally same filtering code path
(fp32 cast -> temperature div -> top-p mask -> softmax -> multinomial).
The Qwen generation_config defaults (T=0.7, top_p=0.8, top_k=20,
repetition_penalty=1.1) are deliberately NOT applied.

Chat rendering: the tokenizer's own chat template with ONLY the user
message — the template injects its default system line ("You are Qwen,
created by Alibaba Cloud. You are a helpful assistant."), the user turn is
byte-identical to bench.build_prompt(problem_prompt) output, and the full
rendered string is carried on every particle (rendered_prompt) for
sha256-recording in the run record.

MPS notes: weights ~15.2 GB bf16 (same footprint as Dream-Coder); the
decode loop is constant-shape ([1, prompt_len] prefill once, then [1,1])
so the MPSGraph recompile storm the fixed-shape sampler solves for the
dLLM does not exist here — no per-step empty_cache dance, and
use_cache=True is safe (the spike's no-cache blocker is a diffusion-loop
issue; AR reuse of K/V across steps is exactly what the cache is for).

Python 3.9 compatible.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable, List, Optional

import torch

from temper.model.dream import ParticleResult
from temper.model.dream_fixed import _top_p_logits  # verbatim HF top-p port

__all__ = ["MODEL_ID", "QwenParticle", "sample", "build_rendered_prompt",
           "get_model_qwen"]

MODEL_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"  # Apache-2.0; HF-cached

# Matched-config parity with the dLLM cells (B2): T=0.2, top_p=0.95, and
# NOTHING else (no rep penalty, no top_k). max_new_tokens = the 384 canvas
# slot (arguable choice — documented in the M2.4b report).
DEFAULT_TEMPERATURE = 0.2
DEFAULT_TOP_P = 0.95
AR_ALG_LABEL = "ar"  # honest row label; "entropy"/"topk_margin" are dLLM-only

_RESIDENT = {"tokenizer": None, "model": None}


@dataclass
class QwenParticle(ParticleResult):
    """One AR sample: ParticleResult interface + additive AR fields.

    nfe              = generated tokens (the task-mandated capture; each
                       token is produced by exactly one forward pass)
    nfe_prefill      = prompt-encoding forward passes (1 for unchunked prefill)
    n_forward_total  = ACTUAL forward passes executed. Decoder-only AR
                       produces the FIRST generated token from the prefill
                       forward's last-position logits, so
                       n_forward_total == nfe + nfe_prefill - 1; recorded
                       explicitly so the record needs no derivation.
    rendered_prompt  = full chat-template rendering (sha256 recorded per row)
    tokens_total     = generated tokens incl. EOS (committed_tokens_total,
                       inherited by the driver's row schema, counts only the
                       non-special visible tokens for AR)
    """

    nfe: int = 0
    nfe_prefill: int = 0
    n_forward_total: int = 0
    rendered_prompt: str = ""
    tokens_total: int = 0
    mean_token_conf: Optional[float] = None  # mean sampled-token softmax prob


def _load_once():
    """Resident loader (mps + bf16 production path; mirrors runtime._load_once
    but for Qwen2ForCausalLM with use_cache=True — AR NEEDS the cache)."""
    from temper.model.runtime import resolve_device

    device, dtype = resolve_device()
    t0 = time.perf_counter()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=dtype,           # transformers 4.57 arg name (was torch_dtype)
        device_map=device,
    )
    model.eval()
    # AR contrast with the dLLM runtime (runtime.py forces use_cache=False —
    # that fix is Dream-loop-specific; the AR loop is broken without cache).
    model.config.use_cache = True
    load_s = time.perf_counter() - t0
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"temper.model.qwen: loaded {MODEL_ID} "
        f"({n_params/1e9:.2f}B params, {str(dtype).split('.')[-1]}, "
        f"device={device}, use_cache={model.config.use_cache}) "
        f"in {load_s:.1f}s",
        flush=True,
    )
    return tokenizer, model


def get_model_qwen():
    """Resident holder: load tokenizer+model once per process, reuse after."""
    if _RESIDENT["model"] is None:
        _RESIDENT["tokenizer"], _RESIDENT["model"] = _load_once()
    return _RESIDENT["tokenizer"], _RESIDENT["model"]


def build_rendered_prompt(tokenizer, user_content: str) -> str:
    """The exact string the model sees: the tokenizer's own chat template
    around a SINGLE user message whose text is byte-identical to
    bench.build_prompt output. The template's default system line is kept
    (publisher's recommended formatting; no custom system prompt)."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content}],
        tokenize=False,
        add_generation_prompt=True,
    )


def _filtered_probs(logits: torch.Tensor, *, temperature: float,
                    top_p: float) -> torch.Tensor:
    """Sampling-tail math BEFORE the draw: fp32 cast, temperature div,
    top-p filter (_top_p_logits — the shared verbatim HF port also used by
    the dLLM fixed-shape tail), softmax. Accepts [1, V] (the decode loop's
    last-position logits) or [V]. Returns probs [1, V]."""
    logits = logits.to(torch.float32)
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
    if temperature > 0:
        logits = logits / temperature
    if top_p is not None and top_p < 1:
        logits = _top_p_logits(logits, top_p)
    return torch.softmax(logits, dim=-1)


def _sample_token(logits: torch.Tensor, *, temperature: float,
                  top_p: float) -> tuple:
    """One-token sampling tail: VERBATIM parity with the dLLM cells' tail
    (dream_fixed._entropy_sample_tokens_padded minus the neg-entropy
    confidence, which is mask-bookkeeping, not sampling): fp32 cast,
    temperature div, top-p filter, softmax, one multinomial draw.
    Returns (token_id, prob)."""
    probs = _filtered_probs(logits, temperature=temperature, top_p=top_p)
    tok = torch.multinomial(probs, num_samples=1)           # [1, 1]
    prob = torch.gather(probs, -1, tok).squeeze()
    return int(tok.item()), float(prob)


def _eos_ids(model, tokenizer) -> set:
    """EOS set from the generation config ([151645 <|im_end|>, 151643
    <|endoftext|>] for Qwen2.5-Coder-7B-Instruct), tokenizer fallback."""
    ids = set()
    gc = getattr(model, "generation_config", None)
    if gc is not None and gc.eos_token_id is not None:
        v = gc.eos_token_id
        ids.update(v if isinstance(v, (list, tuple)) else [v])
    if tokenizer.eos_token_id is not None:
        ids.add(tokenizer.eos_token_id)
    return ids


def _mps_snapshot():
    try:
        if torch.backends.mps.is_available():
            return (torch.mps.current_allocated_memory() / 2**20,
                    torch.mps.driver_allocated_memory() / 2**20)
    except Exception:
        pass
    return None, None


def _generate_one(
    tokenizer,
    model,
    rendered_prompt: str,
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
    device: str,
) -> QwenParticle:
    """One autoregressive sample with KV cache (use_cache=True).

    Instrumentation per token (the AR analog of dream.py's per-step hooks):
    wall time, sampled-token prob, MPS memory peaks. NFE: the prefill
    forward produces the FIRST generated token from its last-position
    logits; each subsequent token costs one [1,1] decode forward — so
    total forwards == generated tokens, decomposed as nfe_prefill (1) +
    decode forwards (nfe - 1); nfe captures the generated-token count.
    """
    torch.manual_seed(seed)
    enc = tokenizer(rendered_prompt, add_special_tokens=False)["input_ids"]
    input_ids = torch.tensor([enc], dtype=torch.long, device=device)
    prompt_len = int(input_ids.shape[1])

    eos = _eos_ids(model, tokenizer)

    res = QwenParticle(index=0, seed=seed, temperature=temperature,
                       alg=AR_ALG_LABEL, prompt_len=prompt_len,
                       canvas_len=max_new_tokens, steps=0,
                       rendered_prompt=rendered_prompt)

    peak_alloc, peak_driver = _mps_snapshot()
    t_gen = time.perf_counter()

    # ---- prefill: one forward over the whole prompt -----------------------
    t0 = time.perf_counter()
    out = model(input_ids, use_cache=True)
    res.nfe_prefill = 1
    res.per_step_s = [round(time.perf_counter() - t0, 3)]  # slot 0 = prefill

    past = out.past_key_values
    logits = out.logits[:, -1, :]

    # ---- decode: [1,1] forwards, one per generated token -------------------
    gen_ids: List[int] = []
    confs: List[float] = []
    n_decode_forwards = 0
    for _ in range(max_new_tokens):
        t0 = time.perf_counter()
        tok, conf = _sample_token(logits, temperature=temperature, top_p=top_p)
        gen_ids.append(tok)
        confs.append(conf)
        if torch.backends.mps.is_available():
            a, d = _mps_snapshot()
            peak_alloc = max(peak_alloc or 0.0, a or 0.0)
            peak_driver = max(peak_driver or 0.0, d or 0.0)
        if tok in eos:
            # the forward that produced EOS already ran (counted in nfe);
            # no further forward is needed.
            res.per_step_s.append(round(time.perf_counter() - t0, 3))
            break
        if len(gen_ids) >= max_new_tokens:
            # budget exhausted: this token was the last — no further forward
            # (and no phantom compute in the wall-clock).
            res.per_step_s.append(round(time.perf_counter() - t0, 3))
            break
        out = model(torch.tensor([[tok]], dtype=torch.long, device=device),
                    past_key_values=past, use_cache=True)
        n_decode_forwards += 1
        past = out.past_key_values
        logits = out.logits[:, -1, :]
        res.per_step_s.append(round(time.perf_counter() - t0, 3))

    res.wall_s = round(time.perf_counter() - t_gen, 2)
    res.nfe = len(gen_ids)              # 1 token per forward (task capture)
    res.n_forward_total = 1 + n_decode_forwards
    res.tokens_total = len(gen_ids)
    vis = [t for t in gen_ids if t not in eos]
    res.committed_tokens_total = len(vis)
    res.truncated = (len(gen_ids) >= max_new_tokens) and not (gen_ids[-1] in eos)
    finite = [c for c in confs if math.isfinite(c)]
    res.mean_token_conf = (round(sum(finite) / len(finite), 4)
                           if finite else None)
    raw = tokenizer.decode(gen_ids, skip_special_tokens=True)
    res.raw_text = raw
    res.committed_tokens_per_step = [1] * len(vis)  # AR commits 1 token/step
    res.step_confidences = [round(c, 4) if math.isfinite(c) else None
                            for c in confs]
    res.mean_commit_conf = res.mean_token_conf  # closest AR analog, see report
    res.peak_mps_allocated_MB = (round(peak_alloc, 1)
                                 if peak_alloc is not None else None)
    res.peak_mps_driver_MB = (round(peak_driver, 1)
                              if peak_driver is not None else None)
    return res


def sample(
    prompt: str,
    *,
    canvas_len: int = 384,
    steps: Optional[int] = 0,
    particles: int = 1,
    temperatures: Optional[List[float]] = None,
    algs: Optional[List[str]] = None,   # accepted for interface parity; AR
    base_seed: int = 0,                 #   rows are always labeled "ar"
    temperature: Optional[float] = None,
    top_p: float = DEFAULT_TOP_P,
    progress: Optional[Callable[[int, QwenParticle], None]] = None,
) -> List[QwenParticle]:
    """Sample `particles` AR completions SEQUENTIALLY (the driver's K=4).

    Mirrors temper.model.dream.sample's contract: canvas_len is the new-token
    budget (max_new_tokens — the 384 canvas slot), steps must be 0/None (an
    AR model has no denoise steps; the record's steps=0 is the honest value),
    per-particle temperature from `temperatures` (driver passes the bench
    T for every sample), seed = base_seed + i. A failing particle carries
    `error` and does not abort the remaining ones.
    """
    if steps not in (None, 0):
        raise ValueError(
            "temper.model.qwen.sample: AR engine takes no denoise steps "
            f"(got steps={steps!r}; pass 0)")
    if particles < 1:
        raise ValueError("particles must be >= 1")
    temps = (list(temperatures) if temperatures else
             [temperature or DEFAULT_TEMPERATURE] * particles)
    if len(temps) < particles:
        temps = temps + [temps[-1]] * (particles - len(temps))

    tokenizer, model = get_model_qwen()
    rendered = build_rendered_prompt(tokenizer, prompt)
    device = str(next(model.parameters()).device)

    results: List[QwenParticle] = []
    for i in range(particles):
        r = QwenParticle(index=i, seed=base_seed + i, temperature=temps[i],
                         alg=AR_ALG_LABEL, rendered_prompt=rendered)
        try:
            r = _generate_one(
                tokenizer, model, rendered,
                max_new_tokens=canvas_len,
                temperature=temps[i],
                top_p=top_p,
                seed=base_seed + i,
                device=device,
            )
            r.index = i
        except Exception as exc:  # keep remaining particles alive
            r.index = i
            r.error = f"{type(exc).__name__}: {exc}"
        if progress is not None:
            progress(i, r)
        results.append(r)
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()   # quiet GPU for the next phase
    return results
