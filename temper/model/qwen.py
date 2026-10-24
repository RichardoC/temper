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
import re
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import torch

from temper.model.dream import ParticleResult
from temper.model.dream_fixed import _top_p_logits  # verbatim HF top-p port

__all__ = ["MODEL_ID", "NonFiniteLogitsError", "QwenParticle", "sample",
           "build_rendered_prompt", "get_model_qwen"]


class NonFiniteLogitsError(RuntimeError):
    """A decode step's logits were inf/nan (measured on MPS with long prompts).

    A DISTINCT class so the bounded retry below can recognise exactly this
    failure and so a record can tell a numerical device artifact apart from any
    other generation failure.
    """

MODEL_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"  # Apache-2.0; HF-cached

# Matched-config parity with the dLLM cells (B2): T=0.2, top_p=0.95, and
# NOTHING else (no rep penalty, no top_k). max_new_tokens = the 384 canvas
# slot (arguable choice — documented in the M2.4b report).
DEFAULT_TEMPERATURE = 0.2
DEFAULT_TOP_P = 0.95
AR_ALG_LABEL = "ar"  # honest row label; "entropy"/"topk_margin" are dLLM-only

# M3.1 retry policy — AMENDED ON MEASUREMENT (operator ruling 2026-10-24).
#
# Ruling (b)(ii) originally allowed ONE bounded retry because the guard was
# first characterised on an INTERMITTENT failure (2/25 at canvas 256, 11/25 at
# 384). Re-measurement showed the dominant class is NOT intermittent:
#   * problem 0 fails at generated token 325 and problem 1 at token 357 --
#     identical at canvas_len 384, 448 and 512, and identical across EVERY seed;
#   * a retry with seed + 1000 fails at the SAME token, because at T=0.2 the
#     decode is near-deterministic, so a fresh stream does not escape it.
# The retry therefore cannot recover the measured class: it burns a second
# attempt and then reports the same failure. Default is now ZERO retries — one
# attempt, hard error — which is the honest record. The retry path is KEPT
# (and tested) so a genuinely stochastic failure mode can be re-enabled by
# raising this constant on evidence, rather than by editing logic.
NONFINITE_RETRY_SEED_OFFSET = 1000
NONFINITE_MAX_RETRIES = 0

# M3.1 (b)(iii): stop generation as soon as the answer's closing fence is
# emitted, checked IN the decode loop rather than after it. Measured why this
# matters: an answer is ~300 tokens, and the measured MPS non-finite failure
# struck at token 325 of a 512 budget — i.e. AFTER the answer was complete, in
# the trailing prose. Stopping at the fence removes both the 256-truncation
# class (100% of a 25-problem pilot) and the dominant NaN class in one change.
FENCE_TOKEN_TEXT = "```"

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
    # M3.1 (b): the fence-close cut and the bounded-retry audit trail. Additive,
    # so every pre-M3.1 reader of a QwenParticle is unaffected.
    fence_closed: bool = False
    fence_stopped_early: bool = False
    fence_truncated_chars: int = 0
    attempts: List[dict] = field(default_factory=list)
    n_attempts: int = 1
    retried: bool = False
    retry_succeeded: bool = False


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


_FENCE_END_RE = re.compile(r"```[ \t]*\r?\n?")


def _answer_closed(tokenizer, gen_ids: List[int], n_fence_tokens: int) -> bool:
    """Has the answer's closing fence been emitted? (cheap, per-step)

    A fenced answer contains exactly two fence markers: the opener and the
    closer. Counting FENCE TOKENS therefore answers "is the block closed?" in
    O(1) per step with no decoding at all.

    Qwen tokenizes ``` as a single token (id 73594) but a closing ```` at the
    end of a body can also arrive fused with preceding text, so ``n_fence_tokens``
    is measured on the decoded prefix rather than assumed: the caller tracks it
    by decoding only when a fence-ish token appears, and the value passed here
    is the count of fence markers in the generated answer so far.
    """
    return n_fence_tokens >= 2


def _closing_fence_end(text: str) -> Optional[int]:
    """Index just past the closing ``` fence, or None if the block never closed.

    A fenced block opens with ```lang and closes with ```: the answer is the
    text through the SECOND fence occurrence (the first match is the opener).
    Returning the match END keeps the closing fence itself inside the answer —
    cutting at its start would delete the delimiter the extraction instrument
    needs, turning a complete block into an unterminated one.
    """
    matches = list(_FENCE_END_RE.finditer(text))
    if len(matches) < 2:
        return None
    return matches[1].end()


def _logits_are_finite(logits: torch.Tensor) -> bool:
    """Is this decode step's logit vector usable?

    M3.1 measured the MPS bf16 failure mode this guards: on a long prompt with
    a long generation, the [1,1] decode loop eventually emits non-finite
    logits, and the NEXT `_sample_token` raises
    ``RuntimeError: probability tensor contains either inf, nan or element < 0``
    mid-canvas. On the canvas-prompt path this never fired (the banked B3 cell
    is 0/100), so it is a long-prompt x long-generation interaction, not a
    broken engine. Checking here turns it into a labelled, countable
    generation error at the exact step it happens.
    """
    return bool(torch.isfinite(logits).all().item())


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

    # inference-only loop: no autograd graph (the probe showed transformers
    # otherwise builds one per decode step — slower and memory-hungry).
    with torch.no_grad():
        # ---- prefill: one forward over the whole prompt -------------------
        t0 = time.perf_counter()
        out = model(input_ids, use_cache=True)
        res.nfe_prefill = 1
        res.per_step_s = [round(time.perf_counter() - t0, 3)]  # slot 0 = prefill

        past = out.past_key_values
        logits = out.logits[:, -1, :]

        # ---- decode: [1,1] forwards, one per generated token ---------------
        gen_ids: List[int] = []
        confs: List[float] = []
        n_decode_forwards = 0
        # the ids whose text can contain a fence marker (Qwen: 73594 for ```)
        fence_tokens = set(tokenizer.encode(FENCE_TOKEN_TEXT,
                                            add_special_tokens=False))
        for _ in range(max_new_tokens):
            t0 = time.perf_counter()
            if not _logits_are_finite(logits):
                # MID-CANVAS NON-FINITE LOGITS (see _logits_are_finite): the
                # honest refusal. The tokens produced so far are kept so the
                # caller can see how far it got, and the error names the step.
                raise NonFiniteLogitsError(
                    f"non-finite logits at generated token {len(gen_ids)} "
                    f"of max_new_tokens={max_new_tokens}")
            tok, conf = _sample_token(logits, temperature=temperature,
                                      top_p=top_p)
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
            # Cheap in-loop fence accounting: decode only when the newest
            # token's text could belong to a fence, and only then re-count.
            # Qwen renders ``` as one token, so in practice this decodes twice
            # per answer (opener, closer).
            if tok in fence_tokens:
                n_fence = tokenizer.decode(gen_ids,
                                           skip_special_tokens=True).count(
                    FENCE_TOKEN_TEXT)
                if _answer_closed(tokenizer, gen_ids, n_fence):
                    # the closing fence just landed: the ANSWER is complete.
                    # Stop rather than burn budget on trailing prose (see
                    # FENCE_TOKEN_TEXT) — measured to be where the MPS
                    # non-finite failure actually strikes.
                    res.fence_closed = True
                    res.fence_stopped_early = True
                    res.per_step_s.append(round(time.perf_counter() - t0, 3))
                    break
            if len(gen_ids) >= max_new_tokens:
                # budget exhausted: this token was the last — no further
                # forward (and no phantom compute in the wall-clock).
                res.per_step_s.append(round(time.perf_counter() - t0, 3))
                break
            out = model(torch.tensor([[tok]], dtype=torch.long,
                                     device=device),
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
    # DRIVER-LEVEL FENCE-CLOSE TRUNCATION (M3.1 ruling (b)(iii)). A model that
    # finishes its fenced code block but does not emit EOS keeps generating
    # prose; the raw text then carries a closed answer followed by commentary.
    # Cutting AT the closing fence is what the extraction instrument does
    # anyway, and doing it here keeps `truncated` a true statement about the
    # ANSWER rather than about the budget: after this cut, truncated means "the
    # fence never closed", which is exactly the failure this fix exists to make
    # visible. The measurement is kept on the particle (`fence_closed`) so the
    # cut itself is auditable.
    fence_closed = _closing_fence_end(raw)
    res.fence_closed = fence_closed is not None
    if fence_closed is not None and fence_closed < len(raw):
        res.fence_truncated_chars = len(raw) - fence_closed
        raw = raw[:fence_closed]
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
    seeds: Optional[List[int]] = None,  # explicit per-sample seeds (M3 P1)
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

    if seeds is not None:
        if len(seeds) < particles:
            raise ValueError(
                f"seeds list has {len(seeds)} entries for {particles} particles")
        seed_of = [int(s) for s in seeds[:particles]]
    else:
        seed_of = [base_seed + i for i in range(particles)]

    results: List[QwenParticle] = []
    for i in range(particles):
        r = QwenParticle(index=i, seed=seed_of[i], temperature=temps[i],
                         alg=AR_ALG_LABEL, rendered_prompt=rendered)
        # ONE bounded retry, and only for the measured numerical failure
        # (NONFINITE_MAX_RETRIES); every attempt is recorded on the particle so
        # a retry can never look like a first-attempt success.
        attempts: List[dict] = []
        for attempt in range(NONFINITE_MAX_RETRIES + 1):
            attempt_seed = seed_of[i] + attempt * NONFINITE_RETRY_SEED_OFFSET
            try:
                r = _generate_one(
                    tokenizer, model, rendered,
                    max_new_tokens=canvas_len,
                    temperature=temps[i],
                    top_p=top_p,
                    seed=attempt_seed,
                    device=device,
                )
                r.index = i
                attempts.append({"attempt": attempt, "seed": attempt_seed,
                                 "outcome": "ok"})
                break
            except NonFiniteLogitsError as exc:
                attempts.append({"attempt": attempt, "seed": attempt_seed,
                                 "outcome": "non_finite_logits",
                                 "detail": str(exc)})
                if attempt >= NONFINITE_MAX_RETRIES:
                    r.index = i
                    r.error = f"{type(exc).__name__}: {exc}"
                    break
            except Exception as exc:  # keep remaining particles alive
                attempts.append({"attempt": attempt, "seed": attempt_seed,
                                 "outcome": f"{type(exc).__name__}: {exc}"})
                r.index = i
                r.error = f"{type(exc).__name__}: {exc}"
                break
        r.attempts = attempts
        r.n_attempts = len(attempts)
        r.retried = len(attempts) > 1 and bool(r.error)
        r.retry_succeeded = len(attempts) > 1 and not r.error
        if progress is not None:
            progress(i, r)
        results.append(r)
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()   # quiet GPU for the next phase
    return results
