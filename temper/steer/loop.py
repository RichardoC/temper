"""The steer loop: a temper-owned masked-diffusion decode with checkpoints.

M3.0 design §2.2. A strict GENERALIZATION of ``dream_fixed._sample_fixed``:
the per-step math is the *same function* (``fixed_shape_step``), the commit
schedule is the same, the tie-free masked-row order is the same — plus
cross-particle state (kill/clone/genealogy) at inside-the-loop granularity, a
hard NFE envelope, and a per-particle decision log (schema v1.3).

Two modes, one code path:

* **all-off** (``policy_enabled=False``): no rungs fire, no kills, no clones.
  Each particle gets a fresh ``torch.manual_seed(seed)`` and runs the
  unsteered step function start-to-finish, so the canvases are BIT-IDENTICAL
  to ``dream.sample`` at the same config. That is the P2 equivalence gate
  (design §5 P2), the same method the M2.3 MPS patch was gated by.
* **steered** (``policy_enabled=True``): R0/R1/R2 on a {25,50,75,100}%
  checkpoint grid, positive-evidence kills, clone-from-checkpoint with a
  +temp-step child, hard envelope.

Sequencing (§2.2): particles are advanced ROUND-ROBIN, one batch-1 forward at
a time — serialized batch-SMC. NFE is invariant to the serialization; batch>1
stays forbidden (MPS NaN evidence, spike §0).
"""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

from temper.model.dream import MASK_TOKEN_ID, _make_hooks
from temper.model.dream_fixed import fixed_shape_step
from temper.steer.emission import (
    EMISSION_SCREEN,
    rank_by_rule,
    rank_screen,
    select_by_rule,
)
from temper.steer.log import Decision, SteerGameLog, decide, persistence_of
from temper.steer.metering import GameMeters
from temper.steer.property import R3Verdict, make_pub_program, run_r3
from temper.steer.verdict import (
    VerdictConfig,
    final_checkpoint_clean,
    ghost_close,
    informed_fatal_evidence,
    r0_shape,
    r1_typecheck,
    r2_stub_markers,
)
from temper.steer.view import SteerProblemView

__all__ = [
    "SteerGameResult",
    "SteerParticle",
    "canvas_prefix_sha256",
    "segment_seed",
    "run_game",
]


def segment_seed(base_seed: int, particle: int, epoch: int, from_step: int) -> int:
    """Deterministic per-(particle, epoch, segment) torch seed.

    The FIRST segment of the first epoch is exactly ``base_seed + particle`` —
    i.e. the seed ``dream.sample`` uses — so an uninterrupted all-off game
    draws the identical RNG stream and reproduces its canvases bit-for-bit
    (the P2 equivalence gate). Any later (re)start gets a derived seed so a
    steered re-decode never re-draws the parent's stream.
    """
    if epoch == 0 and from_step == 0:
        return base_seed + particle
    return (base_seed + particle) * 1000003 + epoch * 10007 + from_step


def canvas_prefix_sha256(x: torch.Tensor, prompt_len: int) -> str:
    """sha256 of a canvas's token ids (prompt positions excluded).

    The genealogy/diversity meter: two particles with the same frozen prefix
    are the same "gene" (design §6.2 item 1 alarm).
    """
    canvas = x[0, prompt_len:].to(torch.int64).cpu().numpy().tobytes()
    return hashlib.sha256(canvas).hexdigest()


@dataclass
class SteerParticle:
    """One live particle: canvas state + lineage + meters."""

    index: int
    resume_epoch: int = 0
    temperature: float = 0.2
    alg: str = "entropy"
    seed: int = 0
    parent: Optional[int] = None
    x: Optional[torch.Tensor] = None        # [1, prompt+canvas]
    nfe: int = 0
    alive: bool = True
    dead_reason: str = ""
    final_raw_text: str = ""
    final_clean: bool = False
    mean_commit_conf: Optional[float] = None
    # Running confidence accumulators. ``mean_commit_conf`` must mean the same
    # thing the bench record's column means (the WHOLE canvas's mean step
    # confidence), because it is the emission rule's ranking statistic; a
    # steered particle is advanced in one segment per checkpoint, so the mean
    # is accumulated across segments instead of being reset by the last one.
    conf_sum: float = 0.0
    conf_count: int = 0
    # Ids the tokenizer could not render, dropped at each decode site (see
    # _decodable_ids). Logged so the artifact's frequency is measured rather
    # than invisible.
    render_dropped_ids: List[int] = field(default_factory=list)
    emission_dropped_ids: List[int] = field(default_factory=list)
    verdict_calls: int = 0
    verifier_wall_s: float = 0.0
    # R3 (M3.1): the property evidence about this canvas, filled at the
    # final window by _fire_r3 and consumed by the v3 emission ranking.
    # ``property_pass_rate`` stays None when nothing was measured — the
    # abstention tier (see temper.steer.emission).
    property_verdict: Optional[R3Verdict] = None
    property_pass_rate: Optional[float] = None
    property_pass_count: int = 0
    property_measured_count: int = 0
    property_call_count: int = 0
    property_statuses: List[str] = field(default_factory=list)
    property_minimal_inputs: List[str] = field(default_factory=list)
    r3_wall_s: float = 0.0


@dataclass
class SteerGameResult:
    """Everything one game (one problem) produced."""

    problem: str
    particles: List[SteerParticle]
    emitted: Optional[SteerParticle]
    meters: GameMeters
    log_path: str
    wall_s: float
    decisions: List[Decision] = field(default_factory=list)


def _committed_mask_share(x: torch.Tensor, prompt_len: int) -> float:
    canvas = x[0, prompt_len:]
    n = int(canvas.numel())
    return float((canvas == MASK_TOKEN_ID).sum()) / n if n else 0.0


def _decodable_ids(tokenizer, gen_ids: List[int],
                   dropped: Optional[List[int]] = None) -> List[int]:
    """Drop ids the tokenizer cannot render (raising nothing, logging all).

    Dream's tokenizer maps every id in ``[0, 151666]`` and returns ``None`` for
    every id in ``[151667, 152064)`` — 397 contiguous slots, 0.3 % of the
    model's ``vocab_size``. Those slots are NOT special tokens, so
    ``skip_special_tokens`` cannot drop them: **both** decode paths raise
    ``TypeError: sequence item N: expected str instance, NoneType found`` when
    such an id sits before the first EOS. Measured on the C-full cell: game 20
    (HumanEval_19) died there after paying its full 2048 NFE, with all 384
    canvas positions committed and zero residual mask — a sampler that draws
    over the full logits width can always land in that band.

    Dropping is the honest treatment: an id with no text form cannot exist in
    any container-scored source, so filtering it is lossless to the container,
    whereas aborting the game would convert a tokenizer artifact into a scored
    failure the unsteered baseline never faced. Every dropped id is handed to
    ``dropped`` so the caller can log it (D2) and the frequency stays
    measurable instead of invisible.
    """
    kept: List[int] = []
    convert = getattr(tokenizer, "convert_ids_to_tokens", None)
    if convert is None:
        # Tokenizers without the method cannot report unmappable ids, so there
        # is nothing to filter. (The real Dream tokenizer always has it; this
        # branch exists for the light test doubles and for any alternative
        # tokenizer wired in later.)
        return [int(i) for i in gen_ids]
    for i in gen_ids:
        if convert(int(i)) is None:
            if dropped is not None:
                dropped.append(int(i))
            continue
        kept.append(int(i))
    return kept


def _render_canvas(tokenizer, x: torch.Tensor, prompt_len: int,
                   dropped: Optional[List[int]] = None) -> str:
    """Decode the canvas region to text (mask positions -> <|mask|> residue).

    The special-token policy here is load-bearing and differs from
    ``_finish``'s on purpose:

    * ``skip_special_tokens=False`` is REQUIRED, because the masked positions
      only reach the rung ladder as ``<|mask|>`` residue and that residue is
      what :func:`temper.steer.verdict.fill_holes` turns into ``todo!()``.
    * The canvas is nonetheless cut at the FIRST EOS exactly as ``_finish``
      does, because the emitted text — the only text the container ever scores
      — is cut there too. Without the cut, a finished canvas carries its
      trailing ``<|endoftext|>`` padding, which renders as literal text AFTER
      the closing fence and makes R0's ``prose_after_fence`` fire on every
      completed canvas. Measured on the C-full cell's logs before this fix:
      92 of 104 final checkpoints flagged ``prose_after_fence``, of which 90
      were the EOS artifact and only 2 were real prose echoes (HumanEval_1,
      HumanEval_9) — a ~45 % false-kill rate on the R0 rung, which would have
      destroyed any cell that executed R0 kills.
    """
    gen_ids = x[0, prompt_len:].tolist()
    eos = tokenizer.eos_token_id
    if eos is not None and eos in gen_ids:
        gen_ids = gen_ids[: gen_ids.index(eos)]
    return tokenizer.decode(_decodable_ids(tokenizer, gen_ids, dropped),
                            skip_special_tokens=False)


def _verdict_at_checkpoint(
    view: SteerProblemView,
    tokenizer,
    particle: SteerParticle,
    *,
    prompt_len: int,
    steps_done: int,
    steps: int,
    config: VerdictConfig,
    gpu_free_mask_share: float,
) -> dict:
    """Fire the rung ladder on one particle's current canvas.

    Returns the checkpoint's verdict record (the exact dict the steer log
    stores and the replayer reads back — evidence only, no policy).
    """
    dropped: List[int] = []
    canvas_text = _render_canvas(tokenizer, particle.x, prompt_len,
                                 dropped=dropped)
    particle.render_dropped_ids.extend(dropped)
    fraction = steps_done / steps
    mask_share = _committed_mask_share(particle.x, prompt_len)
    is_final = steps_done >= steps

    t_v0 = time.perf_counter()
    r0 = r0_shape(view, canvas_text,
                  mask_share=mask_share,
                  expected_mask_share=(0.0 if is_final else 1.0 - fraction),
                  checkpoint_fraction=fraction,
                  config=config)
    # R2 reads COMMITTED text only: strip the mask residue so a masked (not yet
    # committed) `todo!()` cannot be mistaken for a committed stub.
    committed_text = canvas_text.replace("<|mask|>", "")
    stubs = r2_stub_markers(committed_text, config)

    r1 = None
    r1_wall = 0.0
    if config.raw.get("rungs", {}).get("r1_enabled", True):
        try:
            program = ghost_close(view, canvas_text)
            verdict = r1_typecheck(program,
                                   hole_proximity=config.hole_proximity_lines)
            r1_wall = verdict.wall_s
            r1 = {
                "ok": final_checkpoint_clean(verdict) if is_final
                else (not verdict.errors),
                "stderr_sha256": verdict.stderr_sha256,
                "error_codes": list(verdict.error_codes),
                "informed_codes": [e.code for e in
                                   informed_fatal_evidence(verdict, config)],
                "fragment_exact": program.fragment_exact,
            }
        except Exception as exc:  # a verifier crash must never kill a game
            r1 = {"ok": False, "error": f"{type(exc).__name__}: {exc}",
                  "stderr_sha256": "", "error_codes": [],
                  "informed_codes": [], "fragment_exact": False}

    particle.verdict_calls += 1
    particle.verifier_wall_s += time.perf_counter() - t_v0
    return {
        "particle": particle.index,
        "resume_epoch": particle.resume_epoch,
        "step": steps_done,
        "is_final": is_final,
        "nfe_at": particle.nfe,
        "committed_token_count": int(
            (particle.x[0, prompt_len:] != MASK_TOKEN_ID).sum()),
        "canvas_prefix_sha256": canvas_prefix_sha256(particle.x, prompt_len),
        "mask_share": round(mask_share, 4),
        "expected_mask_share": round(0.0 if is_final else 1.0 - fraction, 4),
        "gpu_free_mask_share": round(gpu_free_mask_share, 4),
        "render_dropped_ids": sorted(set(dropped)),
        "render_dropped_count": len(dropped),
        "r0_flags": list(r0.flags),
        "r2_stubs": list(stubs),
        "r1": r1,
    }


def run_game(
    view: SteerProblemView,
    chat_prompt: str,
    *,
    tokenizer,
    model,
    envelope_nfe: int,
    k: int,
    steps: int,
    canvas_len: int,
    temperature: float = 0.2,
    temperatures: Optional[List[float]] = None,
    alg: str = "entropy",
    base_seed: int = 0,
    policy_enabled: bool = True,
    kill_clone_enabled: Optional[bool] = None,
    config: Optional[VerdictConfig] = None,
    harnesses: Optional[Sequence] = None,
    r3_workdir: Path | str | None = None,
    log_path=None,
    top_p: float = 0.95,
    on_checkpoint: Optional[Callable[["SteerParticle", torch.Tensor, float], None]] = None,
    log: Callable[[str], None] = lambda s: None,
) -> SteerGameResult:
    """Play ONE steering game (one problem, one envelope).

    ``policy_enabled=False`` is the equivalence mode: no rungs, no kills, no
    clones — outputs must match ``dream.sample`` bit-for-bit.

    ``temperatures`` gives particle i its own sampling temperature (the rev2
    diversity ladder: one structural canvas at T=0.2 plus a round-robin over
    the measured reachable band). Without it every particle uses the scalar
    ``temperature``, which is what the P2 equivalence gate needs. Particle i
    still draws seed ``base_seed + i`` — the ladder's slots are independent
    configurations, so pairing their seeds would only duplicate canvases.

    ``kill_clone_enabled`` defaults to the FROZEN CONFIG's value (rev2: False)
    and is only ever an explicit override in unit tests. Under the demoted C-full
    role the rung ladder still fires and the policy's would-be verdict is still
    logged, but no kill is executed and no rung fires before the final
    checkpoint — the in-loop machinery's only actionable window was the final
    checkpoint, which is also the emission screen. D4 measured informed-class
    recall 0/5 at <=75% committed, so earlier firings would buy no decision,
    only verifier wall. Each suppressed kill is logged as a ``kill_suppressed``
    event, so the demotion is auditable from the log alone.

    ``on_checkpoint(particle, canvas, fraction)`` is an observation hook called
    at every checkpoint fraction of the frozen grid whether or not the policy is
    enabled; D4 uses it to snapshot real mid-decode canvases for the
    ghost-closure validity table. It must not mutate the canvas.

    ``harnesses`` is this problem's FROZEN property set (M3.1 R3, design §2.3).
    R3 fires at the FINAL WINDOW ONLY, once per R1-clean zero-hole canvas, and
    its evidence feeds the emission ranking — it never kills, clones, reorders
    mid-decode or moves budget. With no harnesses the property ranking has no
    evidence to read, so it degenerates to the screen rule by construction
    (tier 1 for every canvas); the game log records ``r3_available: false`` so
    a cell that was supposed to measure properties cannot silently not do it.
    """
    from temper.steer.verdict import load_verdict_config
    cfg = config or load_verdict_config()
    if kill_clone_enabled is None:
        kill_clone_enabled = cfg.kill_clone_enabled
    # The loop drives ``fixed_shape_step`` itself (it IS the fixed-shape path),
    # so the M2.3 kill-switch is not a fallback here — a run with the native
    # sampler forced on would silently mean something else. Refuse instead of
    # pretending (guardrail: fixed-shape is the default and only path).
    if os.environ.get("TEMPER_NATIVE_DREAM_SAMPLE") == "1":
        raise RuntimeError(
            "TEMPER_NATIVE_DREAM_SAMPLE=1 is set: the steer loop only implements "
            "the fixed-shape sampler. Unset it — the native path is a "
            "regression-contrast escape hatch, not a steering substrate.")
    device = str(next(model.parameters()).device)
    # Per-particle temperature ladder: explicit list wins, else the scalar.
    temps = list(temperatures) if temperatures else [temperature] * k
    if len(temps) < k:
        raise ValueError(
            f"temperatures has {len(temps)} entries for k={k} particles")
    temps = temps[:k]

    enc = tokenizer.apply_chat_template(
        [{"role": "user", "content": chat_prompt}],
        tokenize=True,
        add_generation_prompt=True,
    )
    prompt_len = len(enc)
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    eps = 1e-3

    game_log = SteerGameLog(log_path) if log_path is not None else None
    if game_log is not None:
        game_log.meta(
            game=view.name, problem=view.name, steps=steps, K=k,
            envelope_nfe=envelope_nfe, canvas=canvas_len,
            temperature=temperature, temperatures=temps, alg=alg,
            base_seed=base_seed,
            verdict_config_sha256=cfg.sha256,
            verdict_config_version=cfg.version,
            policy_enabled=policy_enabled,
            kill_clone_enabled=kill_clone_enabled,
            view_sha256=hashlib.sha256(
                (view.prompt + "\x00" + view.fn_name).encode()).hexdigest(),
        )

    meters = GameMeters(envelope_nfe=envelope_nfe, k=k)
    particles: List[SteerParticle] = []
    decisions: List[Decision] = []
    history: List[dict] = []
    checkpoint_steps = {
        max(1, int(round(steps * f))) for f in cfg.checkpoint_grid
    }
    if not policy_enabled:
        # no rungs fire, but the CHECKPOINT GRID is still walked so an
        # observation hook (D4 capture) sees the same fractions the steered
        # path would. Rung work is skipped separately, below.
        pass

    def _spawn(index: int, *, temperature_: float, epoch: int,
               parent: Optional[int], x0: Optional[torch.Tensor]) -> SteerParticle:
        seed = base_seed + index
        p = SteerParticle(index=index, resume_epoch=epoch, temperature=temperature_,
                          alg=alg, seed=seed, parent=parent)
        if x0 is None:
            pad = torch.full((1, canvas_len), MASK_TOKEN_ID, dtype=torch.long,
                             device=device)
            p.x = torch.cat([torch.tensor([enc], dtype=torch.long,
                                          device=device), pad], dim=1)
        else:
            p.x = x0.clone()
        particles.append(p)
        if game_log is not None:
            game_log.emit("spawn", particle=index, resume_epoch=epoch,
                          rng_seed=seed, temp=temperature_, parent=parent,
                          nfe_at=p.nfe)
            game_log.emit("rng", particle=index, resume_epoch=epoch,
                          rng_seed=seed,
                          torch_seed=segment_seed(base_seed, index, epoch, 0))
        return p

    timesteps = torch.linspace(1, eps, steps + 1, device=device)

    def _advance(p: SteerParticle, from_step: int, to_step: int,
                 observe_at: Optional[set] = None) -> None:
        """Run steps [from_step, to_step) for one particle; accrue NFE.

        Runs under ``torch.no_grad()`` because ``diffusion_generate`` — the only
        path the unsteered cells ever take — is decorated ``@torch.no_grad()``
        (vendored generation_utils.py). Building an autograd graph here is not
        merely wasteful: on MPS it selects different kernels and therefore a
        different float path, which is exactly the kind of silent divergence the
        P2 bit-identity gate exists to catch.
        """
        with torch.no_grad():
            _advance_inner(p, from_step, to_step, observe_at)

    def _advance_inner(p: SteerParticle, from_step: int, to_step: int,
                       observe_at: Optional[set] = None) -> None:
        state = {"t0": None, "step_times": [], "step_confs": [],
                 "peak_alloc_MB": None, "peak_driver_MB": None,
                 "hook_errors": []}
        tokens_hook, logits_hook = _make_hooks(state)
        torch.manual_seed(segment_seed(base_seed, p.index, p.resume_epoch,
                                       from_step))
        # pre-loop hook call: opens the timing window the per-step hook closes
        # (same call site as the native/fixed loops')
        x = tokens_hook(None, p.x, None)
        for i in range(from_step, to_step):
            if meters.nfe_model >= meters.envelope_nfe:
                break
            x = fixed_shape_step(
                model, x,
                attention_mask="full", tok_idx=None,
                timesteps=timesteps, i=i, steps=steps, canvas_len=canvas_len,
                temperature=p.temperature, top_p=top_p, top_k=None,
                eos_penalty=0.0, pad_token_id=pad_token_id,
                mask_token_id=MASK_TOKEN_ID, eps=eps,
                generation_tokens_hook_func=tokens_hook,
                generation_logits_hook_func=logits_hook,
            )
            # the tokens hook already purges the Metal driver cache every step
            # (MANDATORY MPS fix, spike §4) — do not double-purge here.
            meters.spend_model(p.index, 1)
            p.nfe += 1
            if observe_at and on_checkpoint is not None and (i + 1) in observe_at:
                p.x = x
                on_checkpoint(p, x, (i + 1) / steps)
        p.x = x
        confs = [c for c in state["step_confs"] if c == c]  # drop NaN
        if confs:
            p.conf_sum += sum(confs)
            p.conf_count += len(confs)
            p.mean_commit_conf = p.conf_sum / p.conf_count

    def _finish(p: SteerParticle) -> None:
        gen_ids = p.x[0, prompt_len:].tolist()
        eos = tokenizer.eos_token_id
        if eos is not None and eos in gen_ids:
            gen_ids = gen_ids[: gen_ids.index(eos)]
        # EMISSION decode site. Same guard as the checkpoint render (see
        # _decodable_ids): the ids in [151667, 152064) are not special tokens,
        # so skip_special_tokens=True cannot drop them and this call raises
        # exactly as the render does. Guarding only the render would leave the
        # identical crash one call later, after the game has spent its whole
        # envelope.
        dropped: List[int] = []
        p.final_raw_text = tokenizer.decode(
            _decodable_ids(tokenizer, gen_ids, dropped),
            skip_special_tokens=True)
        p.emission_dropped_ids = dropped

    # ---- main round-robin loop -------------------------------------------
    t_all = time.perf_counter()
    for i in range(k):
        _spawn(i, temperature_=temps[i], epoch=0, parent=None, x0=None)

    live = list(particles)
    step_cursor: Dict[int, int] = {p.index: 0 for p in particles}
    if not policy_enabled:
        # ALL-OFF (equivalence mode): each particle runs ONE unbroken segment
        # from step 0 to `steps`, exactly as dream.sample does. Splitting at the
        # checkpoint grid would re-seed the RNG mid-canvas and destroy the
        # bit-identity the P2 gate exists to prove; the observation hook is
        # instead fired from inside the segment at the grid steps.
        for p in list(live):
            _advance(p, 0, steps, observe_at=checkpoint_steps)
            step_cursor[p.index] = steps
        live = []
    while live:
        progressed = False
        for p in list(live):
            if meters.remaining_nfe <= 0:
                break
            cur = step_cursor[p.index]
            # advance to the next checkpoint boundary (or the end)
            nxt = min([s for s in checkpoint_steps if s > cur] or [steps])
            if nxt <= cur:
                continue
            _advance(p, cur, nxt)
            step_cursor[p.index] = nxt
            progressed = True
            if policy_enabled and nxt in checkpoint_steps:
                # Rung window (same switch as the kill demotion): with
                # kill/clone OFF the ladder fires only at the final
                # checkpoint — D4 measured no informed evidence at <=75%
                # committed, so earlier firings would buy no decision, only
                # verifier wall.
                if not kill_clone_enabled and nxt < steps:
                    continue
                rec = _verdict_at_checkpoint(
                    view, tokenizer, p, prompt_len=prompt_len,
                    steps_done=nxt, steps=steps, config=cfg,
                    gpu_free_mask_share=_committed_mask_share(p.x, prompt_len))
                meters.spend_verifier(p.verifier_wall_s, 1)
                history.append(rec)
                # Policy STATE the loop knows but replay cannot (live count,
                # remaining budget) is written ON the checkpoint record; the
                # decision itself is always recomputed by decide().
                rec["live_particles"] = len(live)
                rec["budget_remaining_nfe"] = meters.remaining_nfe
                rec["envelope_nfe"] = envelope_nfe
                if game_log is not None:
                    game_log.emit("checkpoint", **rec)
                d = decide(rec, config=cfg,
                           persistence=persistence_of(
                               history, p.index, p.resume_epoch))
                decisions.append(d)
                if game_log is not None:
                    game_log.emit("decision", particle=p.index,
                                  resume_epoch=p.resume_epoch, step=nxt,
                                  action=d.action, reason=d.reason,
                                  nfe_at=p.nfe)
                if d.action == "kill":
                    if kill_clone_enabled:
                        p.alive = False
                        p.dead_reason = d.reason
                        live.remove(p)
                        if game_log is not None:
                            game_log.emit("kill", particle=p.index,
                                          reason=d.reason, nfe_at=p.nfe)
                    elif game_log is not None:
                        # Demoted role: the policy's verdict is recorded but
                        # not executed (see the docstring). D4 measured 0/5
                        # informed recall at <=75% committed, so under the
                        # rev2 demotion these events are the measurement of
                        # what a re-enabled kill rung WOULD have done.
                        game_log.emit("kill_suppressed", particle=p.index,
                                      step=nxt, reason=d.reason,
                                      nfe_at=p.nfe)
        if not progressed:
            break

    # ---- final render + emission ------------------------------------------
    for p in particles:
        _finish(p)
    emitted = None
    eligible = [p for p in particles if p.final_raw_text]
    r3_fired: List[int] = []
    if policy_enabled:
        # R1 on the emitted text (the hard screen), then R3 on the canvases it
        # passed. R3 is the FINAL-WINDOW-ONLY rung (design §2.3): it fires here,
        # after every particle has finished, and its evidence can only affect
        # which finished canvas is emitted — never a kill, a clone, a budget
        # move or a mid-decode ordering.
        for p in eligible:
            p.final_clean = _final_is_clean(view, cfg, p)
        if harnesses:
            for p in eligible:
                if not p.final_clean:
                    # §2.3: R3 requires a completed, zero-hole, R1-clean canvas.
                    continue
                _fire_r3(view, cfg, p, harnesses=harnesses,
                         workdir=r3_workdir, meters=meters,
                         game_log=game_log)
                r3_fired.append(p.index)
        emitted = select_by_rule(eligible, cfg.emission)
    else:
        emitted = eligible[0] if eligible else None
    if game_log is not None:
        game_log.emit("emit", particle=(emitted.index if emitted else None),
                      nfe_at=meters.nfe_model,
                      emission_rule=cfg.emission,
                      r3_available=bool(harnesses),
                      r3_fired=r3_fired,
                      r3_calls=sum(p.property_call_count for p in particles),
                      final_clean=[p.index for p in particles if p.final_clean],
                      screen_ranking=[p.index for p
                                      in rank_screen(eligible)] if
                      policy_enabled else [],
                      emission_ranking=[p.index for p in rank_by_rule(
                          eligible, cfg.emission)] if policy_enabled else [],
                      # The v2 screen's pick is ALWAYS recorded beside the
                      # configured rule's. P-prop-screen vs C-screen is a
                      # same-ensemble rule delta (design §3), so the log must
                      # carry both picks for the flip attribution to be
                      # checkable from the game record alone.
                      screen_selected=(_screen_pick_index(eligible)
                                       if policy_enabled else None),
                      property_pass_rate={str(p.index): p.property_pass_rate
                                          for p in particles},
                      property_statuses={str(p.index): p.property_statuses
                                         for p in particles},
                      mean_commit_conf={str(p.index): p.mean_commit_conf
                                        for p in particles},
                      temperatures={str(p.index): p.temperature
                                    for p in particles},
                      # Unmappable-id artifact, per game and per site (see
                      # _decodable_ids). Emission counts are per particle
                      # because every particle is finished; render counts
                      # aggregate every checkpoint of every particle.
                      emission_dropped_ids={str(p.index): sorted(set(p.emission_dropped_ids))
                                            for p in particles
                                            if p.emission_dropped_ids},
                      emission_dropped_count=sum(
                          len(p.emission_dropped_ids) for p in particles),
                      render_dropped_ids={str(p.index): sorted(set(p.render_dropped_ids))
                                          for p in particles
                                          if p.render_dropped_ids},
                      render_dropped_count=sum(
                          len(p.render_dropped_ids) for p in particles))
        game_log.done(nfe_used=meters.nfe_model,
                      n_particles_live=len(live),
                      emitted_particle=(emitted.index if emitted else None),
                      meters=meters.as_record())
    wall = time.perf_counter() - t_all
    log(f"    game {view.name}: nfe={meters.nfe_model}/{envelope_nfe} "
        f"emitted={emitted.index if emitted else None} wall={wall:.1f}s")
    return SteerGameResult(
        problem=view.name, particles=particles, emitted=emitted,
        meters=meters, log_path=str(log_path or ""), wall_s=wall,
        decisions=decisions,
    )


def _final_is_clean(view: SteerProblemView, cfg: VerdictConfig,
                    p: SteerParticle) -> bool:
    """The final-checkpoint hard screen, re-derived from the emitted text."""
    try:
        program = ghost_close(view, p.final_raw_text)
        verdict = r1_typecheck(program, hole_proximity=cfg.hole_proximity_lines)
        return final_checkpoint_clean(verdict)
    except Exception:
        return False


def _fire_r3(
    view: SteerProblemView,
    cfg: VerdictConfig,
    p: SteerParticle,
    *,
    harnesses: Sequence,
    workdir: Path | str | None,
    meters: GameMeters,
    game_log,
) -> None:
    """Run the frozen property set against ONE finished canvas (R3, §2.3).

    Evidence gathering only: the verdict lands on the particle as
    ``property_pass_rate`` (plus its per-call records) for the emission ranking
    to read. R3 is metered on the VERIFIER axis — 256 proptest cases per
    harness cost cargo wall, never model NFE, and the D3 criterion that caps
    verifier work at 10 % of the budget wall reads ``verifier_wall_s``.

    A verifier crash must never kill a game (same rule as R1): an exception
    here leaves the particle with NO property evidence, i.e. the abstention
    tier, and is logged as such rather than silently scored as a failure.
    """
    t0 = time.perf_counter()
    try:
        program = make_pub_program(view, p.final_raw_text)
        verdict = run_r3(
            program.text, harnesses,
            canvas_ref=f"p{p.index}", problem=view.name,
            cases=int(_r3_cases(cfg)), timeout_ms=int(_r3_timeout_ms(cfg)),
            workdir=workdir,
        )
    except Exception as exc:  # a verifier crash is never a verdict
        p.property_verdict = None
        p.property_pass_rate = None
        p.property_statuses = []
        if game_log is not None:
            game_log.emit("r3_error", particle=p.index,
                          error=f"{type(exc).__name__}: {exc}",
                          nfe_at=p.nfe)
        meters.spend_verifier(time.perf_counter() - t0, 0)
        p.verifier_wall_s += time.perf_counter() - t0
        return
    wall = time.perf_counter() - t0
    p.property_verdict = verdict
    p.property_pass_rate = verdict.pass_rate
    p.property_pass_count = verdict.n_passed
    p.property_measured_count = verdict.n_measured
    p.property_call_count = verdict.n_calls
    p.property_statuses = list(verdict.statuses)
    p.property_minimal_inputs = list(verdict.minimal_inputs)
    p.r3_wall_s = verdict.wall_s
    p.verifier_wall_s += wall
    # D3: R3 calls are verifier work on their own axis (never folded into NFE,
    # which is the budget every cell's envelope is stated in).
    meters.spend_verifier(wall, verdict.n_calls)
    if game_log is not None:
        for call in verdict.calls:
            game_log.emit("r3", particle=p.index, nfe_at=p.nfe,
                          **call.as_event())
        # §7 Q1 (operator ruling 2026-10-24): an all-timeout canvas is its own
        # recorded class — abstention must stay measurable so G4 can weigh it
        # against the container verdict before any future re-tiering.
        if verdict.all_abstained:
            game_log.emit("r3_abstain", particle=p.index, nfe_at=p.nfe,
                          statuses=list(verdict.statuses),
                          pass_rate=None)


def _screen_pick_index(candidates: Sequence) -> Optional[int]:
    """The v2 screen's pick over the same pool, as a reference point."""
    picked = select_by_rule(candidates, EMISSION_SCREEN)
    return picked.index if picked is not None else None


def _r3_cases(cfg: VerdictConfig) -> int:
    return int((cfg.raw.get("m3_1") or {}).get("r3", {}).get("cases", 256))


def _r3_timeout_ms(cfg: VerdictConfig) -> int:
    return int((cfg.raw.get("m3_1") or {}).get("r3", {}).get("timeout_ms", 2000))
