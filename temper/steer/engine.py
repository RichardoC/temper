"""The steer engine as a bench-driver ``sample_fn`` — M3.0 C-full (design §4.1).

Design §4.1 requires steered runs to flow the UNCHANGED export/score
instrument, and §2.2 names the seam: "steering runs entirely through the
existing bench driver via its ``sample_fn`` injection seam (the M2.4b Qwen
path)". This module is that seam for the steer loop.

The cell's unit is one GAME per problem (design §4.1: "B NFE spent under policy
-> one emitted completion -> container verdict (binary)"), so the engine returns
exactly ONE row per problem — the game's emitted canvas — and the C-full run is
invoked with ``-K 1``. That is what makes the cell commensurable with the
unsteered anchors: their per-problem outcome is also a single binary, and the
run record's ``honest pass@1`` is then the fraction of problems whose emitted
canvas passed, with no pass@k machinery in the way.

Per-problem detail the row cannot carry (which particle was emitted, the full
screen ranking, every checkpoint's verdict, suppressed kills, and every R3
call) lives in the D2 steer log at ``<out>.steer/<problem>.jsonl`` — the
forensics record the C-full vs C-screen canvas-pair diff is computed from.

**Every particle's final canvas is banked** to
``<out>.canvases/<problem>__p<i>.txt`` (M3.1 §7 clarification, operator ruling
2026-10-24 Q3). The M3.0 C-full cell banked only the EMITTED canvas, which is
why its oracle could never be measured offline: a non-emitted canvas cannot be
container-scored from a sha. These files are text only — they cost I/O, never
NFE, and the cell's envelope accounting is untouched. They are what makes
P-prop-engine's own oracle audit exact and the P-prop-screen-twin over the
engine's own ensemble computable at all.

**R3 harnesses are injected, not discovered**: the frozen set is read once per
cell from the freeze manifest (``harness_freeze.json``, its own sha) and handed
to each game. A cell whose manifest is missing or tampered with refuses to
start rather than running with an empty property set — a property cell that
silently measures nothing is the worst possible failure mode here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional

__all__ = ["make_steer_sample_fn"]

# The rev2 diversity ladder (steer_verdict_config.json rev2.diversity_ladder):
# one structural canvas at T=0.2 (emission-preferred, protects the banked
# 11-passer cohort) plus a round-robin over the measured reachable band
# {0.8, 0.95, 1.0, 1.1, 1.2}, padded to the 8-canvas envelope slot count.
REV2_LADDER = (0.2, 0.8, 0.95, 1.0, 1.1, 1.2, 0.8, 0.95)


def ladder_for_envelope(envelope_nfe: int, steps: int) -> List[float]:
    """The ladder the envelope buys: floor(envelope/steps) canvases of REV2_LADDER.

    At 2048 NFE and steps=256 that is the full 8-canvas rev2 ladder. Refusing a
    non-integral envelope is deliberate — a partial canvas would silently change
    the ladder's composition (which temperatures got how many draws).
    """
    if steps <= 0:
        raise ValueError("steps must be positive")
    n = envelope_nfe // steps
    if n * steps != envelope_nfe:
        raise ValueError(
            f"envelope {envelope_nfe} is not a whole number of {steps}-step "
            f"canvases — the ladder's composition would be ambiguous")
    if n > len(REV2_LADDER):
        raise ValueError(
            f"envelope {envelope_nfe} buys {n} canvases but the frozen rev2 "
            f"ladder has {len(REV2_LADDER)} slots")
    return list(REV2_LADDER[:n])


def make_steer_sample_fn(
    *,
    out: Path | str,
    envelope_nfe: int,
    steps: int,
    canvas_len: int,
    base_seed: int = 0,
    alg: str = "entropy",
    top_p: float = 0.95,
    harnesses: Optional[Dict[str, List]] = None,
    bank_canvases: bool = True,
    r3_workdir: Path | str | None = None,
    log: Callable[[str], None] = lambda s: None,
) -> Callable:
    """Build the ``sample_fn`` that plays one steering game per problem.

    Lazy-imports torch through ``run_game``'s own dependencies, so importing
    this module (and the bench driver's tests) stays torch-free.

    ``harnesses`` maps problem name -> frozen :class:`temper.steer.property.
    Harness` list; those lists are handed to every game so R3 can fire at the
    final window. ``None`` means "this cell measures no properties" and is only
    correct for cells that do not claim to (P-25re's unsteered role does not
    run the engine seam at all). ``bank_canvases`` writes every particle's
    final text to ``<out>.canvases/`` (see the module docstring).
    """
    from temper.model.dream import ParticleResult
    from temper.multipl_e.bench import build_prompt, load_problems
    from temper.steer.loop import run_game
    from temper.steer.verdict import load_verdict_config
    from temper.steer.view import SteerProblemView

    ladder = ladder_for_envelope(envelope_nfe, steps)
    out = Path(out)
    log_dir = Path(str(out) + ".steer")
    log_dir.mkdir(parents=True, exist_ok=True)
    canvas_dir = Path(str(out) + ".canvases")
    if bank_canvases:
        canvas_dir.mkdir(parents=True, exist_ok=True)

    # problem lookup by the delivered chat prompt: the driver hands the
    # sample_fn the rendered user turn, which is what the model sees, so the
    # mapping stays exact without threading a problem dict through the seam.
    views: Dict[str, SteerProblemView] = {}
    for td in load_problems("humaneval-rs"):
        views[build_prompt(td["prompt"])] = SteerProblemView.from_task(td)

    from temper.model.runtime import get_model

    def sample_fn(prompt: str, *, canvas_len: int, steps: int, particles: int,
                  temperatures=None, algs=None, base_seed: int = 0,
                  seeds=None, progress=None, **kwargs) -> List[ParticleResult]:
        if particles != 1:
            raise ValueError(
                f"the steer engine returns ONE row per problem (the game's "
                f"emitted canvas); got particles={particles}. Run the cell "
                "with -K 1.")
        view = views.get(prompt)
        if view is None:
            raise ValueError(
                "steer engine: prompt does not match any vendored humaneval-rs "
                "problem — refusing to guess which view to steer")
        cfg = load_verdict_config()
        tokenizer, model = get_model()
        res = run_game(
            view, prompt,
            tokenizer=tokenizer, model=model,
            envelope_nfe=envelope_nfe, k=len(ladder), steps=steps,
            canvas_len=canvas_len, temperatures=ladder, alg=alg,
            base_seed=base_seed, policy_enabled=True, config=cfg,
            harnesses=(harnesses or {}).get(view.name),
            r3_workdir=r3_workdir,
            log_path=log_dir / f"{view.name}.jsonl", top_p=top_p, log=log,
        )
        if bank_canvases:
            # Every particle, not just the emitted one (Q3 clarification): the
            # oracle audit and the same-draws rule delta both need the draws the
            # engine did NOT emit, and a sha cannot be container-scored.
            for part in res.particles:
                if not part.final_raw_text:
                    continue
                (canvas_dir / f"{view.name}__p{part.index}.txt").write_text(
                    part.final_raw_text, encoding="utf-8")
        emitted = res.emitted
        row = ParticleResult(
            index=0,
            raw_text=emitted.final_raw_text if emitted else "",
            temperature=emitted.temperature if emitted else 0.0,
            alg=alg,
            seed=base_seed,
            prompt_len=0,
            canvas_len=canvas_len,
            # The row's NFE is the GAME's NFE: that is the envelope the cell
            # spends per problem, and it is the number the anchors are matched
            # against (design §4.1). Per-particle NFE is in the steer log.
            nfe=res.meters.nfe_model,
            wall_s=round(res.wall_s, 2),
        )
        row.nfe_model = res.meters.nfe_model
        row.nfe_verifier_calls = res.meters.nfe_verifier_calls
        row.verifier_wall_s = res.meters.verifier_wall_s
        row.mean_commit_conf = emitted.mean_commit_conf if emitted else None
        row.committed_tokens_total = (
            int((emitted.x[0] != 151666).sum()) if emitted else 0)
        row.truncated = False
        if emitted is None:
            row.error = "game emitted no canvas"
        if progress is not None:
            progress(0, row)
        return [row]

    return sample_fn
