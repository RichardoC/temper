"""Steer-decision replay log, schema v1.3 (M3.0 D2, docs/m3-design.md §7).

One JSONL file per game at ``<out>.steer/<problem>.jsonl``. Events are
append-only and self-contained: every decision the steer loop takes is a pure
function of the events already in the log plus the frozen verdict config, so
:func:`replay_game` re-derives the whole decision history **from the log alone,
with zero GPU**.

Schema (additive over the M2.x record convention — v1.2 rows are untouched):

  meta    one line, first:        {game, problem, schema_version:"1.3",
                                   verdict_config_sha256, envelope_nfe, K, steps,
                                   temperature, alg, base_seed, view_sha256}
  spawn   one per particle start: {particle, resume_epoch, rng_seed, temp,
                                   parent, nfe_at}
  rng     one per (re)seed:       {particle, resume_epoch, rng_seed, torch_seed}
  checkpoint one per particle-step: {particle, resume_epoch, step, nfe_at,
                                   committed_token_count, canvas_prefix_sha256,
                                   mask_share, r0_flags, r1:{ok, stderr_sha256,
                                   error_codes, informed_spans}, r2_stubs}
  decision one per checkpoint:    {particle, resume_epoch, step, action,
                                   reason, nfe_at}
  kill / clone / emit             the actions, with their payloads
  done    one line, last:         {nfe_used, n_particles_live, emitted_particle}

Replay contract (D2 acceptance): ``replay_game(events, config)`` returns the
same decision sequence the loop recorded — anything else raises
:class:`ReplayMismatch`. The poisoning test tampers one verdict and asserts a
mismatch.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from temper.steer.verdict import VerdictConfig

__all__ = [
    "SCHEMA_VERSION",
    "Decision",
    "ReplayMismatch",
    "SteerGameLog",
    "decide",
    "persistence_of",
    "read_game_log",
    "replay_game",
]

SCHEMA_VERSION = "1.3"

# Decisions the policy can take at a checkpoint. "continue" is the no-op the
# all-off mode always returns (P2's bit-identity gate).
ACTIONS = ("continue", "kill", "clone", "emit")


@dataclass(frozen=True)
class Decision:
    """One policy decision, derived from logged evidence only."""

    action: str                     # continue | kill | clone | emit
    reason: str = ""


class ReplayMismatch(RuntimeError):
    """Raised when a replayed decision differs from the recorded one — the
    log is not self-sufficient, or it was tampered with."""


def persistence_of(history: Sequence[dict], particle: int, resume_epoch: int) -> int:
    """Consecutive-checkpoints-with-informed-evidence run length, INCLUSIVE of
    the last checkpoint in ``history`` for that particle/epoch.

    Recomputed by replay from the recorded checkpoint sequence alone — the loop
    never stores this number, so it cannot be tampered into agreement.
    """
    run = 0
    for rec in history:
        if rec.get("particle") != particle or rec.get("resume_epoch") != resume_epoch:
            continue
        informed = bool((rec.get("r1") or {}).get("informed_codes"))
        run = run + 1 if informed else 0
    return run


def decide(
    checkpoint: dict,
    *,
    config: VerdictConfig,
    persistence: int,
) -> Decision:
    """The v0 verdict & action policy (§3.3), as a PURE function of one logged
    checkpoint record plus policy state.

    Positive evidence only — absence of evidence never kills:

      * ``is_final``: the hard screen. R1 on the completed body is exact
        (D1: 36/36 SyntaxError caught, 0/44 false), so *any* error makes the
        particle ineligible to be the emitted answer — but the game still
        emits one (§3.3 item 2), so this is an ELIGIBILITY flag, not a kill.
      * mid-canvas: a kill needs informed-class R1 errors in committed content
        persisting ``config.patience_checkpoints`` consecutive checkpoints
        (``persistence`` counts them, inclusive of this one), or an R0 fatal
        shape flag, or a committed R2 stub marker.
      * clone: while budget remains and live particles < K_max_eff, spawn from
        the healthiest surviving checkpoint (the loop supplies the lineage).
    """
    r0_flags = tuple(checkpoint.get("r0_flags") or ())
    fatal_r0 = [f for f in r0_flags if f in set(config.r0_flags_fatal)]
    if fatal_r0:
        return Decision("kill", "r0_fatal:" + ",".join(sorted(fatal_r0)))

    stubs = tuple(checkpoint.get("r2_stubs") or ())
    if stubs:
        return Decision("kill", "r2_committed_stub:" + ",".join(sorted(stubs)))

    is_final = bool(checkpoint.get("is_final"))
    r1 = checkpoint.get("r1") or {}
    if is_final:
        if r1.get("ok"):
            return Decision("continue", "final_r1_clean")
        return Decision("continue", "final_r1_dirty_ineligible")

    informed = tuple(r1.get("informed_codes") or ())
    if informed and persistence >= config.patience_checkpoints:
        return Decision("kill", "r1_informed_persistent:" + ",".join(sorted(informed)))

    live = int(checkpoint.get("live_particles") or 0)
    remaining = int(checkpoint.get("budget_remaining_nfe") or 0)
    if live < int(config.raw.get("budget", {}).get("k_max_eff", 8)) and remaining > 0:
        return Decision("continue", "eligible_for_clone")
    return Decision("continue", "no_clone_capacity")


@dataclass
class SteerGameLog:
    """Writer for one game's steer log (append-only JSONL)."""

    path: Path
    events: List[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self.path.unlink()

    def emit(self, kind: str, **payload) -> dict:
        event = {"game": payload.pop("game", None), "kind": kind, **payload}
        self.events.append(event)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")
        return event

    def meta(self, **payload) -> dict:
        return self.emit("meta", schema_version=SCHEMA_VERSION, **payload)

    def done(self, **payload) -> dict:
        return self.emit("done", **payload)


def read_game_log(path: Path | str) -> List[dict]:
    """All events of one game log, in file order."""
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(json.loads(line))
    return out


def replay_game(events: Sequence[dict], config: VerdictConfig) -> List[Decision]:
    """Re-derive EVERY decision from the log alone (no GPU, no model, no rustc).

    Each ``checkpoint`` event is fed to :func:`decide` with its own logged
    evidence plus a persistence count RECOMPUTED from the checkpoint history
    (never read from a stored field), and the derived action is compared with
    the ``decision`` event the loop wrote. The policy inputs the loop recorded
    on the checkpoint (live particle count, remaining budget) are treated as
    logged STATE, not as conclusions — the conclusion (action/reason) is always
    recomputed, which is what makes a tampered verdict detectable.

    Raises :class:`ReplayMismatch` on the first divergence.
    """
    decisions: List[Decision] = []
    recorded: List[dict] = []
    history: List[dict] = []
    for ev in events:
        kind = ev.get("kind")
        if kind == "checkpoint":
            history.append(ev)
            d = decide(ev, config=config,
                       persistence=persistence_of(
                           history, ev.get("particle", 0),
                           ev.get("resume_epoch", 0)))
            decisions.append(d)
        elif kind == "decision":
            recorded.append(ev)

    if len(recorded) != len(decisions):
        raise ReplayMismatch(
            f"log has {len(recorded)} decision events but replay derived "
            f"{len(decisions)} — the log is missing checkpoints or decisions")
    for i, (rec, got) in enumerate(zip(recorded, decisions)):
        if rec.get("action") != got.action:
            raise ReplayMismatch(
                f"decision #{i} (particle={rec.get('particle')} "
                f"step={rec.get('step')}): logged {rec.get('action')!r} "
                f"({rec.get('reason')!r}) but replay derived {got.action!r} "
                f"({got.reason!r}) — evidence and decision disagree")
    return decisions


def log_sha256(path: Path | str) -> str:
    """sha256 of a game log file (recorded in the run record)."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
