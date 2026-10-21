"""NFE & budget metering for steering games (M3.0 D3, docs/m3-design.md §7).

Two numbers, never conflated (design §2.2):

  * **NFE** = every executed model forward. A steered game's NFE is the sum of
    its particles' model forwards; a kill stops accrual immediately. NFE is
    the budget axis and the axis every cross-cell comparison uses.
  * **verifier wall** is a separate axis. R1 type-checks do not buy model
    compute; they are metered as ``verifier_wall_s`` and capped by
    kill-criterion 4 (>10% of gen wall ⇒ redesign the cadence).

Envelope discipline: a game may never exceed its envelope, and a cell may
never exceed ``n_problems * envelope``. Both are asserted here, in the loop,
and re-checked on the done-line (design §6.2 item 7 — "the done-line totals
audited by the same scripts/m2_3_cells.py family extended to steered rows").

Schema compatibility: :class:`CellMeters` aggregates in the v1.2 shapes
(``nfe_total``, ``gen_wall_s_total``, ``verifier_wall_s_total``) so a steered
cell record exports and scores through the UNCHANGED instrument.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

__all__ = ["BudgetExceeded", "CellMeters", "GameMeters"]


class BudgetExceeded(RuntimeError):
    """Raised when a game would spend more NFE than its envelope allows."""


@dataclass
class GameMeters:
    """Live NFE/verifier counters for ONE steering game (one problem)."""

    envelope_nfe: int
    k: int
    nfe_model: int = 0
    nfe_verifier_calls: int = 0
    verifier_wall_s: float = 0.0
    per_particle_nfe: Dict[int, int] = field(default_factory=dict)

    def spend_model(self, particle: int, nfe: int = 1) -> int:
        """Account one executed model forward; raise on envelope overflow."""
        if nfe < 0:
            raise ValueError("nfe delta must be >= 0")
        if self.nfe_model + nfe > self.envelope_nfe:
            raise BudgetExceeded(
                f"game would spend {self.nfe_model + nfe} NFE > envelope "
                f"{self.envelope_nfe} (particle {particle})")
        self.nfe_model += nfe
        self.per_particle_nfe[particle] = self.per_particle_nfe.get(particle, 0) + nfe
        return self.nfe_model

    def spend_verifier(self, wall_s: float, calls: int = 1) -> None:
        """Account verifier work (never NFE)."""
        if wall_s < 0 or calls < 0:
            raise ValueError("verifier accounting must be non-negative")
        self.verifier_wall_s += wall_s
        self.nfe_verifier_calls += calls

    @property
    def remaining_nfe(self) -> int:
        return max(0, self.envelope_nfe - self.nfe_model)

    @property
    def verifier_fraction_of_budget_wall(self) -> Optional[float]:
        """Verifier wall as a fraction of the *nominal* canvas wall this game
        may consume (envelope NFE x s/NFE). Kill-criterion 4 watches this."""
        nominal = self.envelope_nfe * S_PER_NFE
        return self.verifier_wall_s / nominal if nominal > 0 else None

    def as_record(self) -> dict:
        return {
            "envelope_nfe": self.envelope_nfe,
            "nfe_model": self.nfe_model,
            "nfe_verifier_calls": self.nfe_verifier_calls,
            "verifier_wall_s": round(self.verifier_wall_s, 3),
            "per_particle_nfe": {str(k): v for k, v in sorted(self.per_particle_nfe.items())},
        }


# Canvas wall per NFE on this host (M2.4a/b least-squares fit; D1 uses the
# same constant so the amortization tables agree).
S_PER_NFE = 1.035


@dataclass
class CellMeters:
    """Cell-level aggregation in the v1.2 record shapes (D3 acceptance)."""

    n_problems: int
    envelope_nfe: int
    games: List[GameMeters] = field(default_factory=list)

    def add(self, game: GameMeters) -> None:
        self.games.append(game)

    @property
    def nfe_total(self) -> int:
        return sum(g.nfe_model for g in self.games)

    @property
    def nfe_verifier_calls_total(self) -> int:
        return sum(g.nfe_verifier_calls for g in self.games)

    @property
    def verifier_wall_s_total(self) -> float:
        return round(sum(g.verifier_wall_s for g in self.games), 3)

    @property
    def envelope_ceiling(self) -> int:
        return self.n_problems * self.envelope_nfe

    def assert_within_envelope(self) -> None:
        """The D3 acceptance assert: a cell never exceeds its envelope."""
        if self.nfe_total > self.envelope_ceiling:
            raise BudgetExceeded(
                f"cell spent {self.nfe_total} NFE > {self.n_problems} problems "
                f"x {self.envelope_nfe} envelope = {self.envelope_ceiling}")

    def as_record(self) -> dict:
        return {
            "nfe_total": self.nfe_total,
            "nfe_verifier_calls_total": self.nfe_verifier_calls_total,
            "verifier_wall_s_total": self.verifier_wall_s_total,
            "nfe_envelope_per_problem": self.envelope_nfe,
            "nfe_envelope_ceiling": self.envelope_ceiling,
            "n_problems": self.n_problems,
            "games": [g.as_record() for g in self.games],
        }


def aggregate_nfe(particles: Iterable) -> dict:
    """v1.2-compatible done-line meters from a list of ParticleResult-likes.

    ``nfe_total`` reads each particle's model forwards (``nfe_total`` property
    when present, else ``nfe``); verifier axes aggregate additively and are
    reported alongside, never inside the NFE number.
    """
    nfe = 0
    vcalls = 0
    vwall = 0.0
    n = 0
    for p in particles:
        n += 1
        nfe += int(getattr(p, "nfe_total", None)
                   or getattr(p, "nfe", None) or 0)
        vcalls += int(getattr(p, "nfe_verifier_calls", 0) or 0)
        vwall += float(getattr(p, "verifier_wall_s", 0.0) or 0.0)
    return {
        "n_particles": n,
        "nfe_total": nfe,
        "nfe_verifier_calls_total": vcalls,
        "verifier_wall_s_total": round(vwall, 3),
    }
