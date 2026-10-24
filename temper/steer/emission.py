"""The emission rules M3.0's and M3.1's cells are scored by (design §3.3, §2.3).

A cell's per-problem outcome is the container verdict of ONE emitted canvas, so
the emission rule is part of the cell definition — it must be stated once and
used by both the in-loop engine and the offline counterfactual computation, or
the comparison between them measures the rule's implementation instead of the
mechanism.

Three rules, deliberately different, because the cells isolate three mechanisms:

* **structural** (C-null, design rev1 §4.1): emit the T=0.2 STRUCTURAL canvas
  if it is R1-clean, else the lowest-temperature clean canvas. This is the
  "diversity ladder only" control — it keeps the banked 11-passer cohort on
  exactly the canvas M2.4b banked (T=0.2) and only falls back when that canvas
  is compile-dead.
* **screen** (C-screen and C-full, config v2 ``emission``):
  ``best_r1_clean_by_mean_commit_conf_ties_lowest_temp`` — among the R1-clean
  canvases, the highest mean commit confidence wins; ties go to the lowest
  temperature, then the lowest particle index. This is the ranking design §2.2
  fixes for the final answer ("clean > mean_commit_conf").
* **property** (M3.1, config v3 ``emission``,
  ``property_pass_rate_then_mean_commit_conf_ties_lowest_temp``): among the
  R1-clean canvases, rank by the R3 property evidence — **pass_rate over
  non-timeout harnesses**, then mean commit confidence, then lowest
  temperature, then lowest index. Design §2.3 pre-registers the leading term
  and keeps the screen's tie order, so `property` and `screen` differ ONLY in
  the leading term: on canvases with no property evidence the two rules agree
  by construction (pinned by test). That is what makes P-prop-screen vs
  C-screen a pure rule delta over held-fixed draws.

All rules fall back to the max-confidence canvas when NO canvas is R1-clean
(design §3.3 item 2: the game always emits one answer, and its compile failure
is scored honestly rather than hidden).

R1-clean here means the FINAL-checkpoint hard screen
(:func:`temper.steer.verdict.final_checkpoint_clean` on a zero-hole ghost
closure) — exact on the measured corpus, 0/44 false flags at full commit.

**The three-tier property ranking (§7 clarification, operator ruling
2026-10-24, Q1).** Design §2.3 defines ``pass_rate`` over non-timeout harnesses,
which leaves the all-timeout canvas with no defined pass_rate. Abstention can
neither beat measured-good (absence of evidence is not better than evidence)
nor rank below measured-bad (that would punish abstention, which §2.3 forbids),
so the ranking has three tiers:

  tier 0  measured pass_rate > 0      — ordered by pass_rate desc, then conf,
                                        temp, index
  tier 1  nothing measured (pass_rate undefined)  — ordered by conf, temp, index
  tier 2  measured pass_rate == 0     — ordered by conf, temp, index

In an ensemble where no canvas is R1-clean the pool is the whole ensemble, and
if none of those has property evidence every candidate lands in one tier — the
rule then reduces EXACTLY to the screen rule, which is the property we want.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

__all__ = [
    "EMISSION_PROPERTY",
    "EMISSION_SCREEN",
    "STRUCTURAL_TEMP",
    "property_rank_key",
    "property_tier",
    "rank_by_rule",
    "rank_property",
    "rank_screen",
    "select_by_rule",
    "select_property",
    "select_screen",
    "select_structural",
]

# The structural (emission-preferred) temperature, from the rev2 ladder.
STRUCTURAL_TEMP = 0.2

# The two ranking rules' names as recorded in the frozen verdict config. A cell
# that asks for a rule the config does not freeze is refused rather than
# silently defaulted, so the record always names the rule that actually ran.
EMISSION_SCREEN = "best_r1_clean_by_mean_commit_conf_ties_lowest_temp"
EMISSION_PROPERTY = "property_pass_rate_then_mean_commit_conf_ties_lowest_temp"


def _confidence_rank(candidates: Sequence) -> List:
    """Highest mean confidence first, then lowest temperature, then index."""
    return sorted(
        candidates,
        key=lambda c: (-(c.mean_commit_conf if c.mean_commit_conf is not None
                         else -1e9), c.temperature, c.index),
    )


def select_screen(candidates: Sequence) -> Optional[object]:
    """``best_r1_clean_by_mean_commit_conf_ties_lowest_temp``."""
    pool = [c for c in candidates if c.final_clean] or list(candidates)
    return _confidence_rank(pool)[0] if pool else None


def rank_screen(candidates: Sequence) -> List:
    """The full screen ranking (same pool/order rule as :func:`select_screen`).

    Exposed so an analysis can report WHERE a flip came from in the ranking
    (rank 1 vs rank 3 of a 8-canvas ensemble), not just the winner.
    """
    pool = [c for c in candidates if c.final_clean] or list(candidates)
    return _confidence_rank(pool)


def property_tier(candidate) -> int:
    """The ranking tier of one candidate's R3 evidence (see the module docstring).

    ``property_pass_rate`` is ``None`` for a candidate with no measured harness
    call — the abstention tier. A candidate the property rung never touched
    (``property_pass_rate`` absent entirely, e.g. an offline analysis of a cell
    that ran no R3) is treated the same way, so a mixed ensemble cannot silently
    rank untouched canvases as if they had failed.
    """
    rate = getattr(candidate, "property_pass_rate", None)
    if rate is None:
        return 1
    return 0 if rate > 0 else 2


def property_rank_key(candidate) -> tuple:
    """The pre-registered v3 sort key: (tier, pass_rate, conf, temp, index).

    Inside a tier the ordering is identical to the screen's (confidence desc,
    temp asc, index asc) with the tier's own evidence in front, so the rule is
    strictly the screen rule plus one leading term.
    """
    rate = getattr(candidate, "property_pass_rate", None)
    conf = candidate.mean_commit_conf
    return (
        property_tier(candidate),
        -(rate if rate is not None else 0.0),
        -(conf if conf is not None else -1e9),
        candidate.temperature,
        candidate.index,
    )


def rank_property(candidates: Sequence) -> List:
    """The full v3 property ranking (same pool rule as the screen)."""
    pool = [c for c in candidates if c.final_clean] or list(candidates)
    return sorted(pool, key=property_rank_key)


def select_property(candidates: Sequence) -> Optional[object]:
    """``property_pass_rate_then_mean_commit_conf_ties_lowest_temp``.

    Falls back to max confidence when no canvas is R1-clean (§3.3 item 2),
    exactly as the other two rules do.
    """
    ranked = rank_property(candidates)
    return ranked[0] if ranked else None


# Which rule name maps to which implementation. Keyed by the frozen config's
# ``emission`` string so a cell cannot run a rule it did not record, and a
# record cannot name a rule the code does not implement (refused, not defaulted).
_RULES = {
    EMISSION_SCREEN: (rank_screen, select_screen),
    EMISSION_PROPERTY: (rank_property, select_property),
}


def _rule(rule: str):
    try:
        return _RULES[rule]
    except KeyError:
        raise ValueError(
            f"emission rule {rule!r} is not implemented (known: "
            f"{sorted(_RULES)}) — a cell may not run an unrecorded rule"
        ) from None


def rank_by_rule(candidates: Sequence, rule: str) -> List:
    """Rank by the rule NAMED in the frozen config (see :data:`_RULES`)."""
    return _rule(rule)[0](candidates)


def select_by_rule(candidates: Sequence, rule: str) -> Optional[object]:
    """Select by the rule NAMED in the frozen config (see :data:`_RULES`)."""
    return _rule(rule)[1](candidates)


def select_structural(candidates: Sequence,
                      structural_temp: float = STRUCTURAL_TEMP) -> Optional[object]:
    """Emit the structural T=0.2 canvas if R1-clean, else the lowest-T clean.

    Ties on temperature (the rev2 ladder places 0.8 and 0.95 twice) go to the
    lowest particle index, and the structural canvas means a canvas AT
    ``structural_temp`` — the ladder's first slot. No clean canvas at all falls
    back to max confidence (§3.3 item 2).
    """
    if not candidates:
        return None
    clean = [c for c in candidates if c.final_clean]
    if not clean:
        return _confidence_rank(list(candidates))[0]
    structural = [c for c in clean
                  if abs(c.temperature - structural_temp) < 1e-9]
    if structural:
        return sorted(structural, key=lambda c: c.index)[0]
    return sorted(clean, key=lambda c: (c.temperature, c.index))[0]
