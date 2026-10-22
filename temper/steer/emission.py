"""The two emission rules M3.0's cells are scored by (design §3.3, §4.1).

A cell's per-problem outcome is the container verdict of ONE emitted canvas, so
the emission rule is part of the cell definition — it must be stated once and
used by both the in-loop engine (C-full) and the offline counterfactual
computation (C-null, C-screen), or the comparison between them measures the
rule's implementation instead of the mechanism.

Two rules, deliberately different, because §4.3 isolates two mechanisms:

* **structural** (C-null, design rev1 §4.1): emit the T=0.2 STRUCTURAL canvas
  if it is R1-clean, else the lowest-temperature clean canvas. This is the
  "diversity ladder only" control — it keeps the banked 11-passer cohort on
  exactly the canvas M2.4b banked (T=0.2) and only falls back when that canvas
  is compile-dead.
* **screen** (C-screen and C-full, config ``emission``):
  ``best_r1_clean_by_mean_commit_conf_ties_lowest_temp`` — among the R1-clean
  canvases, the highest mean commit confidence wins; ties go to the lowest
  temperature, then the lowest particle index. This is the ranking design §2.2
  fixes for the final answer ("clean > mean_commit_conf").

Both rules fall back to the max-confidence canvas when NO canvas is R1-clean
(design §3.3 item 2: the game always emits one answer, and its compile failure
is scored honestly rather than hidden).

R1-clean here means the FINAL-checkpoint hard screen
(:func:`temper.steer.verdict.final_checkpoint_clean` on a zero-hole ghost
closure) — exact on the measured corpus, 0/44 false flags at full commit.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

__all__ = ["STRUCTURAL_TEMP", "rank_screen", "select_screen", "select_structural"]

# The structural (emission-preferred) temperature, from the rev2 ladder.
STRUCTURAL_TEMP = 0.2


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
