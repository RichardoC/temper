"""Held-in guard: the ONLY problem data the steer-side verifier may see (M3.0 D5).

Design §3.1: at steer time the verifier may read the problem prompt (the model
sees it too — symmetric information), the canvas, and rustc's stderr. It may
NEVER read the MultiPL-E ``tests`` string, container output, or any scoring
fixture (docs/benchmarks-decision.md §1; UMF's public/private-reward mismatch
is the named precedent).

Enforcement is structural, not procedural: the reward-side modules receive a
:class:`SteerProblemView` built by :meth:`SteerProblemView.from_task`, whose
field allowlist is exactly ``ALLOWED_TASK_FIELDS``. Extra keys in the task dict
(notably ``tests``) are dropped at construction, so a downstream module cannot
read them even by accident — there is nothing to read.
"""

from __future__ import annotations

from dataclasses import dataclass

from temper.multipl_e.export import _fn_header_line

__all__ = ["ALLOWED_TASK_FIELDS", "SteerProblemView"]

# The complete set of task-dict keys the steer side may copy. Anything else
# (tests, doctests, original, stop_tokens, ...) never crosses the boundary.
ALLOWED_TASK_FIELDS = ("name", "prompt")


@dataclass(frozen=True)
class SteerProblemView:
    """Prompt-side view of one problem: name + prompt, and nothing else.

    ``fn_name``/``signature_line`` are DERIVED from ``prompt`` (the last ``fn``
    header, the same regex the export instrument uses), so the whole view is
    reconstructible from allowlisted text alone.
    """

    name: str
    prompt: str
    fn_name: str
    signature_line: str

    @classmethod
    def from_task(cls, task: dict) -> "SteerProblemView":
        """Allowlist constructor: copy exactly ALLOWED_TASK_FIELDS.

        Raises ValueError when the prompt carries no fn header (the export
        instrument cannot join that shape either) — refusing beats handing the
        reward side a view it cannot render.
        """
        payload = {k: task[k] for k in ALLOWED_TASK_FIELDS}
        m = _fn_header_line(payload["prompt"])
        if m is None:
            raise ValueError(
                f"problem {payload.get('name')!r}: prompt carries no fn header — "
                "not steerable (the export instrument rejects this shape too)")
        return cls(
            name=payload["name"],
            prompt=payload["prompt"],
            fn_name=m.group(1),
            signature_line=m.group(0).rstrip(),
        )

    def as_payload(self) -> dict:
        """Serializable boundary payload (what a log/record may carry)."""
        return {
            "name": self.name,
            "prompt": self.prompt,
            "fn_name": self.fn_name,
            "signature_line": self.signature_line,
        }
