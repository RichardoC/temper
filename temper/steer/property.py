"""R3 — the property rung: frozen proptest harnesses vs one candidate canvas.

M3.1 design §2.2/§2.3. R3 is the first rung whose evidence comes from
**execution** rather than from the compiler: a proptest harness drafted at
inference time from the problem's contract is run against a *completed*
candidate canvas by cargo (``temper.harness.runner.run_candidate`` — the M0
protocol-v1 machinery, 256 cases, seatbelt sandbox).

Three properties of this module are load-bearing and pre-registered:

**Final window only.** D4 measured mid-canvas informed recall ~0, and P3
confirmed reordering mid-decode buys nothing at this scale, so R3 never fires
before the final checkpoint and never acts mid-canvas. The caller enforces it
(``temper.steer.loop`` fires it once per R1-clean canvas during emission);
this module has no notion of a checkpoint.

**Unmeasurable is not wrong.** Design §2.3: "``timeout`` counts as abstention,
never as failure (a candidate the harness cannot judge is unknown, not wrong —
and aborting ≠ failing is the difference between evidence and punishment)."
Generalizing that sentence to every status the protocol can produce, a call is
MEASURED only when the harness actually judged the candidate: ``pass`` or
``fail``. ``timeout``, ``abort`` (proptest gave up generating inputs),
``error`` (harness-side failure) and ``compile_error`` (the harness could not
be linked against this candidate) are all ABSTENTIONS. The distinction is
recorded per call, never smoothed: the §7 clarification row (operator ruling
2026-10-24, Q1) requires the abstention classes to stay separately visible so
G4 can quantify them and so a future rev can re-tier one of them on measured
backing.

**Visibility normalization.** ``properties.rs`` is a sibling module and imports
the candidate by path (``use crate::candidate::<fn>``), but MultiPL-E prompts
emit ``fn``, not ``pub fn``. R1 never cared (it type-checks one crate); R3
cannot link a private item — measured on a banked C-null canvas, the unmodified
ghost program gives ``E0603``. R3 therefore renders the SAME ghost program with
the prompted function widened to ``pub``. That is a visibility widening only:
the body, the signature types and the prompter's contract are untouched, so the
program R3 executes is the program R1 judged (plus visibility), and the
attribution stays clean.

Held-in discipline (D5): the harness text comes from the FROZEN manifest, never
from the task dict, and this module sees a :class:`SteerProblemView`
(prompt + derived fn name) exactly as the other rungs do. The MultiPL-E
``tests`` string is never read here — pinned by test.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from temper.steer.verdict import (
    CONFIG_SCHEMA,
    GhostProgram,
    ghost_close,
    load_verdict_config,
)
from temper.steer.view import SteerProblemView

__all__ = [
    "ABSTAIN_STATUSES",
    "HARNESS_FREEZE_PATH",
    "HARNESS_FREEZE_SCHEMA",
    "MEASURED_STATUSES",
    "Harness",
    "HarnessFreezeError",
    "R3Call",
    "R3Verdict",
    "R3Log",
    "compute_harness_freeze_sha256",
    "load_harness_freeze",
    "make_pub_program",
    "run_r3",
    "strategy_hash_of",
]

# The protocol-v1 statuses that count as EVIDENCE about a candidate, and the
# ones that count as abstention. See the module docstring (§2.3 + §7 Q1).
MEASURED_STATUSES = ("pass", "fail")
ABSTAIN_STATUSES = ("timeout", "abort", "error", "compile_error")

HARNESS_FREEZE_SCHEMA = "harness-freeze/1"
# Repo-root freeze artifact (D7's second sha; the first is the verdict config).
HARNESS_FREEZE_PATH = (Path(__file__).resolve().parents[2]
                       / "harness_freeze.json")

# R3's default workdir is deliberately NOT the M0 default (temper/.work/gen):
# a freeze-validation or screen pass must never wipe a directory another
# process is mid-build in.
DEFAULT_R3_WORKDIR = (Path(__file__).resolve().parents[2]
                      / "temper" / ".work" / "r3")

_FN_HEADER_ANY_RE = re.compile(r"^(\s*)(pub\s+)?(fn\s+)")


class HarnessFreezeError(RuntimeError):
    """Raised when the harness freeze manifest is missing, malformed, or no
    longer matches the sha it records (the harness set is a pre-commitment)."""


def strategy_hash_of(properties_source: str) -> str:
    """sha256 of a harness source (its identity in every R3 record).

    Same digest as :func:`temper.harness.emit.strategy_hash_of` — re-exported
    here so a caller that only reads the freeze manifest does not have to
    import the emitting machinery to name a harness.
    """
    return hashlib.sha256(properties_source.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------
# D7 (second freeze): the harness set
# --------------------------------------------------------------------------


def compute_harness_freeze_sha256(payload: dict) -> str:
    """Canonical sha256 of a freeze manifest (the object WITHOUT ``sha256``).

    Same discipline as the verdict config: sorted keys, no whitespace, so
    formatting churn is free and any semantic edit is detected.
    """
    body = {k: v for k, v in payload.items() if k != "sha256"}
    blob = json.dumps(body, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Harness:
    """One frozen property harness for one problem.

    ``source`` is the harness text as frozen; ``strategy_hash`` is its sha256
    and is what every R3 call record carries, so a verdict is always
    attributable to the exact property text that produced it.
    """

    problem: str
    slot: int
    seed: int
    source: str
    strategy_hash: str
    source_path: str = ""
    replaced: bool = False

    def record(self) -> dict:
        return {"slot": self.slot, "seed": self.seed,
                "strategy_hash": self.strategy_hash,
                "source_path": self.source_path, "replaced": self.replaced}


def load_harness_freeze(path: Path | str | None = None) -> dict:
    """Load + VERIFY the frozen harness set; return ``{problem: [Harness]}``.

    Raises :class:`HarnessFreezeError` when the manifest is missing, malformed,
    tampered with, or when a harness's text no longer hashes to the
    ``strategy_hash`` the manifest recorded (each harness is re-hashed from its
    ``source_path``, so editing a case file cannot pass silently).
    """
    p = Path(path) if path is not None else HARNESS_FREEZE_PATH
    if not p.exists():
        raise HarnessFreezeError(f"harness freeze manifest not found: {p}")
    raw = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise HarnessFreezeError(f"{p}: manifest must be a JSON object")
    if raw.get("schema") != HARNESS_FREEZE_SCHEMA:
        raise HarnessFreezeError(
            f"{p}: schema {raw.get('schema')!r} != {HARNESS_FREEZE_SCHEMA!r}")
    recorded = raw.get("sha256")
    if not recorded:
        raise HarnessFreezeError(
            f"{p}: no 'sha256' field — the harness set was never frozen")
    actual = compute_harness_freeze_sha256(raw)
    if actual != recorded:
        raise HarnessFreezeError(
            f"{p}: TAMPERED — payload sha256 {actual} != recorded {recorded}. "
            "The frozen harness set is a pre-commitment: re-freeze with a "
            "labeled version bump, never edit in place.")
    out: Dict[str, List[Harness]] = {}
    root = p.resolve().parent
    for prob in raw.get("problems", []):
        name = prob["name"]
        harnesses: List[Harness] = []
        for h in prob.get("harnesses", []):
            src_path = Path(h["source_path"])
            if not src_path.is_absolute():
                src_path = (root / src_path).resolve()
            source = src_path.read_text(encoding="utf-8")
            if strategy_hash_of(source) != h["strategy_hash"]:
                raise HarnessFreezeError(
                    f"{src_path}: text sha256 does not match the frozen "
                    f"strategy_hash {h['strategy_hash']} — the harness case "
                    "file was edited after the freeze")
            harnesses.append(Harness(
                problem=name, slot=int(h["slot"]), seed=int(h.get("seed", 0)),
                source=source, strategy_hash=h["strategy_hash"],
                source_path=str(src_path), replaced=bool(h.get("replaced"))))
        out[name] = harnesses
    return out


# --------------------------------------------------------------------------
# the program R3 executes
# --------------------------------------------------------------------------


def make_pub_program(view: SteerProblemView, canvas_text: str) -> GhostProgram:
    """The R1 ghost program with the prompted function widened to ``pub``.

    ``properties.rs`` imports ``crate::candidate::<fn>``; a private item cannot
    be imported (E0603, measured on a banked C-null canvas). Widening the ONE
    header line inside the prompt region is the minimum change that makes the
    harness able to link — ``pub`` on a module-level free function changes no
    type, no body and no call, so the program R3 runs is the program R1 judged.

    Refuses a canvas that still carries mask residue: R3 requires a completed,
    zero-hole canvas (design §2.2), and a holed program would be a ghost
    closure again, i.e. R1's judgement re-run with execution noise on top.
    """
    if "<|mask|>" in canvas_text:
        raise ValueError(
            f"{view.name}: R3 requires a completed, zero-hole canvas — the "
            "rendered text still carries mask residue")
    program = ghost_close(view, canvas_text)
    lines = program.text.splitlines()
    hit = None
    for i in range(min(program.prompt_lines, len(lines))):
        m = _FN_HEADER_ANY_RE.match(lines[i])
        if m and re.match(re.escape(view.fn_name) + r"\b", lines[i][m.end():]):
            hit = i
    if hit is None:
        raise ValueError(
            f"{view.name}: no 'fn {view.fn_name}' header line inside the prompt "
            "region — the R3 program cannot be given a public API")
    m = _FN_HEADER_ANY_RE.match(lines[hit])
    if not m.group(2):
        lines[hit] = m.group(1) + "pub " + lines[hit][len(m.group(1)):]
    text = "\n".join(lines) + ("\n" if program.text.endswith("\n") else "")
    return GhostProgram(
        text=text,
        prompt_lines=program.prompt_lines,
        hole_lines=program.hole_lines,
        tail_lines=program.tail_lines,
        fragment_exact=program.fragment_exact,
    )


# --------------------------------------------------------------------------
# one R3 call / one verdict
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class R3Call:
    """One (canvas, harness) execution — the D2 record of an R3 event.

    Carries exactly the fields design §2.4 asks D2 to log: the problem, the
    harness sha, which canvas, the protocol status, the duration, and the
    proptest-shrunk ``minimal_input`` (empty unless the harness failed).
    """

    problem: str
    harness_sha: str
    canvas: str                     # canvas reference: particle index or label
    status: str
    duration_s: float
    minimal_input: str = ""
    reason: str = ""
    cases: int = 0

    @property
    def measured(self) -> bool:
        return self.status in MEASURED_STATUSES

    def as_event(self, **extra) -> dict:
        ev = {
            "problem": self.problem,
            "harness_sha": self.harness_sha,
            "canvas": self.canvas,
            "status": self.status,
            "duration_s": round(self.duration_s, 3),
            "minimal_input": self.minimal_input,
            "measured": self.measured,
            "cases": self.cases,
        }
        if self.reason:
            ev["reason"] = self.reason[:400]
        ev.update(extra)
        return ev


@dataclass(frozen=True)
class R3Verdict:
    """The property evidence about ONE canvas, aggregated over its harness set.

    ``pass_rate`` is defined over MEASURED calls only (design §2.3) and is
    ``None`` when nothing was measured — the abstention case the §7 Q1
    clarification covers with its own ranking tier.
    """

    calls: Tuple[R3Call, ...] = ()

    @property
    def n_calls(self) -> int:
        return len(self.calls)

    @property
    def n_passed(self) -> int:
        return sum(1 for c in self.calls if c.status == "pass")

    @property
    def n_measured(self) -> int:
        return sum(1 for c in self.calls if c.measured)

    @property
    def n_abstained(self) -> int:
        return self.n_calls - self.n_measured

    @property
    def pass_rate(self) -> Optional[float]:
        if self.n_measured == 0:
            return None
        return self.n_passed / self.n_measured

    @property
    def all_abstained(self) -> bool:
        return self.n_calls > 0 and self.n_measured == 0

    @property
    def wall_s(self) -> float:
        return round(sum(c.duration_s for c in self.calls), 3)

    @property
    def statuses(self) -> Tuple[str, ...]:
        return tuple(c.status for c in self.calls)

    @property
    def minimal_inputs(self) -> Tuple[str, ...]:
        return tuple(c.minimal_input for c in self.calls if c.minimal_input)

    def as_record(self) -> dict:
        return {
            "pass_rate": self.pass_rate,
            "n_calls": self.n_calls,
            "n_passed": self.n_passed,
            "n_measured": self.n_measured,
            "n_abstained": self.n_abstained,
            "statuses": list(self.statuses),
            "minimal_inputs": list(self.minimal_inputs),
            "wall_s": self.wall_s,
        }


def _status_of(outcome) -> str:
    """Reduce a :class:`temper.harness.runner.RunOutcome` to a protocol status."""
    if not outcome.compile.ok:
        # a compile-stage failure is its own abstention class, not a verdict
        return "compile_error" if outcome.compile.status != "timeout" \
            else "timeout"
    if outcome.test is None:
        return "error"
    status = outcome.test.status
    return status if status else "error"


def run_r3(
    program_text: str,
    harnesses: Sequence[Harness],
    *,
    canvas_ref: str = "",
    problem: str = "",
    cases: int = 256,
    timeout_ms: int = 2000,
    sandbox: str | None = None,
    workdir: Path | str | None = None,
) -> R3Verdict:
    """Run the frozen harness set against ONE candidate program (R3).

    Every call is recorded — including the ones that abstain — because the
    abstention classes are exactly what the §7 Q1 ruling requires to stay
    measurable. ``program_text`` is the *pub-normalized* program from
    :func:`make_pub_program`; a harness that cannot link against it is a
    ``compile_error`` abstention, never a failure.
    """
    from temper.harness.runner import run_candidate

    wd = Path(workdir) if workdir is not None else DEFAULT_R3_WORKDIR
    calls: List[R3Call] = []
    for h in harnesses:
        t0 = time.perf_counter()
        try:
            out = run_candidate(program_text, h.source, cases=cases,
                                timeout_ms=timeout_ms, sandbox=sandbox,
                                workdir=wd)
            status = _status_of(out)
            reason = (out.test.reason if out.test is not None
                      else (out.compile.diagnostics[0].message
                            if out.compile.diagnostics else ""))
            minimal = out.test.minimal_input if out.test is not None else ""
        except Exception as exc:  # a harness crash must never kill a game
            status, reason, minimal = "error", f"{type(exc).__name__}: {exc}", ""
        calls.append(R3Call(
            problem=problem or h.problem, harness_sha=h.strategy_hash,
            canvas=canvas_ref, status=status,
            duration_s=time.perf_counter() - t0,
            minimal_input=minimal, reason=str(reason or ""), cases=cases,
        ))
    return R3Verdict(calls=tuple(calls))


class R3Log:
    """Append-only JSONL writer for R3 events (D2's R3 schema).

    Used by the offline cells (freeze validation, P-prop-screen), which have no
    steer game log to write into. In-loop R3 events go to the game's own D2 log
    with ``kind="r3"`` and the same field names, so one reader covers both.
    """

    def __init__(self, path: Path | str | None = None):
        self.path = Path(path) if path is not None else None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if self.path.exists():
                self.path.unlink()
        self.events: List[dict] = []

    def write(self, event: dict) -> dict:
        self.events.append(event)
        if self.path is not None:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event) + "\n")
        return event

    def record_verdict(self, verdict: R3Verdict, **extra) -> None:
        for call in verdict.calls:
            self.write({"kind": "r3", **call.as_event(**extra)})
