"""The verifier ladder the steer loop fires at checkpoints (M3.0 design §3).

Rungs, cheapest first:

  R0 *shape* (free, pure python): fence integrity, prose after the closing
     fence (the HumanEval_17 class), mask-residue share vs the expected
     commit schedule, prompted-fn header presence, final-extraction health.
  R1 *ghost type-check*: the canvas's fenced code region, ghost-closed
     deterministically (every contiguous masked run -> ``todo!()``, plus a
     balanced-closer tail), type-checked as a library crate with stable
     rustc (``--crate-type lib --emit=metadata``). ~35 ms, spawn-dominated.
  R2 *anti-stub*: committed stub markers — commits are sticky, so a committed
     ``todo!(`` can never self-repair (a sound kill category).

The load-bearing measurement behind the design (docs/m3-design.md §3.2): R1's
raw "any error" verdict is UNUSABLE mid-canvas (the ghost closure manufactures
parse noise at hole boundaries — 32-42 of 44 OK canvases flagged in the
pilot), while the *informed-class* filter used here (E0277/E0282/E0283/
E0507/E0382-style substantive diagnostics whose primary span sits inside
committed content, away from hole-adjacent lines) measured 0-2/44 false kills
at every simulated commit level and 36/36 detection at full commit. D4
re-measures the same curve on REAL canvases before any C-cell.

Held-in discipline: this module never sees the MultiPL-E ``tests`` string. Its
entry points take a :class:`temper.steer.view.SteerProblemView` (allowlisted
prompt + derived fn name), the rendered canvas, and rustc's stderr.

The informed-class list and the patience/verdict parameters live in the
versioned ``steer_verdict_config.json`` (design §7 D7) — frozen before any
C-cell and sha-recorded in every run record. :func:`load_verdict_config`
verifies the file's embedded payload sha, so a post-freeze edit is DETECTED,
not silent.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from temper.model.extract import extract_rust
from temper.multipl_e.export import to_body_fragment
from temper.steer.view import SteerProblemView

__all__ = [
    "CONFIG_SCHEMA",
    "DEFAULT_CONFIG_PATH",
    "GhostProgram",
    "R0Verdict",
    "R1Verdict",
    "RustcError",
    "VerdictConfig",
    "VerdictConfigError",
    "compute_config_sha256",
    "body_fragment",
    "code_region",
    "fill_holes",
    "final_checkpoint_clean",
    "find_rustc",
    "ghost_close",
    "informed_fatal_evidence",
    "load_verdict_config",
    "r0_shape",
    "r1_typecheck",
    "r2_stub_markers",
]

CONFIG_SCHEMA = "steer-verdict-config/1"

# Repo-root config path (design §7 D7: committed, sha-recorded per cell).
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "steer_verdict_config.json"

_RUSTC_CANDIDATES = (
    "/opt/homebrew/opt/rustup/bin/rustc",
    "~/.cargo/bin/rustc",
)

# Masked canvas positions decode to this residue (verified against the Dream
# tokenizer: MASK_TOKEN_ID 151666 renders as "<|mask|>"). Runs of it are the
# §3.1 hole runs.
_MASK_RESIDUE_RE = re.compile(r"(?:\s*<\|mask\|>)+\s*")
HOLE_SUBSTITUTE = "todo!()"


class VerdictConfigError(RuntimeError):
    """Raised when the frozen verdict config is missing, unparsable, or its
    embedded payload sha does not match its content (tamper detection)."""


# --------------------------------------------------------------------------
# D7: frozen verdict config
# --------------------------------------------------------------------------


def compute_config_sha256(payload: dict) -> str:
    """Canonical sha256 of a config payload (the object WITHOUT ``sha256``).

    Canonical form: JSON with sorted keys, no whitespace — the digest does not
    depend on formatting choices, only on content.
    """
    body = {k: v for k, v in payload.items() if k != "sha256"}
    blob = json.dumps(body, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class VerdictConfig:
    """The frozen informed-class list + verdict parameters (design §7 D7).

    Every field is a pre-commitment: which rustc error classes count as
    positive fatal evidence, how many consecutive checkpoints such evidence
    must persist, how close to a hole a span may sit and still count, which
    R0 flags are fatal, the emission rule, and the clone temp/lineage caps.
    Post-freeze edits are a labeled-cell decision, never silent.
    """

    version: str
    sha256: str                        # the verified payload digest
    informed_error_classes: Tuple[str, ...]
    patience_checkpoints: int
    hole_proximity_lines: int
    r0_flags_fatal: Tuple[str, ...]
    r0_mask_share_tolerance: float
    stub_markers: Tuple[str, ...]
    checkpoint_grid: Tuple[float, ...]
    emission: str
    clone_temp_step: float
    clone_temp_cap: float
    clone_cap_per_parent: int
    source: Optional[str] = None
    raw: dict = field(default_factory=dict, compare=False, repr=False)

    def as_payload(self) -> dict:
        """The serializable config record a run log carries (sha included)."""
        return dict(self.raw, sha256=self.sha256)

    @property
    def kill_clone_enabled(self) -> bool:
        """Whether the policy may EXECUTE a kill (rev2 ``kill_clone_enabled``).

        Read from the frozen payload so the D4 demotion is one fact stated once
        (config, sha-recorded per cell), never a per-run flag. Absent (v0/v1
        configs, where the demotion had not been applied) means enabled.

        The demotion is a single fact, so it has a single switch: with it OFF
        the loop neither executes a kill nor fires the rung ladder before the
        final checkpoint (D4 measured informed-class recall 0/5 at <=75%
        committed on real canvases — before the final step there is no evidence
        to act on, and every bit of the ladder's measured discriminative power
        sits at the final screen).
        """
        rev2 = self.raw.get("rev2") or {}
        return bool(rev2.get("kill_clone_enabled", True))


def load_verdict_config(path: Path | str | None = None) -> VerdictConfig:
    """Load + VERIFY the frozen verdict config.

    Raises :class:`VerdictConfigError` when the file is missing, malformed, or
    its content no longer matches the sha it records (tamper detection).
    """
    p = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not p.exists():
        raise VerdictConfigError(f"verdict config not found: {p}")
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise VerdictConfigError(f"{p}: unparsable verdict config: {exc}") from exc
    if not isinstance(raw, dict):
        raise VerdictConfigError(f"{p}: verdict config must be a JSON object")
    if raw.get("schema") != CONFIG_SCHEMA:
        raise VerdictConfigError(
            f"{p}: schema {raw.get('schema')!r} != {CONFIG_SCHEMA!r}")
    recorded = raw.get("sha256")
    if not recorded:
        raise VerdictConfigError(
            f"{p}: no 'sha256' field — the config was never frozen")
    actual = compute_config_sha256(raw)
    if actual != recorded:
        raise VerdictConfigError(
            f"{p}: TAMPERED — payload sha256 {actual} != recorded {recorded}. "
            "Post-freeze verdict-rule edits are a labeled-cell decision: bump "
            "the config version and re-record, never edit in place.")
    patience = raw.get("patience", {})
    r0 = raw.get("r0", {})
    clone = raw.get("clone", {})
    return VerdictConfig(
        version=str(raw.get("version", "unknown")),
        sha256=actual,
        informed_error_classes=tuple(raw.get("informed_error_classes", ())),
        patience_checkpoints=int(patience.get("consecutive_checkpoints", 2)),
        hole_proximity_lines=int(patience.get("hole_proximity_lines", 1)),
        r0_flags_fatal=tuple(r0.get("fatal_flags", ())),
        r0_mask_share_tolerance=float(r0.get("mask_share_tolerance", 0.35)),
        stub_markers=tuple(raw.get("r2_stub_markers", ())),
        checkpoint_grid=tuple(raw.get("checkpoint_grid", (0.25, 0.5, 0.75, 1.0))),
        emission=str(raw.get("emission", "best_r1_clean_else_max_conf")),
        clone_temp_step=float(clone.get("temp_step", 0.2)),
        clone_temp_cap=float(clone.get("temp_cap", 0.8)),
        clone_cap_per_parent=int(clone.get("clone_cap_per_parent", 2)),
        source=str(p),
        raw=raw,
    )


# --------------------------------------------------------------------------
# §3.1 hole substitution (masked runs -> todo!())
# --------------------------------------------------------------------------


def fill_holes(canvas_text: str) -> Tuple[str, List[int]]:
    """Replace every contiguous masked run with ``todo!()``.

    Returns (hole-filled text, 0-based line indices carrying a hole). The
    indices are consumed by :func:`ghost_close` so the informed-class filter
    can refuse diagnostics that sit on/next to a substitution (those are the
    ghost closure's own artifacts — the §3.2 negative result).
    """
    out: List[str] = []
    hole_lines: List[int] = []
    pos = 0
    line = 0
    for m in _MASK_RESIDUE_RE.finditer(canvas_text):
        chunk = canvas_text[pos:m.start()]
        out.append(chunk)
        line += chunk.count("\n")
        out.append(HOLE_SUBSTITUTE + "\n")
        hole_lines.append(line)
        line += 1
        pos = m.end()
    out.append(canvas_text[pos:])
    return "".join(out), hole_lines


# --------------------------------------------------------------------------
# R1: ghost closure
# --------------------------------------------------------------------------

_FENCE_RE = re.compile(r"```([a-zA-Z0-9_+#.-]*)[ \t]*\r?\n(.*?)```", re.DOTALL)
_OPEN_FENCE_RE = re.compile(r"```([a-zA-Z0-9_+#.-]*)[ \t]*[ \t]*\r?\n")
_CLOSE_OF = {"{": "}", "(": ")", "[": "]"}
_OPENERS = set(_CLOSE_OF)


@dataclass(frozen=True)
class GhostProgram:
    """A deterministically closed, type-checkable rendering of a partial canvas.

    ``text`` is the program handed to rustc; ``prompt_lines`` is how many of
    its leading lines are the problem prompt (always committed content);
    ``hole_lines`` are 0-based program line indices that are NOT evidence (the
    ``todo!()`` substitutions); ``tail_lines`` are the synthesized closers.
    """

    text: str
    prompt_lines: int
    hole_lines: Tuple[int, ...] = ()
    tail_lines: Tuple[int, ...] = ()
    fragment_exact: bool = True

    @property
    def committed_line_count(self) -> int:
        return len(self.text.splitlines()) - len(self.tail_lines)

    def is_committed_line(self, line_1based: int, hole_proximity: int = 0) -> bool:
        """True when a 1-based rustc span line is real evidence: inside the
        committed (non-tail) region and not within ``hole_proximity`` lines of
        a hole substitution."""
        if line_1based <= 0:
            return False
        idx = line_1based - 1
        if idx >= self.committed_line_count:
            return False
        for g in self.hole_lines:
            if abs(idx - g) <= hole_proximity:
                return False
        return True


def _balanced_body(body_text: str) -> Tuple[str, int]:
    """Close the body's unterminated delimiter runs -> (closed_text, n_tail_lines).

    Brace/paren/bracket scan (string-literal naive, the same heuristic the
    export instrument uses). The scan starts with the prompt's own unclosed
    ``{``, so a body fragment always leaves at least that one to close.
    """
    stack: List[str] = ["{"]
    for ch in body_text:
        if ch in _OPENERS:
            stack.append(ch)
        elif ch in _CLOSE_OF.values():
            if stack and _CLOSE_OF[stack[-1]] == ch:
                stack.pop()
    if not stack:
        return body_text, 0
    tail = "".join(_CLOSE_OF[c] for c in reversed(stack))
    return body_text + "\n" + tail, 1


def code_region(canvas_text: str) -> str:
    """The canvas's fenced code region (largest rust-ish fence wins; with an
    unterminated fence, everything after the opener).

    Mirrors the export instrument's extraction order so the steer-time render
    and the offline export agree on what "the canvas's code" is.
    """
    blocks = _FENCE_RE.findall(canvas_text)
    if blocks:
        rustish = [b for lang, b in blocks if lang.lower() in ("", "rust", "rs")]
        return max(rustish or [b for _, b in blocks], key=len)
    m = _OPEN_FENCE_RE.search(canvas_text)
    return canvas_text[m.end():] if m else canvas_text


def body_fragment(view: SteerProblemView, canvas_text: str) -> Tuple[str, bool]:
    """Reduce a canvas to the prompted function's BODY fragment + a hole map.

    The model emits a COMPLETE function (signature repeated) and the export
    instrument reduces it to a body fragment before the container ever sees it
    (``to_body_fragment``). The ghost closure must do the same, otherwise it
    feeds rustc ``prompt + full fn + closer`` = a duplicated item, and every
    row drowns in E0308/E0425 noise instead of its real diagnosis.

    Returns (body_text, exact) where ``exact`` is False when the fragment could
    not be recovered structurally (renamed/reshaped API, truncated canvas) and
    the fallback used is a best-effort region render.
    """
    filled, _hole_lines = fill_holes(canvas_text)
    ex = extract_rust(filled, must_define=[view.fn_name])
    if ex.ok and ex.code:
        frag = to_body_fragment(ex.code, view.prompt)
        if frag is not None:
            return frag, True
    # fallback: take everything after the last signature-line occurrence, or
    # the raw fenced region when the signature is not found at all
    region = code_region(filled)
    lines = region.splitlines()
    start = None
    for i, ln in enumerate(lines):
        if re.match(r"^\s*(?:pub\s+)?fn\s+" + re.escape(view.fn_name) + r"\b", ln):
            start = i
    if start is None:
        return region, False
    body = lines[start + 1:]
    # drop the fn's own trailing close if the canvas emitted one
    while body and not body[-1].strip():
        body.pop()
    if body and body[-1].strip() == "}":
        body.pop()
    return "\n".join(body), False


def ghost_close(
    view: SteerProblemView,
    canvas_text: str,
) -> GhostProgram:
    """The §3.1 ghost closure of one canvas: prompt verbatim + body fragment
    (holes -> ``todo!()``) + balanced closer tail.

    The body-fragment step is the SAME reduction the export instrument applies
    (:func:`body_fragment` -> ``to_body_fragment``), so the steer-time verdict
    and the offline container verdict read the same program. No vendored tests,
    no ``main``, no execution — ``--crate-type lib`` needs none of them, and the
    MultiPL-E ``tests`` string is never in scope.
    """
    prompt = view.prompt
    if not prompt.rstrip().endswith("{"):
        raise ValueError(
            f"{view.name}: prompt does not end with '{{' — the ghost closure "
            "assumes the MultiPL-E body-fragment prompt contract")
    body, exact = body_fragment(view, canvas_text)
    closed_body, n_tail = _balanced_body(body.rstrip("\n"))
    prompt_block = prompt if prompt.endswith("\n") else prompt + "\n"
    text = prompt_block + closed_body
    prompt_lines = len(prompt_block.rstrip("\n").splitlines())
    total = len(text.splitlines())
    tail_lines = tuple(range(total - n_tail, total)) if n_tail else ()
    # hole lines: every rendered line CARRYING a substitution (the fill is
    # inline, so a hole can share a line with committed tokens; the whole line
    # is still treated as non-evidence, which is the conservative direction)
    hole_lines = tuple(prompt_lines + i
                       for i, ln in enumerate(closed_body.splitlines())
                       if HOLE_SUBSTITUTE in ln)
    return GhostProgram(
        text=text,
        prompt_lines=prompt_lines,
        hole_lines=hole_lines,
        tail_lines=tail_lines,
        fragment_exact=exact,
    )


# --------------------------------------------------------------------------
# R1: rustc type-check
# --------------------------------------------------------------------------


def find_rustc() -> str:
    """Locate a stable rustc: TEMPER_RUSTC, then PATH, then the rustup dirs.

    ``-Zparse-only`` is nightly-only and therefore REJECTED by the design
    (§3.2); this resolver only ever returns a stable toolchain binary.
    """
    env = os.environ.get("TEMPER_RUSTC")
    if env and Path(env).exists():
        return env
    for d in os.environ.get("PATH", "").split(os.pathsep):
        cand = Path(d) / "rustc"
        if cand.exists():
            return str(cand)
    for pat in _RUSTC_CANDIDATES:
        for cand in sorted(glob.glob(os.path.expanduser(pat))):
            if Path(cand).exists():
                return cand
    toolchains = sorted(glob.glob(os.path.expanduser(
        "~/.rustup/toolchains/*/bin/rustc")))
    if toolchains:
        return toolchains[-1]
    raise FileNotFoundError(
        "rustc not found; set TEMPER_RUSTC or put rustc on PATH")


@dataclass(frozen=True)
class RustcError:
    """One rustc error diagnostic reduced to what the verdict filter needs."""

    code: str
    message: str
    primary_line: int
    in_committed: bool


@dataclass(frozen=True)
class R1Verdict:
    """Outcome of one ghost-closure type-check."""

    ok: bool                        # rustc exited 0 with no error diagnostics
    errors: Tuple[RustcError, ...] = ()
    stderr_sha256: str = ""
    wall_s: float = 0.0
    timed_out: bool = False

    @property
    def error_codes(self) -> Tuple[str, ...]:
        return tuple(e.code for e in self.errors)


_ERROR_JSON_PREFIX = '{"$message_type":"diagnostic"'


def _parse_rustc_json(stderr: str) -> List[dict]:
    out = []
    for line in stderr.splitlines():
        line = line.strip()
        if not line.startswith(_ERROR_JSON_PREFIX):
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def r1_typecheck(
    program: str | GhostProgram,
    *,
    hole_proximity: int = 1,
    timeout_s: float = 30.0,
    workdir: Path | str | None = None,
    rustc: str | None = None,
) -> R1Verdict:
    """Type-check one ghost-closed program as a library crate (§3.1).

    ``--emit=metadata`` gives a type-check without codegen; process spawn
    dominates the ~35 ms cost (measured: docs/m3.0-instruments.md). Every
    error's PRIMARY span line is mapped against the ghost program's
    committed/hole line sets, so the informed-class filter can additionally
    require span localization.
    """
    ghost = program if isinstance(program, GhostProgram) else None
    text = ghost.text if ghost is not None else str(program)
    wd = Path(workdir) if workdir is not None else Path(
        os.environ.get("TEMPER_R1_WORKDIR", "/tmp/temper-m3-r1"))
    wd.mkdir(parents=True, exist_ok=True)
    src = wd / "ghost.rs"
    src.write_text(text, encoding="utf-8")
    out_path = wd / "ghost.rmeta"
    bin_path = rustc or find_rustc()
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            [bin_path, "--edition=2021", "--crate-type", "lib",
             "--emit=metadata", "--error-format=json",
             "-o", str(out_path), str(src)],
            capture_output=True, text=True, timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return R1Verdict(ok=False, stderr_sha256="",
                         wall_s=round(time.perf_counter() - t0, 4),
                         timed_out=True)
    wall = time.perf_counter() - t0
    stderr = proc.stderr or ""
    errors: List[RustcError] = []
    for diag in _parse_rustc_json(stderr):
        if diag.get("level") != "error":
            continue
        spans = diag.get("spans") or []
        code = (diag.get("code") or {}).get("code") or ""
        # Drop the "aborting due to N previous errors" summary line: it has no
        # code and no span, so it carries no evidence of its own.
        if not code and not spans:
            continue
        primary_line = 0
        for span in spans:
            if span.get("is_primary"):
                primary_line = int(span.get("line_start") or 0)
                break
        errors.append(RustcError(
            code=code or "unknown",
            message=str(diag.get("message") or ""),
            primary_line=primary_line,
            in_committed=(ghost.is_committed_line(primary_line, hole_proximity)
                          if ghost is not None else True),
        ))
    return R1Verdict(ok=(proc.returncode == 0 and not errors),
                     errors=tuple(errors),
                     stderr_sha256=hashlib.sha256(stderr.encode()).hexdigest(),
                     wall_s=round(wall, 4))


def informed_fatal_evidence(
    verdict: R1Verdict,
    config: VerdictConfig,
) -> List[RustcError]:
    """MID-CANVAS kill evidence (§3.2/§3.3): informed-class errors whose
    primary span lands in committed content. High-precision, low-recall — the
    raw "any error" rule measured 32-42/44 false kills on holed canvases and
    is therefore never used mid-canvas. The policy additionally requires
    persistence across ``config.patience_checkpoints`` checkpoints before it
    acts on this evidence (absence of evidence never kills)."""
    classes = set(config.informed_error_classes)
    return [e for e in verdict.errors
            if e.code in classes and e.in_committed]


def final_checkpoint_clean(verdict: R1Verdict) -> bool:
    """FINAL-checkpoint screen (§3.2/§3.3 item 2): with zero holes R1 is
    EXACT on the measured corpus — 36/36 true fatal detection, 0/44 false —
    so *any* rustc error means the completion is compile-dead.

    Note the asymmetry with :func:`informed_fatal_evidence`: the any-error rule
    is only sound once every hole has been filled by a real committed token.
    """
    return verdict.ok and not verdict.errors


# --------------------------------------------------------------------------
# R0 shape + R2 anti-stub
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class R0Verdict:
    """Free pure-python shape verdict over one rendered canvas."""

    flags: Tuple[str, ...] = ()
    mask_share: float = 0.0
    expected_mask_share: float = 0.0
    fn_header_present: bool = False

    def is_fatal(self, config: VerdictConfig) -> bool:
        fatal = set(config.r0_flags_fatal)
        return any(f in fatal for f in self.flags)


def r0_shape(
    view: SteerProblemView,
    canvas_text: str,
    *,
    mask_share: float,
    expected_mask_share: float,
    checkpoint_fraction: float,
    config: VerdictConfig,
) -> R0Verdict:
    """R0: fence integrity, prose-after-fence, mask-share sanity, fn header,
    final-extraction health.

    ``checkpoint_fraction`` is k/S (how far into the commit schedule we are);
    the expected residual mask share there is ``1 - k/S`` and the tolerance is
    ``config.r0_mask_share_tolerance`` — loose by design: R0 only flags a
    canvas whose committed volume is grossly off schedule (stuck or
    double-committing), never a normal one.
    """
    flags: List[str] = []
    fences = _FENCE_RE.findall(canvas_text)
    if not fences:
        flags.append("fence_unterminated")
    else:
        last_end = max(m.end() for m in _FENCE_RE.finditer(canvas_text))
        if any(ln.strip() for ln in canvas_text[last_end:].splitlines()):
            flags.append("prose_after_fence")
    if abs(mask_share - expected_mask_share) > config.r0_mask_share_tolerance:
        flags.append("mask_share_divergent")
    fn_present = re.search(
        r"\bfn\s+" + re.escape(view.fn_name) + r"\b", canvas_text) is not None
    if checkpoint_fraction >= 0.75 and not fn_present:
        flags.append("fn_header_missing")
    if checkpoint_fraction >= 0.999:
        ex = extract_rust(canvas_text, must_define=[view.fn_name])
        if not ex.ok and "incomplete canvas" not in ex.reason:
            flags.append("extract_rejected_final")
    return R0Verdict(flags=tuple(sorted(set(flags))),
                     mask_share=round(mask_share, 4),
                     expected_mask_share=round(expected_mask_share, 4),
                     fn_header_present=bool(fn_present))


def r2_stub_markers(
    committed_text: str,
    config: VerdictConfig,
) -> Tuple[str, ...]:
    """R2: stub markers present in COMMITTED content (sticky commits => a
    committed stub can never self-repair, so this is a sound kill category)."""
    return tuple(m for m in config.stub_markers if m in committed_text)
