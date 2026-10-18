"""LENIENT parser for cargo/rustc JSON diagnostics.

Rustc reserves the right to add fields and enum variants at any time, so this
parser ignores everything it does not recognize and never raises on unknown
shapes — a diagnostic that parses partially is worth more than a crashed
harness. Non-JSON lines (progress bars, panic backtraces) are skipped.

Surfaces ICEs ("error: internal compiler error") distinctly via
``Diagnostic.is_ice``: an ICE is a *toolchain* failure, not feedback for the
candidate, and must be routed differently by callers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

__all__ = ["Span", "Suggestion", "Diagnostic", "parse_cargo_messages"]

_ICE_MARKER = "internal compiler error"


@dataclass(frozen=True)
class Span:
    """A byte-offset span into a source file (primary spans flag the site)."""

    file_name: str
    byte_start: int
    byte_end: int
    line_start: int = 0
    line_end: int = 0
    is_primary: bool = False
    label: str = ""


@dataclass(frozen=True)
class Suggestion:
    """A machine-applicable replacement proposed by rustc."""

    suggested_replacement: str
    applicability: str = "Unspecified"  # MachineApplicable | MaybeIncorrect | ...
    file_name: str = ""
    byte_start: int = 0
    byte_end: int = 0


@dataclass(frozen=True)
class Diagnostic:
    """One rustc/cargo diagnostic, reduced to the fields temper consumes."""

    level: str = "error"                     # error | warning | note | help | ...
    code: str | None = None                  # e.g. "E0308" (code.code in raw JSON)
    message: str = ""
    rendered: str = ""                       # human-readable ANSI rendering
    spans: list[Span] = field(default_factory=list)
    suggestions: list[Suggestion] = field(default_factory=list)
    children: list[str] = field(default_factory=list)  # sub-diagnostic messages

    @property
    def is_error(self) -> bool:
        return self.level.startswith("error") or self.is_ice

    @property
    def is_ice(self) -> bool:
        """True for 'error: internal compiler error' — a toolchain bug."""
        return _ICE_MARKER in self.level.lower() or _ICE_MARKER in self.message.lower()


def _parse_span(raw: dict) -> Span | None:
    """Lenient span extraction: any missing field just defaults."""
    try:
        return Span(
            file_name=str(raw.get("file_name", "")),
            byte_start=int(raw.get("byte_start", 0)),
            byte_end=int(raw.get("byte_end", 0)),
            line_start=int(raw.get("line_start", 0)),
            line_end=int(raw.get("line_end", 0)),
            is_primary=bool(raw.get("is_primary", False)),
            label=str(raw.get("label") or ""),
        )
    except (TypeError, ValueError):
        return None  # malformed span: skip it, keep the rest


def _parse_suggestions(spans_raw: list) -> list[Suggestion]:
    out: list[Suggestion] = []
    for raw in spans_raw:
        if not isinstance(raw, dict):
            continue
        repl = raw.get("suggested_replacement")
        if repl is None:
            continue
        out.append(Suggestion(
            suggested_replacement=str(repl),
            applicability=str(raw.get("suggestion_applicability") or "Unspecified"),
            file_name=str(raw.get("file_name", "")),
            byte_start=int(raw.get("byte_start", 0) or 0),
            byte_end=int(raw.get("byte_end", 0) or 0),
        ))
    return out


def _parse_message(msg: dict) -> Diagnostic:
    """Reduce one raw cargo ``message`` object; tolerate any deviation."""
    code_raw = msg.get("code")
    code = None
    if isinstance(code_raw, dict):  # {"code": "E0308", "explanation": ...}
        code = code_raw.get("code")
    elif isinstance(code_raw, str):
        code = code_raw

    spans_raw = msg.get("spans") or []
    spans: list[Span] = []
    if isinstance(spans_raw, list):
        for raw in spans_raw:
            if isinstance(raw, dict):
                s = _parse_span(raw)
                if s is not None:
                    spans.append(s)

    children: list[str] = []
    for child in msg.get("children") or []:
        if isinstance(child, dict):
            children.append(str(child.get("message", "")))

    level = str(msg.get("level", "error"))
    rendered = str(msg.get("rendered", ""))
    message = str(msg.get("message", ""))
    if _ICE_MARKER in level.lower() and _ICE_MARKER not in message.lower():
        message = f"{level}: {message}"
        level = "error"

    return Diagnostic(
        level=level,
        code=code,
        message=message,
        rendered=rendered,
        spans=spans,
        suggestions=_parse_suggestions(spans_raw if isinstance(spans_raw, list) else []),
        children=children,
    )


def parse_cargo_messages(text: str) -> list[Diagnostic]:
    """Parse cargo ``--message-format=json`` output into diagnostics.

    Accepts the full cargo JSON stream: lines whose top-level object carries
    ``reason == "compiler-message"`` are reduced; every other line (including
    unparseable ones) is ignored. Unknown fields/variants are silently dropped.
    """
    out: list[Diagnostic] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue  # progress line, stray output — lenient by design
        if not isinstance(obj, dict):
            continue
        if obj.get("reason") != "compiler-message":
            continue
        msg = obj.get("message")
        if isinstance(msg, dict):
            out.append(_parse_message(msg))
        elif isinstance(msg, str):
            # Short-form diagnostic (json=short): capture the text only.
            out.append(Diagnostic(level="error", message=msg))
    return out
