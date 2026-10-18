"""temper.feedback — structured diagnostics from tool output."""

from temper.feedback.diagnostics import (
    Diagnostic,
    Span,
    Suggestion,
    parse_cargo_messages,
)

__all__ = ["Diagnostic", "Span", "Suggestion", "parse_cargo_messages"]
