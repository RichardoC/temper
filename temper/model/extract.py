"""Extract Rust code from raw model output. Deterministic, unit-testable.

Order of preference (design §2):
1. ```rust fenced blocks (or unlabeled fences) — largest wins.
2. Any other fenced block — largest wins (model sometimes tags ```rs).
3. Bare-source span: from the first line that looks like a Rust item header
   to the last standalone `}` line.
Then sanity checks: brace balance, must-define symbol check (rejects prompt
echo / chat residue / wrong-problem answers), mask-token rejection (incomplete
canvas). Returns an ExtractResult; never raises.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = ["ExtractResult", "extract_rust"]

_FENCE_RE = re.compile(r"```([a-zA-Z0-9_+#.-]*)[ \t]*\r?\n(.*?)```", re.DOTALL)
_MASK_RE = re.compile(r"<\|?mask\|?>|\|mask\||<masked>")  # decoded mask-token residue
_ITEM_RE = re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:fn|struct|enum|impl|use|type|const|trait)\b")
_DEFINE_RE_TMPL = r"\b(?:fn|struct|enum|impl|trait)\s+{name}\b"


@dataclass
class ExtractResult:
    ok: bool
    code: str = ""
    reason: str = ""


def _fail(reason: str) -> ExtractResult:
    return ExtractResult(ok=False, code="", reason=reason)


def _largest(blocks):
    return max(blocks, key=len) if blocks else None


def _bare_span(text: str):
    """Largest plausible code span without fences: first item-header line to
    last standalone `}` line, provided braces balance."""
    lines = text.splitlines()
    starts = [i for i, ln in enumerate(lines) if _ITEM_RE.match(ln)]
    ends = [i for i, ln in enumerate(lines) if ln.strip() == "}"]
    if not starts or not ends:
        return None
    start = starts[0]
    end_candidates = [e for e in ends if e > start]
    if not end_candidates:
        return None
    # try the largest end first, shrink until braces balance
    for end in reversed(end_candidates):
        span = "\n".join(lines[start : end + 1])
        if span.count("{") == span.count("}") and span.count("{") > 0:
            return span
    return None


def _looks_like_rust(code: str) -> bool:
    if "fn" not in code and "struct" not in code and "impl" not in code:
        return False
    if code.count("{") == 0 or code.count("{") != code.count("}"):
        return False
    if "```" in code:
        return False
    return True


def extract_rust(text: str, must_define=()) -> ExtractResult:
    """Pull the Rust implementation out of model text.

    must_define: iterable of item names (fns/structs) the task signature
    requires; every one must be defined in the extracted code, otherwise the
    candidate is rejected (catches prompt echo and chat residue).
    """
    if not text or not text.strip():
        return _fail("empty model output")
    if _MASK_RE.search(text):
        # mask tokens in the *output region* mean the canvas never completed;
        # fenced extraction below could still succeed if masks stayed outside
        # the block, so don't fail globally here — re-check on the code itself.
        pass

    blocks = _FENCE_RE.findall(text)
    code = None
    if blocks:
        rustish = [body for lang, body in blocks if lang.lower() in ("", "rust", "rs")]
        code = _largest(rustish) or _largest([body for _, body in blocks])
    if code is None:
        code = _bare_span(text)
        if code is None:
            return _fail("no fenced rust block and no plausible bare code span")

    code = code.strip("\n").strip()
    if not code:
        return _fail("empty code span")
    if _MASK_RE.search(code):
        return _fail("incomplete canvas: mask tokens present in code")
    if not _looks_like_rust(code):
        return _fail("extracted span does not look like Rust (fn/struct + balanced braces)")
    for name in must_define:
        if not re.search(_DEFINE_RE_TMPL.format(name=re.escape(name)), code):
            return _fail(f"required symbol {name!r} not defined (prompt echo or wrong problem?)")
    return ExtractResult(ok=True, code=code, reason="ok")
