"""Extract Rust code from raw model output. Deterministic, unit-testable.

Order of preference (design §2):
1. ```rust fenced blocks (or unlabeled fences) — largest wins.
2. Any other fenced block — largest wins (model sometimes tags ```rs).
3. Bare-source span: from the first line that looks like a Rust item header
   to the last standalone `}` line.
Then sanity checks: brace balance, must-define symbol check (rejects prompt
echo / chat residue / wrong-problem answers; a symbol defined only as an
inherent `impl` method is a distinct, greppable rejection class — the task
signatures require module-level FREE functions), mask-token rejection
(incomplete canvas). Returns an ExtractResult; never raises.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = ["ExtractResult", "extract_rust"]

_FENCE_RE = re.compile(r"```([a-zA-Z0-9_+#.-]*)[ \t]*\r?\n(.*?)```", re.DOTALL)
_MASK_RE = re.compile(r"<\|?mask\|?>|\|mask\||<masked>")  # decoded mask-token residue
_ITEM_RE = re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:fn|struct|enum|impl|use|type|const|trait)\b")
_DEFINE_RE_TMPL = r"\b(?:fn|struct|enum|impl|trait)\s+{name}\b"
_IMPL_HEADER_RE = re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?impl\b[^{]*\{")


def _split_impl_regions(code: str):
    """Split extracted code into (module_scope_text, impl_block_text).

    Line-based brace-depth scan: everything inside an `impl ... { ... }`
    block (header + body, trait impls included) lands in the second part,
    everything else in the first. Heuristic (string-literal naive) but
    sufficient for the module-scope API-shape check.
    """
    module_lines: list = []
    impl_lines: list = []
    depth = 0
    in_impl = False
    impl_open_depth = 0
    for ln in code.splitlines():
        opens, closes = ln.count("{"), ln.count("}")
        if not in_impl and _IMPL_HEADER_RE.match(ln):
            in_impl = True
            impl_open_depth = depth
            impl_lines.append(ln)
            depth += opens - closes
            if depth <= impl_open_depth:  # degenerate one-line `impl X {}`
                in_impl = False
            continue
        (impl_lines if in_impl else module_lines).append(ln)
        depth += opens - closes
        if in_impl and depth <= impl_open_depth:
            in_impl = False
    return "\n".join(module_lines), "\n".join(impl_lines)


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
    requires; every one must be defined at MODULE scope in the extracted
    code, otherwise the candidate is rejected. Rejection reasons distinguish
    two failure classes for run-record analysis:
      - "defined only as inherent impl method(s) ..." -> wrong API shape
        (impl-block answer where free functions are required),
      - "not defined anywhere ..." -> prompt echo / wrong problem.
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
    module_code, impl_code = _split_impl_regions(code)
    impl_only, missing = [], []
    for name in must_define:
        pat = _DEFINE_RE_TMPL.format(name=re.escape(name))
        if re.search(pat, module_code):
            continue
        if re.search(pat, impl_code):
            impl_only.append(name)
        else:
            missing.append(name)
    if impl_only:
        return _fail(
            "required symbol(s) " + ", ".join(repr(n) for n in impl_only)
            + " defined only as inherent impl method(s) inside an `impl` block, "
            "not as module-level free functions (API-shape mismatch: the task "
            "signature requires free functions; impl-method answers are rejected)"
        )
    if missing:
        return _fail(
            "required symbol(s) " + ", ".join(repr(n) for n in missing)
            + " not defined anywhere in extracted code (prompt echo or wrong problem?)"
        )
    return ExtractResult(ok=True, code=code, reason="ok")
