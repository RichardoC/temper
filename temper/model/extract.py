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

M2.3 phase-0 addition (fenced→bare fallback): the fenced path is poisoned
when the model re-renders doc comments containing triple backticks (e.g.
HumanEval_15_string_sequence in the M2.2 B2 pilot: `/// ```` example lines
terminate the outer ```rust fence early, so the "largest fenced block" is
prose and the real `fn` after it is never seen). When the fenced candidate
fails the Rust-shape check, the bare-span path is retried before giving up.
The bare span keeps its own gates (item header + balanced braces), so the
fallback can only turn a guaranteed rejection into a checked candidate —
never degrade a passing extraction. When both paths fail, the rejection
reason stays the historical "no fenced rust block and no plausible bare
code span" so exclusion-class histograms remain comparable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

__all__ = ["ExtractResult", "RepairResult", "extract_rust", "repair_code"]

_FENCE_RE = re.compile(r"```([a-zA-Z0-9_+#.-]*)[ \t]*\r?\n(.*?)```", re.DOTALL)
_MASK_RE = re.compile(r"<\|?mask\|?>|\|mask\||<masked>")  # decoded mask-token residue
_ITEM_RE = re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:fn|struct|enum|impl|use|type|const|trait)\b")
_DEFINE_RE_TMPL = r"\b(?:fn|struct|enum|impl|trait)\s+{name}\b"
_IMPL_HEADER_RE = re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?impl\b[^{]*\{")
# M2.1 repair pass: a DEGENERATE item header looks like `[pub] Ident {` where
# Ident is NOT an item keyword — e.g. run-B p0's stray `pub RingBuf { fn new(..) }`
# (an `impl` block that lost its keyword).
_DEGENERATE_HEADER_RE = re.compile(
    r"^\s*(?:pub(?:\([^)]*\))?\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*\{")
_ITEM_KEYWORDS = {"fn", "struct", "enum", "impl", "trait", "type", "use",
                  "const", "static", "mod", "unsafe", "extern", "async"}
_TYPE_DEF_RE_TMPL = r"\b(?:struct|enum)\s+{name}\b"
_PROSE_LINE_RE = re.compile(
    r"^\s*(?:#{1,6}\s|\*\s|\d+[.)]\s|-\s|\*\*)")  # markdown-ish prose


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


@dataclass
class RepairResult:
    """Outcome of the L0.5 repair pass (M2.1 item 1).

    repaired: at least one strategy changed the code (a repair was applied).
    valid:    the final code passes the extract-level checks (balance +
              must_define); compilation is the CALLER's gate (l0 re-runs the
              harness) — the pass itself never invokes rustc.
    attempts: JSON-able per-strategy attempt records, in order.
    """

    repaired: bool
    valid: bool
    code: str
    attempts: list = field(default_factory=list)


def _check_code(code: str, must_define=()) -> str:
    """Extract-level validity gate shared by extract_rust and the repair
    pass. Returns '' when the code passes, else the rejection reason."""
    if not code or not code.strip():
        return "empty code span"
    if _MASK_RE.search(code):
        return "incomplete canvas: mask tokens present in code"
    if not _looks_like_rust(code):
        return "extracted span does not look like Rust (fn/struct + balanced braces)"
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
        return (
            "required symbol(s) " + ", ".join(repr(n) for n in impl_only)
            + " defined only as inherent impl method(s) inside an `impl` block, "
            "not as module-level free functions (API-shape mismatch: the task "
            "signature requires free functions; impl-method answers are rejected)"
        )
    if missing:
        return (
            "required symbol(s) " + ", ".join(repr(n) for n in missing)
            + " not defined anywhere in extracted code (prompt echo or wrong problem?)"
        )
    return ""


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

    On a must-define rejection the candidate span is still returned in
    ``code`` (ok stays False) so the L0.5 repair pass has material to work
    on; mask-token and non-Rust spans keep code="" (not repairable here).
    """
    if not text or not text.strip():
        return _fail("empty model output")

    blocks = _FENCE_RE.findall(text)
    code = None
    if blocks:
        rustish = [body for lang, body in blocks if lang.lower() in ("", "rust", "rs")]
        code = _largest(rustish) or _largest([body for _, body in blocks])
        if code is not None and not _looks_like_rust(code.strip("\n").strip()) \
                and not _MASK_RE.search(code):
            # M2.3 phase-0 fallback (module docstring): the fenced path can
            # be poisoned (prose block via embedded ``` in doc-comment echo,
            # or an unbalanced span) while a complete, brace-balanced fn is
            # still present later in the text — retry the bare-span path,
            # which keeps its own gates (item header + balanced braces),
            # before giving up. Mask-token residue is exempt: the mask check
            # in _check_code is the sharper diagnosis (incomplete canvas),
            # and a bare span is meaningless there. Never degrades a
            # passing extraction.
            code = None
    if code is None:
        code = _bare_span(text)
        if code is None:
            return _fail("no fenced rust block and no plausible bare code span")

    code = code.strip("\n").strip()
    if not code:
        return _fail("empty code span")
    reason = _check_code(code, must_define)
    if not reason:
        return ExtractResult(ok=True, code=code, reason="ok")
    # keep the span on must-define failures (repairable); drop it for the
    # non-repairable classes (mask residue / not-Rust)
    repairable = "not defined" in reason or "impl method" in reason
    return ExtractResult(ok=False, code=(code if repairable else ""), reason=reason)


# --------------------------------------------------------------------------
# M2.1 item 1: L0.5 repair pass (deterministic, conservative, never re-prompts
# the model). Given extracted code that fails compile/must_define, try in
# order, each attempt recorded:
#   (a) degenerate-item repair (brace scan):
#       - drop: any degenerate item (`pub Ident { ... }` with a non-keyword
#         Ident) whose identifier appears NOWHERE else in the code, when the
#         removal keeps braces balanced;
#       - convert: a degenerate item whose Ident IS an existing module-scope
#         struct/enum and whose body holds only fn items gets its header
#         rewritten to `impl Ident {` (the model's evident intent — run-B p0's
#         stray `pub RingBuf { fn new(..) }` is referenced by the rest of the
#         code, so dropping it would leave `RingBuf::new` dangling and could
#         never reach compile-ok; conversion is the minimal fix).
#   (b) strip trailing prose after the final code block (markdown-ish,
#       brace-free, non-item lines after the last balanced item).
# --------------------------------------------------------------------------


def _top_level_items(code: str):
    """Line-based brace scan of module-scope items.

    Returns a list of dicts {start, end, header} (line indices, inclusive
    end) for every item that starts at depth 0, INCLUDING degenerate ones
    (`pub Ident {` with a non-keyword Ident). Attributes/doc comments on the
    lines immediately above an item belong to its span. String literals with
    braces are handled by the same naive heuristic as _split_impl_regions.
    """
    lines = code.splitlines()
    items = []
    depth = 0
    i = 0
    while i < len(lines):
        ln = lines[i]
        stripped = ln.strip()
        if depth == 0 and stripped and not stripped.startswith("//") \
                and not stripped.startswith("#[") \
                and (_ITEM_RE.match(ln) or _DEGENERATE_HEADER_RE.match(ln)):
            start = i
            # pull in immediately-preceding attribute/doc lines
            while start > 0 and lines[start - 1].strip().startswith(("#[", "///", "//!")):
                start -= 1
            j = i
            opened = ln.count("{") - ln.count("}")
            while j < len(lines) and opened > 0:
                j += 1
                if j < len(lines):
                    opened += lines[j].count("{") - lines[j].count("}")
            end = j if opened <= 0 else len(lines) - 1
            items.append({"start": start, "end": end, "header": lines[i]})
            i = end + 1
            continue
        depth += ln.count("{") - ln.count("}")
        if depth < 0:
            depth = 0
        i += 1
    return items


def _is_degenerate(header: str):
    """Return the item identifier if `header` is a degenerate pseudo-impl
    (`[pub] Ident {` with Ident not an item keyword), else None."""
    m = _DEGENERATE_HEADER_RE.match(header)
    if not m:
        return None
    ident = m.group(1)
    if ident in _ITEM_KEYWORDS:
        return None
    return ident


def _body_only_fns(lines, start, end):
    """True if every top-level item strictly inside lines[start:end] is a fn."""
    inner = "\n".join(lines[start + 1 : end])
    for it in _top_level_items(inner):
        if not re.match(r"^\s*(?:pub(?:\([^)]*\))?\s+)?fn\b", it["header"]):
            return False
    return True


def _single_line_convertible(header: str, ident: str) -> bool:
    """A ONE-line degenerate item (`[pub] Ident { ... }`) is convertible to
    `impl Ident {` only if its brace body is exactly one fn item (no `;` at
    body depth 0, nothing after the first fn closes)."""
    m = re.match(r"^\s*(?:pub(?:\([^)]*\))?\s+)?" + re.escape(ident)
                 + r"\s*\{(.*)\}\s*$", header)
    if not m:
        return False
    inner = m.group(1)
    if not inner.lstrip().startswith("fn"):
        return False
    depth = 0
    for i, ch in enumerate(inner):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and inner[i + 1:].strip():
                return False  # content after the (only) fn item
        elif ch == ";" and depth == 0:
            return False      # separates multiple items
    return depth == 0


def _repair_degenerate_items(code: str, must_define=()):
    """Strategy (a): per degenerate item, convert when its type is defined
    (keeps references resolvable), else drop when unreferenced."""
    lines = code.splitlines()
    actions = []
    modified = False
    # iterate over a materialized list; spans are computed on the ORIGINAL
    # layout and applied bottom-up so indices stay valid
    items = _top_level_items(code)
    for it in reversed(items):
        ident = _is_degenerate(it["header"])
        if ident is None:
            continue
        outside = "\n".join(lines[: it["start"]] + lines[it["end"] + 1 :])
        type_defined = re.search(
            _TYPE_DEF_RE_TMPL.format(name=re.escape(ident)), outside)
        referenced = re.search(rf"\b{re.escape(ident)}\b", outside) is not None
        convertible = (
            _body_only_fns(lines, it["start"], it["end"])
            if it["end"] > it["start"]
            else _single_line_convertible(it["header"], ident)
        )
        if type_defined and convertible:
            lines[it["start"]] = re.sub(
                r"^\s*pub\s+", "impl ", it["header"])
            if not lines[it["start"]].lstrip().startswith("impl"):
                # header had no `pub` — plain `Ident {`
                lines[it["start"]] = re.sub(
                    rf"\b{re.escape(ident)}\b", f"impl {ident}", it["header"], count=1)
            actions.append({"action": "convert-pseudo-impl", "ident": ident,
                            "line": it["start"] + 1})
            modified = True
        elif not referenced and it["end"] >= it["start"] \
                and "\n".join(lines[it["start"]: it["end"] + 1]).count("{") == \
                "\n".join(lines[it["start"]: it["end"] + 1]).count("}"):
            del lines[it["start"]: it["end"] + 1]
            actions.append({"action": "drop-degenerate-item", "ident": ident,
                            "line": it["start"] + 1})
            modified = True
        else:
            actions.append({"action": "skip-degenerate-item", "ident": ident,
                            "line": it["start"] + 1,
                            "why": ("type not defined locally and identifier "
                                    "referenced elsewhere — not conservatively repairable")})
    new_code = "\n".join(lines) if modified else code
    return new_code, actions


def _repair_strip_trailing_prose(code: str):
    """Strategy (b): strip markdown-ish, brace-free, non-item lines after the
    last balanced top-level item."""
    lines = code.splitlines()
    items = _top_level_items(code)
    if not items:
        return code, []
    last = items[-1]["end"]
    tail = lines[last + 1 :]
    prose = [ln for ln in tail
             if ln.strip() and "{" not in ln and "}" not in ln
             and not _ITEM_RE.match(ln) and not ln.strip().startswith(("#[", "//", "///"))
             and _PROSE_LINE_RE.match(ln)]
    if not prose:
        return code, []
    prose_set = set(prose)
    kept = lines[: last + 1] + [ln for ln in tail if ln not in prose_set]
    new_code = "\n".join(kept)
    if new_code == code:
        return code, []
    return new_code, [{"action": "strip-trailing-prose",
                       "lines": len(prose)}]


def repair_code(code: str, must_define=()) -> RepairResult:
    """L0.5 repair pass. Deterministic + conservative: only the strategies
    above, each attempted in order (a) then (b); every attempt is recorded.
    Never calls the model.

    Called by the pipeline ONLY on code that failed compile or must_define
    (extract-level validity alone is not the trigger: prose-tail code passes
    the extract checks yet fails rustc). The pass never invokes rustc itself;
    ``valid`` reports extract-level status so the caller can decide whether
    to re-run the harness / accept the code.
    """
    attempts = []
    cur = code

    # (a) degenerate-item repair (drop / convert)
    c2, actions = _repair_degenerate_items(cur, must_define)
    attempts.append({
        "strategy": "degenerate-items",
        "changed": any(a["action"] in ("drop-degenerate-item",
                                       "convert-pseudo-impl")
                       for a in actions),
        "actions": actions,
    })
    if c2 != cur:
        cur = c2

    # (b) strip trailing prose
    c3, actions = _repair_strip_trailing_prose(cur)
    attempts.append({
        "strategy": "strip-trailing-prose", "changed": bool(actions),
        "actions": actions,
    })
    if c3 != cur:
        cur = c3

    changed = any(a["changed"] for a in attempts)
    return RepairResult(changed, not _check_code(cur, must_define), cur, attempts)
