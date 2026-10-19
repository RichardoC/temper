"""M2.1 item 2: failure taxonomy for non-scoring candidate rows.

Pure functions — no I/O beyond what the caller passes in, no LLM. Classes
(docs/m2-design.md M2.1 item 2):

    impl_method        required symbols exist ONLY as inherent impl methods
    renamed_or_self    required symbols absent; raw text shows renamed
                       cousins (e.g. `new` for `ring_new`) and/or
                       self-parameterized "free functions"
    truncated_canvas   canvas exhausted: unterminated fence, unbalanced
                       braces, or mask tokens in the raw output
    prose_only         no code span at all; prose output
    echo               prompt/signature echoed back without new code
    compile_error      extracted + accepted by extract, rustc rejected
    test_fail          compiled but the property/verifier rejected it

Evidence is a short list of strings explaining the decision — every
classification must be auditable against the raw sidecar (M1.5 lesson).
"""

from __future__ import annotations

import re
from typing import List, Optional

__all__ = ["CLASSES", "classify_row", "histogram"]

CLASSES = ("impl_method", "renamed_or_self", "truncated_canvas", "prose_only",
           "echo", "compile_error", "test_fail")

_FENCE_MARKER_RE = re.compile(r"^\s*```", re.M)
_SELF_PARAM_RE = re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?fn\s+\w+\s*\([^)]*&\s*(?:mut\s+)?self", re.M)
_MASK_RE = re.compile(r"<\|?mask\|?>|\|mask\||<masked>")


def _rename_cousins(required: List[str], raw: str) -> List[str]:
    """Required names whose last `_`-segment appears as a bare `fn <seg>(`
    in the raw text (e.g. ring_new -> `fn new(`)."""
    found = []
    for name in required or []:
        seg = name.split("_")[-1]
        if seg and seg != name and re.search(rf"\bfn\s+{re.escape(seg)}\s*\(", raw):
            found.append(seg)
    return found


def _code_spans(raw: str):
    """(fenced bodies, bare span or None, fence_marker_count) for raw text."""
    fences = re.findall(r"```[a-zA-Z0-9_+#.-]*[ \t]*\r?\n(.*?)```", raw, re.DOTALL)
    markers = len(_FENCE_MARKER_RE.findall(raw))
    # bare span heuristic mirrors extract._bare_span (balanced braces only)
    bare = None
    lines = raw.splitlines()
    starts = [i for i, ln in enumerate(lines)
              if re.match(r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:fn|struct|enum|impl|use|type|const|trait)\b", ln)]
    ends = [i for i, ln in enumerate(lines) if ln.strip() == "}"]
    if starts and ends:
        start = starts[0]
        for end in reversed([e for e in ends if e > start]):
            span = "\n".join(lines[start:end + 1])
            if span.count("{") == span.count("}") and span.count("{") > 0:
                bare = span
                break
    return fences, bare, markers


def _prompt_echo_fraction(raw: str, prompt: Optional[str]) -> float:
    if not prompt or not prompt.strip():
        return 0.0
    plines = [ln.strip() for ln in prompt.splitlines()
              if len(ln.strip()) >= 15]
    if not plines:
        return 0.0
    hit = sum(1 for ln in plines if ln in raw)
    return hit / len(plines)


def classify_row(row: dict, raw_text: Optional[str] = None,
                 prompt: Optional[str] = None) -> dict:
    """Classify one candidate row. Returns {"class": str|None, "evidence": []}.

    class None = not a model-output failure (test pass, or infrastructure
    error) — the caller's histogram skips it under `unclassified`.
    """
    ev: List[str] = []
    test_status = row.get("test_status")
    compile_status = row.get("compile_status")
    reason = row.get("extract_reason") or ""
    reason_l = reason.lower()

    # --- successes are not failures ---------------------------------------
    if test_status == "pass":
        return {"class": None, "evidence": ["tests pass"]}

    # --- generation errors are infrastructure, not model-output classes ----
    if row.get("gen_error"):
        return {"class": None,
                "evidence": [f"generation error (infrastructure): {row['gen_error'][:120]}"]}

    # --- post-extract failure classes ---------------------------------------
    if compile_status == "compile_error":
        if row.get("diag_codes"):
            ev.append("diag codes: " + ",".join(row["diag_codes"]))
        if row.get("repaired"):
            ev.append("L0.5 repair attempted but did not fix compilation")
        return {"class": "compile_error", "evidence": ev}

    if compile_status == "timeout":
        ev.append("harness timeout during run")
        return {"class": "test_fail", "evidence": ev}

    if test_status in ("fail", "abort", "error", "timeout"):
        if row.get("reason"):
            ev.append(f"verifier reason: {str(row['reason'])[:160]}")
        return {"class": "test_fail", "evidence": ev}

    # --- extraction rejections ----------------------------------------------
    if "defined only as inherent impl method" in reason_l:
        ev.append("required symbols exist only as inherent impl methods")
        return {"class": "impl_method", "evidence": ev}

    raw = raw_text or ""

    if "not defined anywhere" in reason_l:
        if raw:
            if _SELF_PARAM_RE.search(raw):
                ev.append("module-level fn with &self/&mut self parameter in raw output")
                return {"class": "renamed_or_self", "evidence": ev}
            cousins = _rename_cousins(row.get("missing_symbols", []), raw)
            # fall back to parsing the missing names out of the reason string
            if not cousins:
                m = re.search(r"required symbol\(s\) (.+?) not defined", reason)
                if m:
                    names = re.findall(r"'([^']+)'", m.group(1))
                    cousins = _rename_cousins(names, raw)
            if cousins:
                ev.append(f"renamed cousins present in raw: fn {', fn '.join(cousins)}(")
                return {"class": "renamed_or_self", "evidence": ev}
        if not raw:
            ev.append("no raw sidecar; reason wording defaults to echo")
            return {"class": "echo", "evidence": ev}

    if raw:
        fences, bare, markers = _code_spans(raw)
        truncated = False
        if _MASK_RE.search(raw):
            truncated = True
            ev.append("mask tokens present in raw output")
        if markers % 2 == 1:
            truncated = True
            ev.append("unterminated code fence in raw output")
        span = (fences[-1] if fences else bare) or ""
        if span and span.count("{") != span.count("}"):
            truncated = True
            ev.append("largest code span has unbalanced braces")
        if not span and raw.count("{") != raw.count("}") and \
                re.search(r"\b(?:fn|struct|impl)\b", raw):
            truncated = True
            ev.append("raw output starts code but braces never balance")
        # echo: raw mostly repeats the prompt and adds no new code
        echo_frac = _prompt_echo_fraction(raw, prompt)
        if echo_frac >= 0.6 and not (fences or bare):
            ev.append(f"{echo_frac:.0%} of prompt lines repeated verbatim, no code span")
            return {"class": "echo", "evidence": ev}
        if truncated:
            return {"class": "truncated_canvas", "evidence": ev}
        if not (fences or bare):
            if "no fenced rust block" in reason_l or "no plausible bare code span" in reason_l:
                ev.append("no code span in raw output; prose present"
                          if raw.strip() else "raw output empty")
                return {"class": "prose_only", "evidence": ev}
        # zero-brace span = signature-only echo (or prose snippet)
        if span and span.count("{") == 0:
            m = re.search(r"required symbol\(s\) (.+?) not defined", reason)
            names = re.findall(r"'([^']+)'", m.group(1)) if m else []
            if names and all(re.search(rf"\bfn\s+{re.escape(n)}\b", span) for n in names):
                ev.append("signature-only echo: required names appear as bare fn headers")
                return {"class": "echo", "evidence": ev}
            ev.append("no braces in code span; not identifiable as code")
            return {"class": "prose_only", "evidence": ev}
        if "does not look like rust" in reason_l:
            ev.append("span rejected by extract without a sharper signature")
            return {"class": "prose_only", "evidence": ev}

    if "no fenced rust block" in reason_l or "no plausible bare code span" in reason_l:
        ev.append("no raw sidecar; extract found no code span")
        return {"class": "prose_only", "evidence": ev}
    if "mask tokens" in reason_l:
        ev.append("mask tokens in extracted code (canvas never completed)")
        return {"class": "truncated_canvas", "evidence": ev}
    if "does not look like rust" in reason_l:
        ev.append("extracted span not Rust-shaped; no raw sidecar to refine")
        return {"class": "prose_only", "evidence": ev}

    ev.append(f"unclassified: extract_reason={reason[:140]!r}")
    return {"class": None, "evidence": ev}


def histogram(classifications: List[dict]) -> dict:
    """Counts per class; 'unclassified' buckets rows whose class is None."""
    h = {c: 0 for c in CLASSES}
    h["unclassified"] = 0
    for c in classifications:
        cls = c.get("class")
        h[cls if cls in CLASSES else "unclassified"] += 1
    return {k: v for k, v in h.items() if v}
