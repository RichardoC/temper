"""Unit tests for the M2.1 failure taxonomy (temper.model.classify).

Includes the run-B fixture hand-count check (docs/m1.5-report.md):
p0=compile_error, p1=renamed_or_self, p2=truncated_canvas, p3=truncated_canvas.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from temper.model.classify import classify_row, histogram

REPO = Path(__file__).resolve().parents[1]
RUN_B = REPO / "bench" / "ringbuf" / "runs" / "20261018T224453Z.raw"


def _raw(name):
    return (RUN_B / name).read_text()


class TestClasses(unittest.TestCase):
    def test_impl_method(self):
        row = {"extract_ok": False, "compile_status": "not_attempted",
               "test_status": "not_attempted",
               "extract_reason": "required symbol(s) 'ring_new' defined only as "
                                 "inherent impl method(s) inside an `impl` block, "
                                 "not as module-level free functions (API-shape "
                                 "mismatch: the task signature requires free "
                                 "functions; impl-method answers are rejected)"}
        c = classify_row(row)
        self.assertEqual(c["class"], "impl_method")

    def test_renamed_or_self(self):
        row = {"extract_ok": False, "test_status": "not_attempted",
               "extract_reason": "required symbol(s) 'ring_new' not defined "
                                 "anywhere in extracted code (prompt echo or "
                                 "wrong problem?)"}
        raw = "pub fn push(&mut self, v: i64) {\n    self.tail += 1;\n}"
        c = classify_row(row, raw_text=raw)
        self.assertEqual(c["class"], "renamed_or_self")
        self.assertIn("&self/&mut self", c["evidence"][0])

    def test_renamed_cousin_without_self(self):
        row = {"extract_ok": False, "test_status": "not_attempted",
               "extract_reason": "required symbol(s) 'ring_new' not defined "
                                 "anywhere in extracted code (prompt echo or "
                                 "wrong problem?)"}
        raw = "pub fn new(cap: usize) -> Ring { Ring { cap } }"
        c = classify_row(row, raw_text=raw)
        self.assertEqual(c["class"], "renamed_or_self")
        self.assertIn("fn new(", c["evidence"][0])

    def test_truncated_unterminated_fence(self):
        row = {"extract_ok": False, "test_status": "not_attempted",
               "extract_reason": "required symbol(s) 'ring_pop' not defined "
                                 "anywhere in extracted code (prompt echo or "
                                 "wrong problem?)"}
        raw = "```rust\npub fn ring_new() {\n    x\n"  # fence never closes
        c = classify_row(row, raw_text=raw)
        self.assertEqual(c["class"], "truncated_canvas")

    def test_truncated_unbalanced_bare_span(self):
        row = {"extract_ok": False, "test_status": "not_attempted",
               "extract_reason": "required symbol(s) 'f' not defined anywhere "
                                 "in extracted code (prompt echo or wrong problem?)"}
        raw = "pub fn f(v: i64) {\n    let x = v;"
        c = classify_row(row, raw_text=raw)
        self.assertEqual(c["class"], "truncated_canvas")

    def test_truncated_mask_tokens(self):
        row = {"extract_ok": False, "test_status": "not_attempted",
               "extract_reason": "incomplete canvas: mask tokens present in code"}
        c = classify_row(row, raw_text="```rust\nfn f() { |mask| }\n```")
        self.assertEqual(c["class"], "truncated_canvas")

    def test_prose_only(self):
        row = {"extract_ok": False, "test_status": "not_attempted",
               "extract_reason": "no fenced rust block and no plausible bare code span"}
        raw = ("To implement the ring buffer we need to define a structure that\n"
               "holds the buffer data and the necessary functions:\n"
               "1. The RingBuf struct will use a vector store the head tail pointers\n")
        c = classify_row(row, raw_text=raw)
        self.assertEqual(c["class"], "prose_only")

    def test_prose_only_without_raw(self):
        row = {"extract_ok": False, "test_status": "not_attempted",
               "extract_reason": "no fenced rust block and no plausible bare code span"}
        c = classify_row(row)
        self.assertEqual(c["class"], "prose_only")
        self.assertIn("no raw sidecar", c["evidence"][0])

    def test_echo_prompt_repetition(self):
        row = {"extract_ok": False, "test_status": "not_attempted",
               "extract_reason": "no fenced rust block and no plausible bare code span"}
        prompt = ("You are writing a Rust module for a crate that will be "
                  "property-tested. Implement exactly this public API:\n\n"
                  "pub fn sort_i32(v: &mut Vec<i32>) is the signature to define\n")
        c = classify_row(row, raw_text=prompt, prompt=prompt)
        self.assertEqual(c["class"], "echo")

    def test_echo_signature_only(self):
        # signature-only fence: balanced?? no braces at all -> extract fails
        # with "does not look like Rust"; with prompt absent, the required
        # names as bare fn headers identify the echo
        row = {"extract_ok": False, "test_status": "not_attempted",
               "extract_reason": "required symbol(s) 'sort_i32' not defined "
                                 "anywhere in extracted code (prompt echo or "
                                 "wrong problem?)"}
        raw = "```rust\npub fn sort_i32(v: &mut Vec<i32>)\n```"
        c = classify_row(row, raw_text=raw)
        self.assertEqual(c["class"], "echo")

    def test_compile_error(self):
        row = {"extract_ok": True, "compile_status": "compile_error",
               "test_status": "not_attempted", "diag_codes": [],
               "extract_reason": "ok"}
        c = classify_row(row)
        self.assertEqual(c["class"], "compile_error")

    def test_compile_error_after_failed_repair_notes_it(self):
        row = {"extract_ok": True, "compile_status": "compile_error",
               "test_status": "not_attempted", "diag_codes": ["E0425"],
               "extract_reason": "ok", "repaired": True}
        c = classify_row(row)
        self.assertEqual(c["class"], "compile_error")
        self.assertTrue(any("repair" in e for e in c["evidence"]))

    def test_test_fail(self):
        row = {"extract_ok": True, "compile_status": "ok",
               "test_status": "fail", "reason": "sort_i32(v) != sorted(v)"}
        c = classify_row(row)
        self.assertEqual(c["class"], "test_fail")

    def test_pass_is_not_a_failure(self):
        row = {"extract_ok": True, "compile_status": "ok", "test_status": "pass"}
        c = classify_row(row)
        self.assertIsNone(c["class"])

    def test_gen_error_is_infrastructure(self):
        row = {"extract_ok": False, "test_status": "not_attempted",
               "extract_reason": "skipped (generation error: boom)",
               "gen_error": "boom"}
        c = classify_row(row)
        self.assertIsNone(c["class"])


class TestRunBHandCounts(unittest.TestCase):
    """M1.5 hand-counts: p0 compile_error, p1 renamed_or_self,
    p2+p3 truncated_canvas."""

    def test_histogram_matches_hand_count(self):
        rows = [
            {"i": 0, "extract_ok": True, "compile_status": "compile_error",
             "test_status": "not_attempted", "extract_reason": "ok",
             "raw_path": str(RUN_B / "p0.txt")},
            {"i": 1, "extract_ok": False, "test_status": "not_attempted",
             "extract_reason": "required symbol(s) 'ring_new', 'ring_push', "
                               "'ring_pop' not defined anywhere in extracted "
                               "code (prompt echo or wrong problem?)"},
            {"i": 2, "extract_ok": False, "test_status": "not_attempted",
             "extract_reason": "required symbol(s) 'ring_pop' not defined "
                               "anywhere in extracted code (prompt echo or "
                               "wrong problem?)"},
            {"i": 3, "extract_ok": False, "test_status": "not_attempted",
             "extract_reason": "required symbol(s) 'ring_push', 'ring_pop' "
                               "not defined anywhere in extracted code "
                               "(prompt echo or wrong problem?)"},
        ]
        raws = {1: _raw("p1.txt"), 2: _raw("p2.txt"), 3: _raw("p3.txt")}
        cls = []
        for r in rows:
            c = classify_row(r, raw_text=raws.get(r["i"]))
            cls.append(c)
            print(f"p{r['i']}: {c['class']}  ev={c['evidence']}")
        h = histogram(cls)
        self.assertEqual(h, {"compile_error": 1, "renamed_or_self": 1,
                             "truncated_canvas": 2})


class TestHistogram(unittest.TestCase):
    def test_histogram_skips_empty(self):
        h = histogram([{"class": None}, {"class": "echo"},
                       {"class": "unknown-ish"}])
        self.assertEqual(h, {"echo": 1, "unclassified": 2})


if __name__ == "__main__":
    unittest.main(verbosity=2)
