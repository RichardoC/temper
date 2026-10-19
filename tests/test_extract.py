"""Unit tests for temper.model.extract (stdlib unittest, no model needed)."""

from __future__ import annotations

import unittest

from temper.model.extract import extract_rust


FENCED = """Here is the implementation:

```rust
pub fn sort_i32(v: &mut Vec<i32>) {
    v.sort_unstable();
}
```

This sorts in place.
"""

FENCED_TWO = """```rust
fn helper(x: u8) -> u8 { x }
```
And the main one:
```rust
pub fn sort_i32(v: &mut Vec<i32>) {
    let n = v.len();
    for i in 0..n {
        for j in i + 1..n {
            if v[j] < v[i] { v.swap(i, j); }
        }
    }
}
```
"""

BARE = """Sure! Here is a Rust function that does that:

pub fn sort_i32(v: &mut Vec<i32>) {
    v.sort_unstable();
}

Hope this helps."""

ECHO = """The signature is:

```rust
pub fn sort_i32(v: &mut Vec<i32>)
```

as requested above. Let me know if you need more!"""

WRONG_PROBLEM = """```rust
pub fn bubble(xs: &mut Vec<i32>) {
    xs.sort_unstable();
}
```
"""

MASKED = """```rust
pub fn sort_i32(v: &mut Vec<i32>) {
    v.sort_unst|mask|able();
}
```
"""

TRUNCATED_CANVAS = """```rust
pub fn sort_i32(v: &mut Vec<i32>) {
    v.sort_unstable();
"""

RS_TAG = "```rs\npub fn sort_i32(v: &mut Vec<i32>) {\n    v.sort_unstable();\n}\n```"


class TestExtract(unittest.TestCase):
    def test_fenced_single(self):
        r = extract_rust(FENCED, must_define=["sort_i32"])
        self.assertTrue(r.ok, r.reason)
        self.assertIn("sort_unstable", r.code)
        self.assertNotIn("Here is", r.code)
        self.assertNotIn("Hope this helps", r.code)

    def test_fenced_prefers_largest_rust_block(self):
        r = extract_rust(FENCED_TWO, must_define=["sort_i32"])
        self.assertTrue(r.ok, r.reason)
        self.assertIn("for i in 0..n", r.code)
        self.assertNotIn("helper", r.code)

    def test_bare_span(self):
        r = extract_rust(BARE, must_define=["sort_i32"])
        self.assertTrue(r.ok, r.reason)
        self.assertIn("sort_unstable", r.code)
        self.assertNotIn("Sure!", r.code)

    def test_prompt_echo_rejected_by_must_define(self):
        # signature-only echo: no body -> rejected (either reason is fine)
        r = extract_rust(ECHO, must_define=["sort_i32"])
        self.assertFalse(r.ok)

    def test_symbol_missing_rejected_with_reason(self):
        code = "```rust\npub fn other(v: &mut Vec<i32>) { v.push(0); }\n```"
        r = extract_rust(code, must_define=["sort_i32"])
        self.assertFalse(r.ok)
        self.assertIn("not defined", r.reason)

    def test_wrong_problem_rejected(self):
        r = extract_rust(WRONG_PROBLEM, must_define=["sort_i32"])
        self.assertFalse(r.ok)

    def test_mask_tokens_rejected(self):
        r = extract_rust(MASKED, must_define=["sort_i32"])
        self.assertFalse(r.ok)
        self.assertIn("mask", r.reason)

    def test_unbalanced_truncated_canvas_rejected(self):
        r = extract_rust(TRUNCATED_CANVAS, must_define=["sort_i32"])
        self.assertFalse(r.ok)

    def test_rs_language_tag_accepted(self):
        r = extract_rust(RS_TAG, must_define=["sort_i32"])
        self.assertTrue(r.ok, r.reason)

    def test_empty(self):
        self.assertFalse(extract_rust("").ok)
        self.assertFalse(extract_rust("   \n").ok)

    def test_struct_must_define(self):
        code = "```rust\npub struct RingBuf { cap: usize }\npub fn ring_new(c: usize) -> RingBuf { RingBuf { cap: c } }\n```"
        r = extract_rust(code, must_define=["RingBuf", "ring_new"])
        self.assertTrue(r.ok, r.reason)
        r2 = extract_rust(code, must_define=["ring_push"])
        self.assertFalse(r2.ok)

    IMPL_METHOD_ANSWER = '''```rust
pub struct RingBuf {
    cap: usize,
    buf: Vec<i64>,
}

impl RingBuf {
    pub fn ring_new(capacity: usize) -> RingBuf {
        RingBuf { cap: capacity, buf: Vec::new() }
    }

    pub fn ring_push(&mut self, v: i64) {
        if self.buf.len() == self.cap {
            self.buf.remove(0);
        }
        self.buf.push(v);
    }

    pub fn ring_pop(&mut self) -> Option<i64> {
        self.buf.pop()
    }
}
```
'''

    def test_impl_method_answer_rejected_with_api_shape_reason(self):
        # the M1 ringbuf failure class: right logic, wrong API shape —
        # required fns present but only as methods inside `impl RingBuf`
        r = extract_rust(self.IMPL_METHOD_ANSWER,
                         must_define=["RingBuf", "ring_new", "ring_push", "ring_pop"])
        self.assertFalse(r.ok)
        self.assertIn("inherent impl method", r.reason)
        self.assertIn("ring_new", r.reason)
        # struct IS at module scope -> must not be flagged
        self.assertNotIn("'RingBuf'", r.reason)

    def test_renamed_methods_rejected_as_undefined(self):
        # methods renamed to new/push/pop: the required symbol truly is absent
        code = self.IMPL_METHOD_ANSWER.replace(
            "ring_new", "new").replace("ring_push", "push").replace("ring_pop", "pop")
        r = extract_rust(code, must_define=["ring_new"])
        self.assertFalse(r.ok)
        self.assertIn("not defined anywhere", r.reason)

    def test_impl_block_plus_module_fns_accepted(self):
        # a private impl method must not shadow or block module-level fns
        code = '''```rust
pub struct RingBuf { cap: usize, buf: Vec<i64> }

impl RingBuf {
    fn capacity(&self) -> usize {
        if self.cap == 0 { 0 } else { self.cap }
    }
}

pub fn ring_new(capacity: usize) -> RingBuf { RingBuf { cap: capacity, buf: Vec::new() } }
pub fn ring_push(rb: &mut RingBuf, v: i64) { rb.buf.push(v); }
pub fn ring_pop(rb: &mut RingBuf) -> Option<i64> { rb.buf.pop() }
```
'''
        r = extract_rust(code,
                         must_define=["RingBuf", "ring_new", "ring_push", "ring_pop"])
        self.assertTrue(r.ok, r.reason)

    def test_all_missing_symbols_listed(self):
        code = "```rust\npub fn helper() {}\n```"
        r = extract_rust(code, must_define=["ring_new", "ring_push"])
        self.assertFalse(r.ok)
        self.assertIn("not defined anywhere", r.reason)
        self.assertIn("ring_new", r.reason)
        self.assertIn("ring_push", r.reason)


# --- M2.3 phase-0: fenced→bare-span fallback ------------------------------
# Real fixture: the M2.2 B2 pilot's HumanEval_15_string_sequence p0 raw
# sidecar, verbatim (all 4 samples of this problem failed extraction the
# same way). The model re-rendered the doc comments with `/// ``` `
# example fences; those embedded ``` pairs terminate the outer ```rust
# fence early, so the "largest fenced block" is prose — while the complete
# `fn string_sequence` sits right after it and was thrown away.
EMBEDDED_FENCE_DOC_ECHO = '''```rust
/// Return a string containing the sequence from 0 to n inclusive, space-separated.
.
/// # Examples
////// ```
 string_sequence(0)
/// String::from("0")
/// ```
 string_sequence(5)
///String::from("0 1 2 3 4 5")
/// ```
fn string_sequence(n: isize) -> String {
    // Create a vector to hold the numbers
    let mut numbers = Vec::new();
    
    // Iterate from 0 to n inclusive
    for i in ..=n {
        numbers.push(i);
    }
    
    // Convert the numbers into a string and join them with a space
    let result = numbers.into_iter().collect(" ");
    result
}
'''


class TestFenceFallbackToBareSpan(unittest.TestCase):
    def test_embedded_fence_doc_echo_recovered(self):
        # the exact M2.2 pilot exclusion class (4 rows): fenced path poisoned
        # by ``` inside doc-comment echo; bare span must recover the fn
        r = extract_rust(EMBEDDED_FENCE_DOC_ECHO, must_define=["string_sequence"])
        self.assertTrue(r.ok, r.reason)
        self.assertTrue(r.code.startswith("fn string_sequence"))
        self.assertIn("for i in", r.code)
        # the doc-comment echo must NOT leak into the extracted code
        self.assertNotIn("///", r.code)

    def test_unbalanced_fenced_block_with_later_bare_span_recovered(self):
        # fenced block unbalanced (truncated fence), complete fn after it
        text = (
            "```rust\nsure here is some prose { without closing\n"
            "```\n\n"
            "pub fn sort_i32(v: &mut Vec<i32>) {\n    v.sort_unstable();\n}\n"
        )
        r = extract_rust(text, must_define=["sort_i32"])
        self.assertTrue(r.ok, r.reason)
        self.assertIn("sort_unstable", r.code)

    def test_unbalanced_fenced_no_bare_span_still_rejected(self):
        # TRUNCATED_CANVAS shape: fenced + unbalanced, no item header+balance
        # anywhere -> still rejected (historical combined message)
        r = extract_rust(TRUNCATED_CANVAS, must_define=["sort_i32"])
        self.assertFalse(r.ok)
        self.assertIn("no fenced rust block", r.reason)

    def test_mask_residue_keeps_incomplete_canvas_diagnosis(self):
        # the fallback must NOT swallow the mask-token class (regression
        # guard for the bench driver's failure-path contract)
        r = extract_rust("```rust\n<|mask|>\n```", must_define=["sort_i32"])
        self.assertFalse(r.ok)
        self.assertIn("incomplete canvas", r.reason)

    def test_fallback_never_degrades_passing_extraction(self):
        # a healthy fenced block must be used as-is (fallback not even
        # considered) — the pre-existing fenced tests already pin this;
        # this one asserts it with prose AFTER the fence that a naive
        # largest-bare-span would prefer
        text = FENCED + "\nfn decoy() { { }\n"
        r = extract_rust(text, must_define=["sort_i32"])
        self.assertTrue(r.ok, r.reason)
        self.assertIn("sort_unstable", r.code)
        self.assertNotIn("decoy", r.code)

    def test_recovered_span_exports_as_body_fragment(self):
        # end-to-end instrument check on the real disease: the recovered
        # extraction must convert to the official completion shape via
        # export.to_body_fragment (prompt from the vendored dataset)
        from temper.multipl_e.export import load_benchmark, to_body_fragment
        td = load_benchmark("humaneval-rs")["HumanEval_15_string_sequence"]
        r = extract_rust(EMBEDDED_FENCE_DOC_ECHO, must_define=["string_sequence"])
        self.assertTrue(r.ok, r.reason)
        body = to_body_fragment(r.code, td["prompt"])
        self.assertIsNotNone(body)
        self.assertNotIn("fn string_sequence", body)  # body fragment, no sig
        self.assertIn("numbers.push(i);", body)


if __name__ == "__main__":
    unittest.main(verbosity=2)
