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


if __name__ == "__main__":
    unittest.main(verbosity=2)
