"""Unit tests for the M2.1 L0.5 repair pass (temper.model.extract.repair_code).

The repair pass is deterministic + conservative and never re-prompts the
model. These tests exercise every strategy to exhaustion on synthetic cases
PLUS the real run-B fixtures (bench/ringbuf/runs/20261018T224453Z.raw/).
"""

from __future__ import annotations

import unittest
from pathlib import Path

from temper.model.extract import _check_code, extract_rust, repair_code

REPO = Path(__file__).resolve().parents[1]
RUN_B = REPO / "bench" / "ringbuf" / "runs" / "20261018T224453Z.raw"
MUST = ["RingBuf", "ring_new", "ring_push", "ring_pop"]


class TestDegenerateDrop(unittest.TestCase):
    def test_unreferenced_degenerate_item_dropped(self):
        code = ("pub fn target(x: i32) -> i32 { x + 1 }\n"
                "pub Stray { fn orphan() {} }\n")
        rp = repair_code(code)
        self.assertTrue(rp.repaired)
        self.assertNotIn("Stray", rp.code)
        self.assertTrue(rp.valid)
        self.assertEqual(rp.attempts[0]["strategy"], "degenerate-items")
        self.assertIn("drop-degenerate-item",
                      [a["action"] for a in rp.attempts[0]["actions"]])

    def test_referenced_degenerate_item_not_dropped(self):
        # identifier referenced elsewhere -> dropping would dangle it
        code = ("pub struct T { x: i32 }\n"
                "pub T { fn mk(x: i32) -> T { T { x } } }\n"
                "pub fn use_t() -> i32 { T { x: 1 }.x }\n")
        rp = repair_code(code)
        self.assertTrue(rp.repaired)  # converted, not dropped
        self.assertIn("impl T {", rp.code)
        self.assertNotIn("pub T {", rp.code)

    def test_non_keyword_header_required(self):
        # valid items are never touched
        code = "pub fn f() {}\npub struct S { x: i32 }\n"
        rp = repair_code(code)
        self.assertFalse(rp.repaired)
        self.assertEqual(rp.code, code)


class TestDegenerateConvert(unittest.TestCase):
    P0_LIKE = ("pub struct RingBuf {\n    cap: usize,\n}\n"
               "pub RingBuf {\n    fn new(cap: usize) -> RingBuf {\n"
               "        RingBuf { cap }\n    }\n}\n"
               "pub fn ring_new(capacity: usize) -> RingBuf {\n"
               "    RingBuf::new(capacity)\n}\n"
               "pub fn ring_push(rb: &mut RingBuf, v: i64) {\n"
               "    rb.cap = rb.cap;\n}\n"
               "pub fn ring_pop(rb: &mut RingBuf) -> Option<i64> {\n"
               "    None\n}\n")

    def test_convert_pseudo_impl_over_existing_struct(self):
        rp = repair_code(self.P0_LIKE, MUST)
        self.assertTrue(rp.repaired)
        self.assertIn("impl RingBuf {", rp.code)
        self.assertNotIn("pub RingBuf {", rp.code)
        self.assertTrue(rp.valid, rp.attempts)
        self.assertIn("convert-pseudo-impl",
                      [a["action"] for a in rp.attempts[0]["actions"]])

    def test_unreferenced_degenerate_item_dropped(self):
        # no type definition + zero references elsewhere -> drop is the
        # conservative repair (nothing can dangle)
        code = ("pub RingBuf { fn new() -> RingBuf { RingBuf {} } }\n"
                "pub fn f() -> i32 { 1 }\n")
        rp = repair_code(code)
        self.assertTrue(rp.repaired)
        self.assertTrue(rp.valid)
        self.assertNotIn("RingBuf", rp.code)
        self.assertIn("drop-degenerate-item",
                      [a["action"] for a in rp.attempts[0]["actions"]])

    def test_single_line_fn_body_convertible(self):
        # one-line degenerate whose body is exactly one fn: convertible
        code = ("pub struct T { x: i32 }\n"
                "pub T { fn mk() -> T { T { x: 0 } } }\n"
                "pub fn f() -> i32 { 1 }\n")
        rp = repair_code(code)
        self.assertIn("impl T {", rp.code)
        self.assertTrue(rp.valid)

    def test_no_convert_when_body_has_non_fn_items(self):
        code = ("pub struct T { x: i32 }\n"
                "pub T { const X: i32 = 1; }\n"
                "pub fn f() -> i32 { 1 }\n")
        rp = repair_code(code)
        self.assertFalse(rp.repaired)  # referenced + non-fn body: skip
        self.assertIn("skip-degenerate-item",
                      [a["action"] for a in rp.attempts[0]["actions"]])

    def test_conversion_keeps_balance_and_must_define(self):
        rp = repair_code(self.P0_LIKE, MUST)
        self.assertEqual(rp.code.count("{"), rp.code.count("}"))
        self.assertEqual(_check_code(rp.code, MUST), "")


class TestStripTrailingProse(unittest.TestCase):
    def test_prose_after_final_item_stripped(self):
        code = ("pub fn f() -> i32 {\n    1\n}\n"
                "### Explanation:\n"
                "1. **The struct**:\n"
                "   - points to the start\n")
        rp = repair_code(code)
        self.assertTrue(rp.repaired)
        self.assertTrue(rp.valid)
        self.assertNotIn("Explanation", rp.code)
        self.assertIn("strip-trailing-prose",
                      [a["strategy"] for a in rp.attempts if a["changed"]])

    def test_prose_with_braces_not_touched(self):
        # brace-bearing tails are not prose under a conservative reading
        code = "pub fn f() -> i32 {\n    1\n}\nuse std::{fmt};\n"
        rp = repair_code(code)
        self.assertFalse(rp.repaired)

    def test_comments_kept(self):
        code = "pub fn f() -> i32 {\n    1\n}\n// a legit trailing comment\n"
        rp = repair_code(code)
        self.assertFalse(rp.repaired)
        self.assertIn("legit trailing comment", rp.code)

    def test_run_b_p0_trailing_prose_already_outside_code(self):
        raw = (RUN_B / "p0.txt").read_text()
        ex = extract_rust(raw, MUST)
        self.assertTrue(ex.ok)
        self.assertNotIn("Explanation", ex.code)  # fenced extraction excludes it


class TestOrderAndGates(unittest.TestCase):
    def test_strategy_order_a_then_b(self):
        code = ("pub fn f() -> i32 {\n    1\n}\n"
                "pub Stray { fn g() {} }\n"
                "## Notes\nmore words\n")
        rp = repair_code(code)
        strategies = [a["strategy"] for a in rp.attempts]
        self.assertEqual(strategies, ["degenerate-items", "strip-trailing-prose"])
        self.assertTrue(rp.repaired and rp.valid)

    def test_both_strategies_always_attempted_and_recorded(self):
        # design: try (a) then (b), each attempt recorded — even when (a)
        # already suffices, (b) is attempted and reports changed=False
        code = ("pub fn f() -> i32 {\n    1\n}\n"
                "pub Stray { fn g() {} }\n")
        rp = repair_code(code)
        self.assertEqual([a["strategy"] for a in rp.attempts],
                         ["degenerate-items", "strip-trailing-prose"])
        self.assertTrue(rp.attempts[0]["changed"])
        self.assertFalse(rp.attempts[1]["changed"])
        self.assertTrue(rp.repaired and rp.valid)

    def test_already_valid_code_repaired_false_but_recorded(self):
        code = "pub fn f() -> i32 {\n    1\n}\n"
        rp = repair_code(code)
        self.assertFalse(rp.repaired)
        self.assertTrue(rp.valid)
        self.assertFalse(any(a["changed"] for a in rp.attempts))

    def test_never_fixes_mask_tokens(self):
        code = "pub fn f() {\n    let x = |mask|;\n}\n"
        rp = repair_code(code)
        self.assertFalse(rp.valid)
        self.assertFalse(any(a["changed"] for a in rp.attempts))

    def test_unrepairable_left_untouched(self):
        # missing required symbol: no strategy applies, code unchanged
        code = "pub fn other() {}\n"
        rp = repair_code(code, ["sort_i32"])
        self.assertFalse(rp.repaired)
        self.assertFalse(rp.valid)
        self.assertEqual(rp.code, code)


class TestRunBFixtures(unittest.TestCase):
    """The real M1.5 fixtures (bench/ringbuf/runs/20261018T224453Z.raw/)."""

    def test_p0_reaches_valid_code(self):
        raw = (RUN_B / "p0.txt").read_text()
        ex = extract_rust(raw, MUST)
        rp = repair_code(ex.code, MUST)
        self.assertTrue(rp.repaired)
        self.assertTrue(rp.valid)
        # the repaired module still defines exactly the required API
        self.assertEqual(_check_code(rp.code, MUST), "")
        self.assertIn("impl RingBuf {", rp.code)
        self.assertIn("pub fn ring_push(rb: &mut RingBuf, v: i64)", rp.code)

    def test_p1_not_repaired_but_recorded(self):
        # renamed/self-parameterized API: no conservative strategy applies
        raw = (RUN_B / "p1.txt").read_text()
        ex = extract_rust(raw, MUST)
        self.assertFalse(ex.ok)
        self.assertTrue(ex.code)  # span preserved for repair
        rp = repair_code(ex.code, MUST)
        self.assertFalse(rp.repaired)
        self.assertFalse(rp.valid)

    def test_p2_p3_truncated_not_repaired(self):
        for name in ("p2.txt", "p3.txt"):
            raw = (RUN_B / name).read_text()
            ex = extract_rust(raw, MUST)
            self.assertFalse(ex.ok)
            rp = repair_code(ex.code or "", MUST)
            self.assertFalse(rp.repaired, name)
            self.assertFalse(rp.valid, name)


class TestExtractRegression(unittest.TestCase):
    """The repair hook must NOT alter currently-working extractions."""

    def test_must_define_failure_keeps_span_for_repair(self):
        ex = extract_rust("```rust\npub fn other() {}\n```", ["sort_i32"])
        self.assertFalse(ex.ok)
        self.assertIn("not defined anywhere", ex.reason)
        # span preserved so the repair pass has material
        self.assertIn("pub fn other", ex.code)

    def test_mask_failure_keeps_code_empty(self):
        ex = extract_rust(
            "```rust\npub fn sort_i32(v: &mut Vec<i32>) {\n v.sort_unst|mask|able();\n}\n```",
            ["sort_i32"])
        self.assertFalse(ex.ok)
        self.assertIn("mask", ex.reason)
        self.assertEqual(ex.code, "")

    def test_ok_extraction_unchanged(self):
        ex = extract_rust(
            "```rust\npub fn sort_i32(v: &mut Vec<i32>) {\n v.sort_unstable();\n}\n```",
            ["sort_i32"])
        self.assertTrue(ex.ok)
        self.assertIn("sort_unstable", ex.code)


if __name__ == "__main__":
    unittest.main(verbosity=2)
