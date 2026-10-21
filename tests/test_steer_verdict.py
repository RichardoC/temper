"""Verifier-ladder + D7 verdict-config freeze tests (M3.0).

Pins, on the REAL vendored problems where it matters:
  * the ghost closure produces a library-checkable program whose line map
    correctly separates committed content from holes and synthesized closers;
  * the informed-class filter accepts the type-level blunders the corrected
    s256 forensics found (E0277 on `f64: Ord`, E0282 annotations) and refuses
    hole-adjacent or parse-level noise;
  * at the FINAL checkpoint the screen is exact (any error = compile-dead);
  * the frozen config's embedded sha detects tampering (D7 acceptance);
  * R0 flags the prose-after-fence class and mask-share divergence;
  * R2 flags committed stubs only.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from temper.multipl_e.bench import load_problems
from temper.steer.verdict import (
    CONFIG_SCHEMA,
    DEFAULT_CONFIG_PATH,
    VerdictConfigError,
    compute_config_sha256,
    final_checkpoint_clean,
    find_rustc,
    ghost_close,
    informed_fatal_evidence,
    load_verdict_config,
    r0_shape,
    r1_typecheck,
    r2_stub_markers,
)
from temper.steer.view import SteerProblemView

REPO = Path(__file__).resolve().parents[1]


def _view(name_fragment: str) -> SteerProblemView:
    for td in load_problems("humaneval-rs"):
        if name_fragment in td["name"]:
            return SteerProblemView.from_task(td)
    raise AssertionError(f"no vendored problem matching {name_fragment!r}")


def _module(view: SteerProblemView, body: str) -> str:
    """A whole-function canvas the way the model emits it (one rust fence)."""
    return "```rust\n" + view.signature_line + "\n" + body + "\n}\n```"


def _have_rustc() -> bool:
    try:
        find_rustc()
        return True
    except FileNotFoundError:
        return False


class TestVerdictConfigFreeze(unittest.TestCase):
    """D7: the config is versioned, sha-frozen, and tamper-DETECTED."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_committed_config_loads_and_verifies(self):
        cfg = load_verdict_config()
        self.assertEqual(cfg.sha256, compute_config_sha256(cfg.raw))
        self.assertGreaterEqual(len(cfg.informed_error_classes), 5)
        self.assertIn("E0277", cfg.informed_error_classes)
        self.assertIn("E0282", cfg.informed_error_classes)
        self.assertIn("E0507", cfg.informed_error_classes)
        self.assertGreaterEqual(cfg.patience_checkpoints, 1)
        self.assertTrue(cfg.r0_flags_fatal)
        self.assertIn("todo!(", cfg.stub_markers)
        self.assertEqual(cfg.checkpoint_grid, (0.25, 0.5, 0.75, 1.0))

    def test_tampering_a_verdict_parameter_is_detected(self):
        """D7 acceptance: edit one informed class -> load must REFUSE."""
        raw = json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
        raw["informed_error_classes"] = list(raw["informed_error_classes"]) + ["E9999"]
        tampered = self.tmp / "tampered.json"
        tampered.write_text(json.dumps(raw), encoding="utf-8")
        with self.assertRaises(VerdictConfigError) as ctx:
            load_verdict_config(tampered)
        self.assertIn("TAMPERED", str(ctx.exception))

    def test_tampering_patience_is_detected(self):
        raw = json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
        raw["patience"]["consecutive_checkpoints"] += 1
        tampered = self.tmp / "tampered2.json"
        tampered.write_text(json.dumps(raw), encoding="utf-8")
        with self.assertRaises(VerdictConfigError):
            load_verdict_config(tampered)

    def test_refreezing_after_a_labeled_bump_is_accepted(self):
        """Re-pinning is legal ONLY as a labeled edit: bump version, re-stamp,
        and the file verifies again (the D4 repin path)."""
        raw = json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
        raw["version"] = "v0-test-repin"
        raw["patience"]["consecutive_checkpoints"] = 3
        raw["sha256"] = compute_config_sha256(raw)
        p = self.tmp / "repinned.json"
        p.write_text(json.dumps(raw), encoding="utf-8")
        cfg = load_verdict_config(p)
        self.assertEqual(cfg.version, "v0-test-repin")
        self.assertEqual(cfg.patience_checkpoints, 3)

    def test_missing_sha_is_refused(self):
        raw = json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
        raw.pop("sha256")
        p = self.tmp / "unfrozen.json"
        p.write_text(json.dumps(raw), encoding="utf-8")
        with self.assertRaises(VerdictConfigError) as ctx:
            load_verdict_config(p)
        self.assertIn("never frozen", str(ctx.exception))

    def test_wrong_schema_is_refused(self):
        raw = json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
        raw["schema"] = "something-else/9"
        raw["sha256"] = compute_config_sha256(raw)
        p = self.tmp / "schema.json"
        p.write_text(json.dumps(raw), encoding="utf-8")
        with self.assertRaises(VerdictConfigError):
            load_verdict_config(p)

    def test_formatting_churn_does_not_change_the_digest(self):
        raw = json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
        p = self.tmp / "reindented.json"
        p.write_text(json.dumps(raw, indent=7, sort_keys=True), encoding="utf-8")
        self.assertEqual(load_verdict_config(p).sha256,
                         load_verdict_config().sha256)


class TestGhostClosure(unittest.TestCase):
    def setUp(self):
        self.view = _view("has_close_elements")

    def test_closure_balances_braces_and_keeps_prompt_verbatim(self):
        ghost = ghost_close(self.view, _module(self.view, "    true"))
        self.assertTrue(ghost.text.startswith(self.view.prompt.rstrip("\n")))
        self.assertEqual(ghost.text.count("{"), ghost.text.count("}"))
        self.assertEqual(ghost.tail_lines, (len(ghost.text.splitlines()) - 1,))

    def test_mask_runs_become_todo_and_are_not_evidence(self):
        canvas = ("```rust\n" + self.view.signature_line +
                  "\n    let mut s = numbers;\n    <|mask|><|mask|>\n    false\n}\n```")
        ghost = ghost_close(self.view, canvas)
        self.assertIn("todo!()", ghost.text)
        self.assertTrue(ghost.hole_lines)
        for line in ghost.hole_lines:
            self.assertFalse(ghost.is_committed_line(line + 1, 0))

    def test_unterminated_fence_still_closes(self):
        ghost = ghost_close(self.view, "```rust\n" + self.view.signature_line
                            + "\n    true")
        self.assertEqual(ghost.text.count("{"), ghost.text.count("}"))

    def test_no_tests_ever_enter_the_program(self):
        td = next(t for t in load_problems("humaneval-rs")
                  if "has_close_elements" in t["name"])
        ghost = ghost_close(self.view, _module(self.view, "    true"))
        # the vendored tests bind the fn and assert on it; neither the binding
        # line nor the fixture bodies may appear in the checked program
        self.assertNotIn("let candidate =", ghost.text)
        for line in td["tests"].splitlines():
            if line.strip() and line.strip() != "}":
                self.assertNotIn(line.strip(), ghost.text)


@unittest.skipUnless(_have_rustc(), "rustc not available on this host")
class TestR1InformedClassFilter(unittest.TestCase):
    """The §3.2 measured profile, pinned on real problems."""

    @classmethod
    def setUpClass(cls):
        cls.cfg = load_verdict_config()

    def _verdict(self, view, body):
        return r1_typecheck(ghost_close(view, _module(view, body)),
                            hole_proximity=self.cfg.hole_proximity_lines)

    def test_type_level_blunder_is_informed_fatal(self):
        # HumanEval_0's real deterministic failure: `sort()` on Vec<f64>
        view = _view("has_close_elements")
        v = self._verdict(view, "    let mut s = numbers;\n    s.sort();\n    false")
        codes = [e.code for e in informed_fatal_evidence(v, self.cfg)]
        self.assertIn("E0277", codes)

    def test_annotation_blunder_is_informed_fatal(self):
        # HumanEval_4's E0282 class
        view = _view("mean_absolute_deviation")
        v = self._verdict(
            view, "    let mut total = 0.0;\n    for x in &numbers { total += *x; }\n"
                  "    let mean = total / (numbers.len() as f64);\n"
                  "    let devs = vec![];\n    devs")
        codes = [e.code for e in informed_fatal_evidence(v, self.cfg)]
        self.assertTrue(codes, f"expected an informed code, got {v.error_codes}")

    def test_clean_body_has_no_evidence(self):
        view = _view("has_close_elements")
        v = self._verdict(
            view,
            "    let mut s = numbers.clone();\n"
            "    s.sort_by(|a, b| a.partial_cmp(b).unwrap());\n"
            "    for w in s.windows(2) {\n"
            "        if (w[1] - w[0]).abs() < threshold { return true; }\n"
            "    }\n    false")
        self.assertEqual(informed_fatal_evidence(v, self.cfg), [])
        self.assertTrue(final_checkpoint_clean(v))

    def test_parse_error_is_not_informed_class(self):
        # a mid-expression parse error carries no E-code -> never mid-canvas
        # kill evidence (it IS fatal at the final checkpoint, see next test)
        view = _view("has_close_elements")
        v = self._verdict(view, "    let mut = = ;\n    false")
        self.assertEqual(informed_fatal_evidence(v, self.cfg), [])
        self.assertFalse(final_checkpoint_clean(v))

    def test_hole_adjacent_evidence_is_discarded(self):
        """A diagnostic that only exists because of a hole boundary must not
        count: the §3.2 false-kill mechanism, pinned directly."""
        view = _view("has_close_elements")
        canvas = ("```rust\n" + view.signature_line +
                  "\n    let mut s = numbers;\n"
                  "    <|mask|><|mask|><|mask|><|mask|><|mask|>\n}\n```")
        ghost = ghost_close(view, canvas)
        v = r1_typecheck(ghost, hole_proximity=self.cfg.hole_proximity_lines)
        for err in informed_fatal_evidence(v, self.cfg):
            self.assertTrue(ghost.is_committed_line(
                err.primary_line, self.cfg.hole_proximity_lines))


@unittest.skipUnless(_have_rustc(), "rustc not available on this host")
class TestR0AndR2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_verdict_config()
        cls.view = _view("has_close_elements")

    def test_prose_after_fence_is_fatal(self):
        canvas = _module(self.view, "    true") + "\nHere is an explanation.\n"
        r0 = r0_shape(self.view, canvas, mask_share=0.0, expected_mask_share=0.0,
                      checkpoint_fraction=1.0, config=self.cfg)
        self.assertIn("prose_after_fence", r0.flags)
        self.assertTrue(r0.is_fatal(self.cfg))

    def test_unterminated_fence_is_fatal(self):
        canvas = "```rust\n" + self.view.signature_line + "\n    true"
        r0 = r0_shape(self.view, canvas, mask_share=0.0, expected_mask_share=0.0,
                      checkpoint_fraction=0.5, config=self.cfg)
        self.assertIn("fence_unterminated", r0.flags)

    def test_normal_canvas_has_no_flags(self):
        r0 = r0_shape(self.view, _module(self.view, "    true"),
                      mask_share=0.0, expected_mask_share=0.0,
                      checkpoint_fraction=1.0, config=self.cfg)
        self.assertEqual(r0.flags, ())
        self.assertFalse(r0.is_fatal(self.cfg))

    def test_mask_share_divergence_flagged(self):
        r0 = r0_shape(self.view, _module(self.view, "    true"),
                      mask_share=0.9, expected_mask_share=0.25,
                      checkpoint_fraction=0.75, config=self.cfg)
        self.assertIn("mask_share_divergent", r0.flags)

    def test_r2_flags_only_committed_stubs(self):
        self.assertEqual(r2_stub_markers("    todo!()", self.cfg),
                         ("todo!(",))
        self.assertEqual(r2_stub_markers("    return true", self.cfg), ())
        self.assertEqual(r2_stub_markers("    unimplemented!()", self.cfg),
                         ("unimplemented!(",))


if __name__ == "__main__":
    unittest.main()
