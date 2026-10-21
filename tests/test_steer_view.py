"""D5: held-in guard — the SteerProblemView allowlist (M3.0, design §3.1/§7).

The hard rule: the steer-side verifier may never read the MultiPL-E ``tests``
string, container output, or any scoring fixture. Enforcement is structural
(an allowlist constructor), so these tests attack the constructor: adversarial
task dicts, smuggled keys, and a static scan asserting no reward-side module
mentions the forbidden fields.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

from temper.multipl_e.bench import load_problems
from temper.steer.view import ALLOWED_TASK_FIELDS, SteerProblemView

REPO = Path(__file__).resolve().parents[1]
# Every module that runs at steer time (verdict ladder, loop, log, metering).
REWARD_SIDE_MODULES = (
    "temper/steer/view.py",
    "temper/steer/verdict.py",
    "temper/steer/loop.py",
    "temper/steer/log.py",
    "temper/steer/metering.py",
)


class TestAllowlistConstructor(unittest.TestCase):
    def test_allowlist_is_exactly_name_and_prompt(self):
        self.assertEqual(ALLOWED_TASK_FIELDS, ("name", "prompt"))

    def test_real_task_yields_view_without_tests(self):
        td = load_problems("humaneval-rs", 1)[0]
        view = SteerProblemView.from_task(td)
        self.assertEqual(view.name, td["name"])
        self.assertEqual(view.prompt, td["prompt"])
        self.assertEqual(view.fn_name, "has_close_elements")
        # the view has no attribute through which tests are reachable
        for attr in dir(view):
            if attr.startswith("__"):
                continue
            val = getattr(view, attr)
            if isinstance(val, str):
                self.assertNotIn("let candidate =", val)
                self.assertNotIn(td["tests"], val)

    def test_serialized_payload_carries_no_tests(self):
        td = load_problems("humaneval-rs", 1)[0]
        payload = SteerProblemView.from_task(td).as_payload()
        self.assertEqual(set(payload), {"name", "prompt", "fn_name",
                                        "signature_line"})
        self.assertNotIn(td["tests"], repr(payload))

    def test_adversarial_task_with_readable_tests_yields_empty_view(self):
        """The design's D5 acceptance case: a hypothetical task dict whose
        tests are perfectly readable must still produce a view that cannot
        reach them."""
        adversarial = {
            "name": "HumanEval_XX_adversarial",
            "prompt": "/// doc\nfn f(x: i32) -> i32 {\n",
            "tests": 'fn main() { assert_eq!(f(1), 2); }  // SECRET',
            "doctests": "SECRET_DOCTESTS",
            "original": "SECRET_ORIGINAL",
            "stop_tokens": ["\n}"],
        }
        view = SteerProblemView.from_task(adversarial)
        blob = repr(view) + repr(view.as_payload())
        for secret in ("SECRET", "let candidate", "assert_eq!"):
            self.assertNotIn(secret, blob)
        self.assertFalse(hasattr(view, "tests"))

    def test_missing_key_fails_loudly(self):
        with self.assertRaises(KeyError):
            SteerProblemView.from_task({"prompt": "fn f() -> i32 {\n"})

    def test_prompt_without_fn_header_refused(self):
        with self.assertRaises(ValueError):
            SteerProblemView.from_task({"name": "x", "prompt": "just prose"})


class TestNoRewardModuleTouchesForbiddenFields(unittest.TestCase):
    """Static allowlist check (the D5 'grep rule' as an executable test).

    Reward-side modules never reference ``tests``/``doctests``/``original``/
    ``stop_tokens`` task keys at all — the only way to reach them would be a
    subscript of a task dict, which is exactly what the allowlist removed.
    """

    FORBIDDEN_SUBSCRIPTS = ('"tests"', "'tests'", '"doctests"', "'doctests'",
                            '"original"', "'original'", '"stop_tokens"',
                            "'stop_tokens'")

    def test_no_forbidden_key_is_subscripted(self):
        for rel in REWARD_SIDE_MODULES:
            src = (REPO / rel).read_text(encoding="utf-8")
            for bad in self.FORBIDDEN_SUBSCRIPTS:
                self.assertNotIn(
                    f"[{bad}]", src,
                    f"{rel}: subscript of forbidden task key {bad}")

    def test_no_reward_module_imports_the_scorer_or_container(self):
        for rel in REWARD_SIDE_MODULES:
            tree = ast.parse((REPO / rel).read_text(encoding="utf-8"))
            imported = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported.update(a.name for a in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported.add(node.module)
            for mod in imported:
                self.assertNotIn(
                    "multipl_e.score", mod,
                    f"{rel}: the steer side must not import the scorer")

    def test_view_module_imports_nothing_but_the_fn_header_regex(self):
        # The allowlist constructor's ONLY dependency is the prompt-side regex
        # already used by the export instrument — no dataset loader, no scorer.
        tree = ast.parse((REPO / "temper/steer/view.py").read_text(
            encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
        self.assertEqual(imported, {"__future__", "dataclasses",
                                    "temper.multipl_e.export"})


if __name__ == "__main__":
    unittest.main()
