"""Unit tests for temper.multipl_e.bench (M2.2 Phase 1: MultiPL-E driver).

The load-bearing contract under test (MultiPL-E scorer semantics):
  program = prompt + completion + "\\n" + tests
  * vendored prompts end mid-function: ``... -> T {\\n``
  * vendored tests strings BEGIN with the prompted function's closing ``}``
  * so a completion MUST be the prompted function's body fragment: no
    signature repetition, no fn-closing brace at column 0.
The driver emits whole-function completions; the M2.0 export
(``to_body_fragment``) reduces them to body fragments. These tests pin every
invariant of that handoff, on the REAL vendored data.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from temper.multipl_e.bench import (
    INSTRUCTION_TEMPLATE,
    build_prompt,
    load_problems,
    run_bench,
)
from temper.multipl_e.export import _fn_header_line, export_runs, to_body_fragment

REPO = Path(__file__).resolve().parents[1]


def _fake_particle(i, raw_text, *, error=None, nfe=4, truncated=False):
    return SimpleNamespace(
        index=i, raw_text=raw_text, temperature=0.2, alg="entropy",
        seed=i, prompt_len=10, canvas_len=8, steps=4, nfe=nfe,
        wall_s=0.5, per_step_s=[0.1, 0.1, 0.1, 0.1],
        committed_tokens_total=3, mean_commit_conf=0.9, truncated=truncated,
        error=error,
    )


def _compliant_module(chat_prompt: str, body="\n    unimplemented!()") -> str:
    """What a COMPLIANT model emits: the signature line from the instruction
    prompt, verbatim, plus a body, in one rust fence."""
    sig = None
    for m in _FN_LINE_RE.finditer(chat_prompt):
        sig = m.group(0)
    assert sig is not None, "chat prompt carries no fn signature line"
    return "```rust\n" + sig + body + "\n}\n```"


_FN_LINE_RE = __import__("re").compile(r"^fn [^\n]*\{\s*$", __import__("re").M)


class TestVendoredConcatInvariants(unittest.TestCase):
    """Pin the concatenation contract on the REAL vendored benchmark."""

    @classmethod
    def setUpClass(cls):
        cls.problems = load_problems("humaneval-rs")

    def test_all_problems_end_mid_function(self):
        for td in self.problems:
            self.assertTrue(
                td["prompt"].rstrip().endswith("{"),
                f"{td['name']}: prompt does not end with '{{' (ends "
                f"{td['prompt'][-40:]!r})")

    def test_all_tests_open_with_closing_brace(self):
        for td in self.problems:
            self.assertTrue(td["tests"].startswith("}"),
                            f"{td['name']}: tests do not open with '}}'")

    def test_all_stop_tokens_are_newbrace(self):
        for td in self.problems:
            self.assertEqual(td["stop_tokens"], ["\n}"])

    def test_all_prompts_have_fn_header_and_tests_bind_it(self):
        for td in self.problems:
            m = _fn_header_line(td["prompt"])
            self.assertIsNotNone(m, f"{td['name']}: no fn header in prompt")
            self.assertIn(f"let candidate = {m.group(1)};", td["tests"],
                          f"{td['name']}: tests do not bind the prompted fn")

    def test_body_fragment_concat_balances_braces(self):
        # for the first 3 problems: a compliant whole-module completion,
        # reduced to a body fragment, makes prompt+completion+'\n'+tests a
        # brace-balanced program whose completion re-states no signature
        # and carries no fn-closing brace (tests supply it).
        for td in self.problems[:3]:
            sig = _fn_header_line(td["prompt"]).group(0)
            module = sig + "\n    unimplemented!()\n}"
            body = to_body_fragment(module, td["prompt"])
            self.assertIsNotNone(body, td["name"])
            self.assertNotIn(sig, body)
            for ln in body.splitlines():
                self.assertFalse(ln.startswith("}"),
                                 f"{td['name']}: body line {ln!r} would "
                                 "close the prompted fn early")
            program = td["prompt"] + body + "\n" + td["tests"]
            self.assertEqual(program.count("{"), program.count("}"),
                             f"{td['name']}: program braces unbalanced")

    def test_stop_strip_keeps_body_intact(self):
        # GOTCHA 1 machinery must not truncate an indented body: stop token
        # '\n}' never occurs in a properly-indented body fragment
        td = self.problems[0]
        from temper.multipl_e.export import strip_stop_residue
        body = "    let a = 1;\n    a > 0"
        self.assertEqual(strip_stop_residue(body, td["stop_tokens"]), body)


class TestPromptBuilder(unittest.TestCase):
    def test_format_survives_rust_braces(self):
        p = build_prompt("/// doc\nfn f(x: i32) -> bool {\n")
        self.assertIn("/// doc\nfn f(x: i32) -> bool {\n", p)

    def test_instruction_is_format_only(self):
        p = build_prompt("fn f() -> bool {\n")
        self.assertIn("ONE ```rust", p)
        self.assertIn("no tests", p)
        self.assertIn("repeat the signature line exactly", p)
        self.assertIn("INSIDE the function body", p)

    def test_template_has_single_placeholder(self):
        self.assertEqual(INSTRUCTION_TEMPLATE.count("{prompt}"), 1)


class TestDriverWithFakeSampler(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.out = self.tmp / "run.jsonl"

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def _fake_sample(self, prompt, *, canvas_len, steps, particles,
                     temperatures, algs, base_seed, progress=None):
        # a compliant model: copy the signature from the instruction prompt
        raw = _compliant_module(prompt)
        out = []
        for i in range(particles):
            p = _fake_particle(i, raw)
            p.temperature = temperatures[i]
            p.alg = algs[i]
            p.seed = base_seed + i
            out.append(p)
            if progress:
                progress(i, p)
        return out

    def test_record_structure_and_lineage(self):
        res = run_bench(
            "humaneval-rs", n_problems=2, k=2, steps=4, canvas=8,
            base_seed=7, temperature=0.2, alg="entropy", out=self.out,
            sample_fn=self._fake_sample, log=lambda s: None,
        )
        lines = [json.loads(l) for l in
                 self.out.read_text(encoding="utf-8").splitlines() if l.strip()]
        meta, cands, dones = lines[0], lines[1:-1], lines[-1]
        self.assertEqual(meta["type"], "run")
        self.assertEqual(meta["benchmark"], "humaneval-rs")
        self.assertEqual(meta["milestone"], "M2.2-B2")
        self.assertEqual(meta["mode"], "unsteered")
        self.assertEqual(meta["K"], 2)
        self.assertEqual(meta["problem_names"],
                         ["HumanEval_0_has_close_elements",
                          "HumanEval_1_separate_paren_groups"])
        self.assertEqual(len(cands), 4)
        # stable order: problem order = vendored file order, samples 0..k-1
        self.assertEqual([(c["problem_index"], c["i"]) for c in cands],
                         [(0, 0), (0, 1), (1, 0), (1, 1)])
        # seed lineage: sample i of every problem uses base_seed + i
        self.assertEqual([c["seed"] for c in cands], [7, 8, 7, 8])
        self.assertEqual([c["temperature"] for c in cands], [0.2] * 4)
        self.assertEqual([c["alg"] for c in cands], ["entropy"] * 4)
        self.assertEqual([c["nfe"] for c in cands], [4] * 4)
        self.assertEqual([c["prompted_fn"] for c in cands],
                         ["has_close_elements", "has_close_elements",
                          "separate_paren_groups", "separate_paren_groups"])
        # raw sidecars exist, sha-referenced
        import hashlib
        tds = load_problems("humaneval-rs", 2)
        for c in cands:
            td = tds[c["problem_index"]]
            sidecar = Path(c["raw_path"])
            if not sidecar.is_absolute():
                sidecar = REPO / sidecar
            self.assertTrue(sidecar.exists(), sidecar)
            self.assertEqual(
                hashlib.sha256(sidecar.read_bytes()).hexdigest(),
                c["raw_sha256"])
            self.assertEqual(c["prompt_sha256"],
                             hashlib.sha256(td["prompt"].encode()).hexdigest())
        self.assertTrue((self.out.parent / (self.out.name + ".raw") /
                         "HumanEval_0_has_close_elements__prompt.txt").exists())
        # done summary line
        self.assertEqual(dones["type"], "done")
        self.assertEqual(dones["n_rows"], 4)
        self.assertEqual(dones["nfe_total"], 16)
        self.assertEqual(dones["extract_ok_rows"], 4)
        self.assertEqual(res["summary"]["n_rows"], 4)

    def test_export_body_fragment_with_toplevel_block_not_stopcut(self):
        # M2.4b strip-artifact regression: a body whose prompted fn contains a
        # TOP-LEVEL block dedents its close to col 0 — the `\n}` stop-token
        # cut (GOTCHA 1 machinery) must NOT amputate the fragment there.
        # Found on the B3 Qwen cell (65/100 rows) and present in every
        # committed dLLM cell (64-76 SyntaxErrors per cell); see
        # scripts/m2_4b_strip_artifact.py + docs/m2.4b-report.md.
        run_bench(
            "humaneval-rs", n_problems=1, k=1, steps=4, canvas=8,
            out=self.out, sample_fn=self._fake_sample, log=lambda s: None,
        )
        # overwrite the fake's simple body with a top-level-for body
        # (HumanEval_0's real shape: `for` at fn-body depth)
        from temper.multipl_e.bench import load_problems
        from temper.multipl_e.export import _fn_header_line
        td = load_problems("humaneval-rs", 1)[0]
        sig = _fn_header_line(td["prompt"]).group(0)
        raw_dir = self.out.parent / (self.out.name + ".raw")
        (raw_dir / f"{td['name']}__p0.txt").write_text(
            "```rust\n" + sig + "\n    for i in 0..10 {\n        true;\n    }\n    false\n}\n```",
            encoding="utf-8")
        outdir = self.tmp / "export2"
        export_runs([self.out], outdir, benchmark="humaneval-rs")
        comp = json.loads(
            (outdir / f"{td['name']}.json").read_text())["completions"][0]
        # body dedented by min indent (4): the top-level for lands at col 0
        self.assertEqual(
            comp,
            "for i in 0..10 {\n    true;\n}\nfalse")
        self.assertEqual(comp.count("{"), comp.count("}"),
                         "top-level block close must survive the export")

    def test_export_roundtrip_produces_official_files(self):
        run_bench(
            "humaneval-rs", n_problems=2, k=2, steps=4, canvas=8,
            out=self.out, sample_fn=self._fake_sample, log=lambda s: None,
        )
        outdir = self.tmp / "export"
        m = export_runs([self.out], outdir, benchmark="humaneval-rs")
        self.assertEqual(m["counts"]["candidate_rows"], 4)
        self.assertEqual(m["counts"]["exported_completions"], 4)
        self.assertEqual(m["counts"]["excluded_rows"], 0)
        for td in load_problems("humaneval-rs", 2):
            pf = outdir / f"{td['name']}.json"
            data = json.loads(pf.read_text())
            self.assertEqual(data["prompt"], td["prompt"])
            self.assertEqual(data["tests"], td["tests"])
            self.assertEqual(data["language"], "rs")
            self.assertEqual(len(data["completions"]), 2)
            for comp in data["completions"]:
                self.assertNotIn("fn ", comp)  # body fragment, no signature
                program = td["prompt"] + comp + "\n" + td["tests"]
                self.assertEqual(program.count("{"), program.count("}"))


class TestDriverFailurePaths(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.out = self.tmp / "run.jsonl"

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def _fake_sample_mixed(self, prompt, *, particles, **kw):
        good = _compliant_module(prompt)
        return [
            _fake_particle(0, good),
            _fake_particle(1, "```rust\n<|mask|>\n```"),           # mask residue
            _fake_particle(2, "", error="RuntimeError: MPS boom"),  # gen error
            _fake_particle(3, "```rust\nfn renamed() -> bool {\n    true\n}\n```"),
        ]

    def test_failures_recorded_honestly(self):
        res = run_bench(
            "humaneval-rs", n_problems=1, k=4, steps=4, canvas=8,
            out=self.out, sample_fn=self._fake_sample_mixed,
            log=lambda s: None,
        )
        cands = [json.loads(l) for l in
                 self.out.read_text(encoding="utf-8").splitlines()
                 if json.loads(l)["type"] == "candidate"]
        self.assertEqual(len(cands), 4)
        self.assertTrue(cands[0]["extract_ok"])
        self.assertFalse(cands[1]["extract_ok"])
        self.assertIn("incomplete canvas", cands[1]["extract_reason"])
        self.assertEqual(cands[2]["gen_error"], "RuntimeError: MPS boom")
        self.assertIn("generation error", cands[2]["extract_reason"])
        self.assertFalse(cands[3]["extract_ok"])
        self.assertIn("has_close_elements", cands[3]["extract_reason"])
        # honest counting end to end: 1 exported, 3 excluded with reasons
        outdir = self.tmp / "export"
        m = export_runs([self.out], outdir, benchmark="humaneval-rs")
        self.assertEqual(m["counts"]["exported_completions"], 1)
        self.assertEqual(m["counts"]["excluded_rows"], 3)
        self.assertEqual(m["counts"]["candidate_rows"], 4)

    def test_overwrite_refused_without_resume(self):
        run_bench("humaneval-rs", n_problems=1, k=1, steps=4, canvas=8,
                  out=self.out, sample_fn=lambda *a, **kw: [], log=lambda s: None)
        with self.assertRaises(SystemExit):
            run_bench("humaneval-rs", n_problems=1, k=1, steps=4, canvas=8,
                      out=self.out, sample_fn=lambda *a, **kw: [],
                      log=lambda s: None)


class TestResume(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.out = self.tmp / "run.jsonl"

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def _fake_sample(self, prompt, *, particles, base_seed, temperatures,
                     algs, **kw):
        raw = _compliant_module(prompt)
        return [_fake_particle(i, raw, nfe=4) for i in range(particles)]

    def test_resume_skips_completed_pairs(self):
        run_bench("humaneval-rs", n_problems=2, k=2, steps=4, canvas=8,
                  base_seed=3, out=self.out, sample_fn=self._fake_sample,
                  log=lambda s: None)
        calls = []

        def counting_sample(prompt, **kw):
            calls.append(prompt)
            return self._fake_sample(prompt, **kw)

        run_bench("humaneval-rs", n_problems=2, k=2, steps=4, canvas=8,
                  base_seed=3, out=self.out, resume=True,
                  sample_fn=counting_sample, log=lambda s: None)
        self.assertEqual(calls, [])  # nothing left to sample
        cands = [json.loads(l) for l in
                 self.out.read_text(encoding="utf-8").splitlines()
                 if json.loads(l)["type"] == "candidate"]
        self.assertEqual(len(cands), 4)  # rows unchanged

    def test_resume_config_mismatch_refused(self):
        run_bench("humaneval-rs", n_problems=2, k=2, steps=4, canvas=8,
                  out=self.out, sample_fn=self._fake_sample,
                  log=lambda s: None)
        with self.assertRaises(SystemExit):
            run_bench("humaneval-rs", n_problems=2, k=3, steps=4, canvas=8,
                      out=self.out, resume=True, sample_fn=self._fake_sample,
                      log=lambda s: None)
        with self.assertRaises(SystemExit):
            run_bench("mbpp-rs", n_problems=2, k=2, steps=4, canvas=8,
                      out=self.out, resume=True, sample_fn=self._fake_sample,
                      log=lambda s: None)

    def test_resume_without_record_refused(self):
        with self.assertRaises(SystemExit):
            run_bench("humaneval-rs", n_problems=1, k=1, steps=4, canvas=8,
                      out=self.out, resume=True, sample_fn=self._fake_sample,
                      log=lambda s: None)


class TestLoadProblems(unittest.TestCase):
    def test_stable_order_and_slice(self):
        full = load_problems("humaneval-rs")
        self.assertEqual(len(full), 156)
        self.assertEqual([t["name"] for t in full[:3]],
                         ["HumanEval_0_has_close_elements",
                          "HumanEval_1_separate_paren_groups",
                          "HumanEval_2_truncate_number"])
        self.assertEqual(load_problems("humaneval-rs", 5), full[:5])

    def test_n_too_large_refused(self):
        with self.assertRaises(ValueError):
            load_problems("humaneval-rs", 157)


if __name__ == "__main__":
    unittest.main(verbosity=2)
