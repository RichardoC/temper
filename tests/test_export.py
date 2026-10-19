"""Unit tests for temper.multipl_e.export (M2.0 item 2). Stdlib unittest."""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from temper.multipl_e.export import (
    export_runs,
    load_benchmark,
    strip_stop_residue,
    to_body_fragment,
)

REPO = Path(__file__).resolve().parents[1]


def _write_run(path: Path, meta_extra=None, cands=(), task_dir=None):
    meta = {
        "type": "run", "milestone": "M2-L0", "task": "faketask",
        "task_dir": str(task_dir) if task_dir else str(path.parent),
        "K": len(cands), "steps": 8, "canvas": 64, "base_seed": 0,
        "model": "test-model", "utc": "2026-10-19T00:00:00+00:00",
    }
    if meta_extra:
        meta.update(meta_extra)
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(meta) + "\n")
        for c in cands:
            f.write(json.dumps({"type": "candidate", **c}) + "\n")
    return meta


RAW_GOOD = """```rust
pub fn sort_i32(v: &mut Vec<i32>) {
    v.sort_unstable();
}
```
Explanation: sorts in place.
"""

RAW_WRONG_API = """```rust
impl Sorter {
    pub fn sort_i32(&mut self) {}
}
```
"""


class TestStripStopResidue(unittest.TestCase):
    def test_cuts_at_stop_token(self):
        # GOTCHA 1: retained stop token '\n}' duplicates the closing brace
        code = "    let x = 1;\n    x\n}\n}\n"
        out = strip_stop_residue(code, ["\n}"])
        self.assertEqual(out, "    let x = 1;\n    x")

    def test_fence_residue_stripped(self):
        code = "fn f() {}\n```\n"
        self.assertEqual(strip_stop_residue(code), "fn f() {}")

    def test_no_stop_tokens_no_cut(self):
        code = "fn f() {\n    g();\n}\n"
        self.assertEqual(strip_stop_residue(code), "fn f() {\n    g();\n}")

    def test_earliest_stop_token_wins(self):
        code = "a\n}\nb\n}\n"
        self.assertEqual(strip_stop_residue(code, ["\n}"]), "a")


class TestBodyFragment(unittest.TestCase):
    PROMPT = ("/// doc\nfn has_close_elements(numbers: Vec<f64>, "
              "threshold: f64) -> bool {\n")

    def test_single_fn_converted(self):
        module = ("fn has_close_elements(numbers: Vec<f64>, threshold: f64) -> bool {\n"
                  "    numbers.contains(&threshold)\n}")
        body = to_body_fragment(module, self.PROMPT)
        self.assertEqual(body, "numbers.contains(&threshold)")

    def test_helpers_move_into_body_region(self):
        # helper defined after the prompted fn in extraction order -> appears
        # inside the completion (Rust allows nested items in fn bodies)
        module = ("fn has_close_elements(numbers: Vec<f64>, threshold: f64) -> bool {\n"
                  "    let t = thr(threshold);\n"
                  "    numbers.iter().any(|&x| (x - t).abs() < 1.0)\n"
                  "}\n"
                  "fn thr(t: f64) -> f64 { t }")
        body = to_body_fragment(module, self.PROMPT)
        self.assertIn("thr(threshold)", body)
        self.assertTrue(body.startswith("let t ="))

    def test_renamed_fn_rejected(self):
        module = "fn close_enough(numbers: Vec<f64>, threshold: f64) -> bool {\n    true\n}"
        self.assertIsNone(to_body_fragment(module, self.PROMPT))

    def test_reshaped_signature_rejected(self):
        module = ("fn has_close_elements(numbers: Vec<i64>, threshold: f64) -> bool {\n"
                  "    true\n}")
        self.assertIsNone(to_body_fragment(module, self.PROMPT))

    def test_missing_fn_rejected(self):
        self.assertIsNone(to_body_fragment("fn other() {}", self.PROMPT))

    def test_unbalanced_rejected(self):
        module = "fn has_close_elements(numbers: Vec<f64>, threshold: f64) -> bool {\n    true;"
        self.assertIsNone(to_body_fragment(module, self.PROMPT))


class TestExportTemperFlavor(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.task_dir = REPO / "bench" / "sort"  # real task dir: has task.toml
        self.out = self.tmp / "out"

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def _run(self, name, cands, raws=None):
        run = self.tmp / f"{name}.jsonl"
        _write_run(run, meta_extra={"task": "sort", "task_dir": str(self.task_dir)},
                   cands=cands)
        raw_dir = self.tmp / f"{name}.raw"
        raw_dir.mkdir(exist_ok=True)
        for i, txt in (raws or {}).items():
            (raw_dir / f"p{i}.txt").write_text(txt, encoding="utf-8")
        return run

    def test_roundtrip_export_and_manifest(self):
        run = self._run("20261019T000000Z", [
            {"i": 0, "extract_ok": True,
             "raw_path": f"bench/_tmp_test/raw_missing/p0.txt"},  # missing sidecar
            {"i": 1, "extract_ok": True,
             "raw_path": str(self.tmp / "20261019T000000Z.raw" / "p1.txt"),
             "gen_error": "boom"},
        ])
        # fix raw_path for i=0 to an existing file
        rec = [json.loads(l) for l in run.read_text().splitlines()]
        rec[1]["raw_path"] = str(self.tmp / "20261019T000000Z.raw" / "p0.txt")
        run.write_text("".join(json.dumps(r) + "\n" for r in rec))
        (self.tmp / "20261019T000000Z.raw" / "p0.txt").write_text(RAW_GOOD)
        (self.tmp / "20261019T000000Z.raw" / "p1.txt").write_text(RAW_GOOD)

        m = export_runs([run], self.out)
        probs = {p["name"]: p for p in m["problems"]}
        self.assertIn("temper-sort", probs)
        pf = self.out / "temper-sort.json"
        data = json.loads(pf.read_text())
        self.assertEqual(data["language"], "rs")
        self.assertEqual(data["prompt"], "")
        self.assertIn("fn main()", data["tests"])
        self.assertEqual(len(data["completions"]), 1)
        self.assertIn("sort_unstable", data["completions"][0])
        # honest counting: 2 candidate rows -> 1 exported + 1 excluded (gen_error)
        self.assertEqual(m["counts"]["candidate_rows"], 2)
        self.assertEqual(m["counts"]["exported_completions"], 1)
        self.assertEqual(m["counts"]["excluded_rows"], 1)
        self.assertEqual(m["excluded"][0]["reason"], "generation error: boom")

    def test_extract_reject_excluded(self):
        run = self._run("20261019T000100Z", [
            {"i": 0, "extract_ok": False, "extract_reason": "must_define",
             "raw_path": str(self.tmp / "20261019T000100Z.raw" / "p0.txt")},
        ])
        (self.tmp / "20261019T000100Z.raw" / "p0.txt").write_text(RAW_WRONG_API)
        m = export_runs([run], self.out)
        self.assertEqual(m["counts"]["exported_completions"], 0)
        self.assertEqual(m["counts"]["excluded_rows"], 1)
        self.assertIn("extract rejected", m["excluded"][0]["reason"])
        # zero-completion problem file still written (scored failure evidence)
        data = json.loads((self.out / "temper-sort.json").read_text())
        self.assertEqual(data["completions"], [])

    def test_compile_error_row_repaired_at_export(self):
        # record says compile_error (e.g. run-B p0's stray block): the export
        # applies the same L0.5 repair round and exports the repaired code
        stray = ("```rust\npub struct RingBuf { cap: usize }\n"
                 "pub RingBuf { fn new(c: usize) -> RingBuf { RingBuf { cap: c } } }\n"
                 "pub fn sort_i32(v: &mut Vec<i32>) { v.sort_unstable(); }\n```\n")
        run = self._run("20261019T000600Z", [
            {"i": 0, "extract_ok": True, "compile_status": "compile_error",
             "raw_path": str(self.tmp / "20261019T000600Z.raw" / "p0.txt")},
        ])
        (self.tmp / "20261019T000600Z.raw" / "p0.txt").write_text(stray)
        m = export_runs([run], self.out)
        self.assertEqual(m["counts"]["exported_completions"], 1)
        data = json.loads((self.out / "temper-sort.json").read_text())
        self.assertEqual(len(data["completions"]), 1)
        self.assertIn("impl RingBuf {", data["completions"][0])
        self.assertNotIn("pub RingBuf {", data["completions"][0])
        self.assertTrue(m["problems"][0]["source_rows"][0]["repaired_at_export"])

    def test_no_repair_flag_skips_salvage(self):
        # M2.2 ablation column: repair=False exports the model's raw
        # extraction quality — the same row the repair=on path saves is
        # exported UNREPAIRED and the container fails it (honest)
        stray = ("```rust\npub struct RingBuf { cap: usize }\n"
                 "pub RingBuf { fn new(c: usize) -> RingBuf { RingBuf { cap: c } } }\n"
                 "pub fn sort_i32(v: &mut Vec<i32>) { v.sort_unstable(); }\n```\n")
        run = self._run("20261019T000700Z", [
            {"i": 0, "extract_ok": True, "compile_status": "compile_error",
             "raw_path": str(self.tmp / "20261019T000700Z.raw" / "p0.txt")},
        ])
        (self.tmp / "20261019T000700Z.raw" / "p0.txt").write_text(stray)
        m = export_runs([run], self.out, repair=False)
        self.assertEqual(m["counts"]["exported_completions"], 1)
        self.assertEqual(m["counts"]["excluded_rows"], 0)
        self.assertFalse(m["repair"])
        data = json.loads((self.out / "temper-sort.json").read_text())
        self.assertEqual(len(data["completions"]), 1)
        self.assertIn("pub RingBuf {", data["completions"][0])
        self.assertFalse(m["problems"][0]["source_rows"][0]["repaired_at_export"])

    def test_pre_m15_record_excluded(self):
        run = self._run("20261019T000200Z", [{"i": 0, "extract_ok": True}])
        m = export_runs([run], self.out)
        self.assertIn("raw text unavailable", m["excluded"][0]["reason"])
        self.assertEqual(m["counts"]["exported_completions"], 0)

    def test_unsteered_guise_identical(self):
        run = self._run("20261019T000300Z", [
            {"i": 0, "extract_ok": True,
             "raw_path": str(self.tmp / "20261019T000300Z.raw" / "p0.txt")},
        ])
        (self.tmp / "20261019T000300Z.raw" / "p0.txt").write_text(RAW_GOOD)
        m = export_runs([run], self.out, also_unsteered_guise=True)
        names = sorted(p["name"] for p in m["problems"])
        self.assertEqual(names, ["temper-sort", "temper-sort-unsteered-guise"])
        a = json.loads((self.out / "temper-sort.json").read_text())
        b = json.loads((self.out / "temper-sort-unsteered-guise.json").read_text())
        self.assertEqual(a["completions"], b["completions"])
        self.assertEqual(a["prompt"], b["prompt"])
        self.assertEqual(a["tests"], b["tests"])
        self.assertNotEqual(a["name"], b["name"])

    def test_two_runs_merge_into_one_problem(self):
        r1 = self._run("20261019T000400Z", [
            {"i": 0, "extract_ok": True,
             "raw_path": str(self.tmp / "20261019T000400Z.raw" / "p0.txt")}])
        r2 = self._run("20261019T000500Z", [
            {"i": 0, "extract_ok": True,
             "raw_path": str(self.tmp / "20261019T000500Z.raw" / "p0.txt")}])
        (self.tmp / "20261019T000400Z.raw" / "p0.txt").write_text(RAW_GOOD)
        (self.tmp / "20261019T000500Z.raw" / "p0.txt").write_text(RAW_GOOD)
        m = export_runs([r1, r2], self.out)
        self.assertEqual(len(m["problems"]), 1)
        self.assertEqual(m["problems"][0]["completions"], 2)


class TestExportMultiPLEJoin(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.out = self.tmp / "out"

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_join_produces_body_fragment(self):
        bench = load_benchmark("humaneval-rs")
        td = bench["HumanEval_0_has_close_elements"]
        fn_line = [l for l in td["prompt"].splitlines() if l.startswith("fn ")][0]
        module = (fn_line + "\n"
                  "    numbers.iter().any(|&x| (x - threshold).abs() < 1.0)\n"
                  "}")
        run = self.tmp / "r.jsonl"
        _write_run(run, meta_extra={"benchmark": "humaneval-rs"}, cands=[
            {"i": 0, "problem": "HumanEval_0_has_close_elements", "extract_ok": True,
             "raw_path": str(self.tmp / "r.raw" / "p0.txt")},
            {"i": 1, "problem": "HumanEval_999_nope", "extract_ok": True},
        ])
        (self.tmp / "r.raw").mkdir()
        (self.tmp / "r.raw" / "p0.txt").write_text("```rust\n" + module + "\n```")
        m = export_runs([run], self.out, benchmark="humaneval-rs")
        data = json.loads((self.out / "HumanEval_0_has_close_elements.json").read_text())
        self.assertEqual(data["prompt"], td["prompt"])
        self.assertEqual(data["tests"], td["tests"])
        self.assertEqual(data["completions"],
                         ["numbers.iter().any(|&x| (x - threshold).abs() < 1.0)"])
        # unknown problem excluded with reason
        self.assertEqual(m["counts"]["excluded_rows"], 1)
        self.assertIn("not in vendored benchmark", m["excluded"][0]["reason"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
