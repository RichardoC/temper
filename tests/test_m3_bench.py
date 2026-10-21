"""M3 driver extensions: problem cohorts, per-canvas temperatures, hint wrap.

These are the mechanisms P0.5 (C-hint quartet) and P1 (9-problem x 3-temp
probe) need, and they must not change what the bench driver already does for
the committed M2 cells. Pinned here:

  * ``--problem-indices`` selects an explicit cohort, and a conflicting
    ``--problems N`` is refused rather than silently resolved;
  * a cohort cell records the vendored indices, so the run record says which
    problems it actually ran;
  * ``temperatures`` gives each problem len(temps) canvases, one per temp, with
    per-row temperature + seed lineage — the grid the reachability curve needs;
  * ``hint_line`` appends exactly ONE line and pins its exact text + sha256 in
    the run meta, so the knowledge-injection null is reproducible.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from temper.multipl_e.bench import INSTRUCTION_TEMPLATE, load_problems, run_bench

REPO = Path(__file__).resolve().parents[1]


class _FakeParticle:
    def __init__(self, i, temperature=0.2):
        self.index = i
        self.raw_text = "```rust\nfn f() -> bool {\n    true\n}\n```"
        self.temperature = temperature
        self.alg = "entropy"
        self.seed = i
        self.prompt_len = 4
        self.canvas_len = 8
        self.steps = 4
        self.nfe = 4
        self.wall_s = 0.1
        self.per_step_s = [0.025] * 4
        self.committed_tokens_total = 1
        self.mean_commit_conf = 0.5
        self.truncated = False
        self.error = None


def _fake_sample(prompt, *, particles, temperatures, algs, base_seed,
                 progress=None, **kw):
    out = []
    for i in range(particles):
        p = _FakeParticle(base_seed + i if False else i,
                          temperature=temperatures[i])
        p.seed = base_seed + i
        out.append(p)
        if progress:
            progress(i, p)
    return out


class TestProblemIndices(unittest.TestCase):
    def test_cohort_selection_is_explicit_and_ordered(self):
        picked = load_problems("humaneval-rs", problem_indices=[0, 4, 20, 21])
        self.assertEqual([td["name"] for td in picked],
                         ["HumanEval_0_has_close_elements",
                          "HumanEval_4_mean_absolute_deviation",
                          "HumanEval_20_find_closest_elements",
                          "HumanEval_21_rescale_to_unit"])

    def test_duplicate_indices_collapse(self):
        picked = load_problems("humaneval-rs", problem_indices=[4, 0, 4])
        self.assertEqual(len(picked), 2)

    def test_out_of_range_index_refused(self):
        with self.assertRaises(ValueError):
            load_problems("humaneval-rs", problem_indices=[9999])

    def test_conflicting_problem_count_refused(self):
        with self.assertRaises(ValueError):
            load_problems("humaneval-rs", n=9, problem_indices=[0, 4])

    def test_matching_problem_count_is_accepted(self):
        picked = load_problems("humaneval-rs", n=4, problem_indices=[0, 4, 20, 21])
        self.assertEqual(len(picked), 4)


class TestCohortRunRecord(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.out = self.tmp / "run.jsonl"

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_cohort_cell_records_vendored_indices(self):
        run_bench("humaneval-rs", n_problems=None, k=1, steps=4, canvas=8,
                  out=self.out, sample_fn=_fake_sample,
                  problem_indices=[0, 20], log=lambda s: None)
        lines = [json.loads(l) for l in
                 self.out.read_text(encoding="utf-8").splitlines() if l.strip()]
        meta = lines[0]
        self.assertEqual(meta["problem_indices"], [0, 20])
        self.assertEqual(meta["problem_names"],
                         ["HumanEval_0_has_close_elements",
                          "HumanEval_20_find_closest_elements"])
        cands = [o for o in lines if o["type"] == "candidate"]
        self.assertEqual([c["problem_vendored_index"] for c in cands], [0, 20])

    def test_mismatched_cohort_resume_is_refused(self):
        run_bench("humaneval-rs", n_problems=None, k=1, steps=4, canvas=8,
                  out=self.out, sample_fn=_fake_sample,
                  problem_indices=[0, 20], log=lambda s: None)
        with self.assertRaises(SystemExit):
            run_bench("humaneval-rs", n_problems=None, k=1, steps=4, canvas=8,
                      out=self.out, sample_fn=_fake_sample, resume=True,
                      problem_indices=[0, 21], log=lambda s: None)


class TestPerCanvasTemperatures(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.out = self.tmp / "run.jsonl"

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_one_canvas_per_temperature_with_seed_lineage(self):
        run_bench("humaneval-rs", n_problems=None, k=4, steps=4, canvas=8,
                  base_seed=3, out=self.out, sample_fn=_fake_sample,
                  problem_indices=[0], temperatures=[0.4, 0.8, 1.2],
                  log=lambda s: None)
        cands = [json.loads(l) for l in
                 self.out.read_text(encoding="utf-8").splitlines()
                 if json.loads(l)["type"] == "candidate"]
        self.assertEqual(len(cands), 3, "one canvas per temperature")
        self.assertEqual([c["temperature"] for c in cands], [0.4, 0.8, 1.2])
        self.assertEqual([c["i"] for c in cands], [0, 1, 2])
        self.assertEqual([c["seed"] for c in cands], [3, 4, 5])
        meta = json.loads(self.out.read_text(encoding="utf-8").splitlines()[0])
        self.assertEqual(meta["temperatures"], [0.4, 0.8, 1.2])
        self.assertEqual(meta["K_effective"], 3)

    def test_temperature_grid_resume_is_refused_on_mismatch(self):
        run_bench("humaneval-rs", n_problems=None, k=3, steps=4, canvas=8,
                  out=self.out, sample_fn=_fake_sample, problem_indices=[0],
                  temperatures=[0.4, 0.8, 1.2], log=lambda s: None)
        with self.assertRaises(SystemExit):
            run_bench("humaneval-rs", n_problems=None, k=3, steps=4, canvas=8,
                      out=self.out, sample_fn=_fake_sample, resume=True,
                      problem_indices=[0], temperatures=[0.4, 0.8],
                      log=lambda s: None)


class TestHintLine(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.out = self.tmp / "run.jsonl"
        self.hint = "For `Vec<f64>` sorting use `sort_by(|a, b| a.partial_cmp(b).unwrap())`; prefer explicit type annotations."

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_hint_is_appended_as_one_line_and_pinned(self):
        import hashlib
        run_bench("humaneval-rs", n_problems=1, k=1, steps=4, canvas=8,
                  out=self.out, sample_fn=_fake_sample, hint_line=self.hint,
                  log=lambda s: None)
        meta = json.loads(self.out.read_text(encoding="utf-8").splitlines()[0])
        self.assertEqual(meta["hint_text"], self.hint)
        self.assertEqual(meta["hint_sha256"],
                         hashlib.sha256(self.hint.encode()).hexdigest())
        # the delivered prompt is the unsteered template + exactly one added line
        prompt_file = (self.tmp / (self.out.name + ".raw") /
                       "HumanEval_0_has_close_elements__prompt.txt")
        delivered = prompt_file.read_text(encoding="utf-8")
        base = INSTRUCTION_TEMPLATE.format(
            prompt=load_problems("humaneval-rs", 1)[0]["prompt"])
        self.assertEqual(delivered, base + "\n" + self.hint)
        self.assertEqual(delivered.count(self.hint), 1)

    def test_no_hint_means_no_hint_fields(self):
        run_bench("humaneval-rs", n_problems=1, k=1, steps=4, canvas=8,
                  out=self.out, sample_fn=_fake_sample, log=lambda s: None)
        meta = json.loads(self.out.read_text(encoding="utf-8").splitlines()[0])
        self.assertNotIn("hint_text", meta)
        self.assertNotIn("hint_sha256", meta)

    def test_hint_mismatch_blocks_resume(self):
        run_bench("humaneval-rs", n_problems=1, k=1, steps=4, canvas=8,
                  out=self.out, sample_fn=_fake_sample, hint_line=self.hint,
                  log=lambda s: None)
        with self.assertRaises(SystemExit):
            run_bench("humaneval-rs", n_problems=1, k=1, steps=4, canvas=8,
                      out=self.out, sample_fn=_fake_sample, resume=True,
                      hint_line=self.hint + " (mutated)", log=lambda s: None)


if __name__ == "__main__":
    unittest.main()


class TestProblemNumbersVsIndices(unittest.TestCase):
    """The M3 cohort tables name problems by HumanEval NUMBER; the vendored
    file's POSITION diverges after 21 (HumanEval_22 is absent). Selecting by
    position would silently run the wrong problem — pinned here."""

    def test_position_and_number_diverge_past_21(self):
        by_pos = load_problems("humaneval-rs", problem_indices=[23])
        by_num = load_problems("humaneval-rs", problem_numbers=[24])
        self.assertEqual(by_pos[0]["name"], "HumanEval_24_largest_divisor")
        self.assertEqual(by_num[0]["name"], "HumanEval_24_largest_divisor")
        # ...and position 24 is a DIFFERENT problem from number 24
        other = load_problems("humaneval-rs", problem_indices=[24])
        self.assertEqual(other[0]["name"], "HumanEval_25_factorize")
        self.assertNotEqual(other[0]["name"], by_num[0]["name"])

    def test_number_selection_resolves_the_design_cohorts(self):
        quartet = load_problems("humaneval-rs", problem_numbers=[0, 4, 20, 21])
        self.assertEqual([t["name"] for t in quartet],
                         ["HumanEval_0_has_close_elements",
                          "HumanEval_4_mean_absolute_deviation",
                          "HumanEval_20_find_closest_elements",
                          "HumanEval_21_rescale_to_unit"])
        probe = load_problems(
            "humaneval-rs", problem_numbers=[0, 4, 20, 21, 11, 18, 24, 17, 19])
        self.assertEqual([t["name"] for t in probe], [
            "HumanEval_0_has_close_elements",
            "HumanEval_4_mean_absolute_deviation",
            "HumanEval_20_find_closest_elements",
            "HumanEval_21_rescale_to_unit",
            "HumanEval_11_string_xor",
            "HumanEval_18_how_many_times",
            "HumanEval_24_largest_divisor",
            "HumanEval_17_parse_music",
            "HumanEval_19_sort_numbers",
        ])

    def test_absent_number_is_refused_loudly(self):
        with self.assertRaises(ValueError) as ctx:
            load_problems("humaneval-rs", problem_numbers=[22])
        self.assertIn("HumanEval_22", str(ctx.exception))

    def test_both_selectors_at_once_is_refused(self):
        with self.assertRaises(ValueError):
            load_problems("humaneval-rs", problem_indices=[0],
                          problem_numbers=[0])

    def test_cohort_run_records_numbers_and_names(self):
        import tempfile, shutil
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp)
        out = tmp / "r.jsonl"
        run_bench("humaneval-rs", n_problems=None, k=1, steps=4, canvas=8,
                  out=out, sample_fn=_fake_sample, problem_numbers=[24, 17],
                  log=lambda s: None)
        meta = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
        self.assertEqual(meta["problem_numbers"], [24, 17])
        self.assertEqual(meta["problem_names"],
                         ["HumanEval_24_largest_divisor",
                          "HumanEval_17_parse_music"])
