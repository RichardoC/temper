#!/usr/bin/env python3
"""M0 acceptance demo: "buggy sort" scenario, fully sandboxed (seatbelt).

Scenario: a diffusion-LM-style candidate emits a sort with an index bug.
temper verifies it with a proptest property and produces structured,
seed-exact feedback. Four steps, every cargo invocation and the harness
binary itself run inside the seatbelt sandbox:

  1. buggy candidate   -> protocol JSON `fail` with a persisted seed + the
                          shrunk minimal input.
  2. re-run with TEMPER_SEED_HEX=<that seed> -> EXACTLY the same minimal
                          input (seed-exact replay).
  3. fixed candidate   -> `pass`.
  4. broken candidate (type error) -> structured diagnostics with error code
                          + byte spans, status compile_error, run NOT attempted.

Exit code 0 only if all four steps pass. Prints PASS/FAIL per step.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from temper.feedback.diagnostics import parse_cargo_messages  # noqa: E402
from temper.harness.runner import run_candidate  # noqa: E402
from temper.harness.sandbox import select_backend  # noqa: E402

# The property under test: candidate_sort(v) == std sorted v, over Vec<i32>
# with length 0..64. Identical text to the template placeholder property, so
# the strategy (and therefore seed replay) is stable across candidates.
PROPERTIES = (Path(__file__).resolve().parents[1] / "rust-template" / "src" / "properties.rs").read_text()

# Candidate with the classic off-by-one: scans j in i+1..=len and indexes
# v[j] -> panics (index out of bounds) as soon as any case reaches the scan
# at i == len-1 ... i.e. for ANY non-empty vector, incl. shrunk len-1.
BUGGY = """\
//! Candidate: selection sort (buggy — off-by-one scan bound).
pub fn candidate_sort(v: &mut Vec<i32>) {
    let len = v.len();
    for i in 0..len {
        let mut min = i;
        for j in i + 1..=len {
            if v[j] < v[min] {
                min = j;
            }
        }
        v.swap(i, min);
    }
}
"""

# Repaired candidate: scan bound fixed to exclusive len.
FIXED = """\
//! Candidate: selection sort (repaired — exclusive scan bound).
pub fn candidate_sort(v: &mut Vec<i32>) {
    let len = v.len();
    for i in 0..len {
        let mut min = i;
        for j in i + 1..len {
            if v[j] < v[min] {
                min = j;
            }
        }
        v.swap(i, min);
    }
}
"""

# Deliberate type error: assigning usize to String (E0308).
BROKEN = """\
//! Candidate: deliberate type error for diagnostics-path verification.
pub fn candidate_sort(v: &mut Vec<i32>) {
    let n: String = v.len();
    for i in 0..n {
        v.push(i);
    }
}
"""


def banner(step: int, title: str) -> None:
    print(f"\n=== step {step}: {title} ===", flush=True)


def main() -> int:
    backend = select_backend()  # seatbelt on macOS unless TEMPER_SANDBOX says otherwise
    print(f"temper M0 acceptance demo — sandbox backend: {backend}")
    failures: list[str] = []

    # ---- step 1: buggy candidate must FAIL with seed + minimal input -------
    banner(1, "buggy candidate -> fail + reel seed + minimal_input (sandboxed)")
    r1 = run_candidate(BUGGY, PROPERTIES, sandbox=backend)
    if r1.test:
        print("raw protocol line:", r1.test.report)
    ok1 = (
        r1.test is not None
        and r1.test.status == "fail"
        and bool(r1.test.seed)
        and bool(r1.test.minimal_input)
    )
    print(f"STEP 1 {'PASS' if ok1 else 'FAIL'}: status={r1.status} "
          f"seed={r1.test.seed if r1.test else '-'} "
          f"minimal_input={r1.test.minimal_input if r1.test else '-'}")
    if not ok1:
        failures.append("step 1: buggy candidate did not fail with seed+minimal_input")

    # ---- step 2: seed-exact replay reproduces the same minimal input -------
    banner(2, "re-run with TEMPER_SEED_HEX=<seed> -> same minimal_input")
    seed = r1.test.seed if r1.test else ""
    r2 = run_candidate(BUGGY, PROPERTIES, sandbox=backend, seed_hex=seed)
    ok2 = (
        r2.test is not None
        and r2.test.status == "fail"
        and r2.test.minimal_input == (r1.test.minimal_input if r1.test else None)
    )
    print(f"STEP 2 {'PASS' if ok2 else 'FAIL'}: replay minimal_input="
          f"{r2.test.minimal_input if r2.test else '-'} "
          f"(expected {r1.test.minimal_input if r1.test else '-'})")
    if not ok2:
        failures.append("step 2: seed-exact replay did not reproduce minimal_input")

    # ---- step 3: fixed candidate must PASS ---------------------------------
    banner(3, "fixed candidate -> pass (sandboxed)")
    r3 = run_candidate(FIXED, PROPERTIES, sandbox=backend)
    ok3 = r3.test is not None and r3.test.status == "pass"
    print(f"STEP 3 {'PASS' if ok3 else 'FAIL'}: status={r3.status} "
          f"report={r3.test.report if r3.test else r3.compile.stderr[-300:]}")
    if not ok3:
        failures.append("step 3: fixed candidate did not pass")

    # ---- step 4: type error -> structured diagnostics, run not attempted ---
    banner(4, "broken candidate -> structured compile diagnostics (run skipped)")
    r4 = run_candidate(BROKEN, PROPERTIES, sandbox=backend)
    errs = [d for d in r4.compile.diagnostics if d.is_error]
    ok4 = (
        r4.compile.status == "compile_error"
        and r4.test is None
        and len(errs) > 0
        and any(d.code for d in errs)
        and any(d.spans for d in errs)
    )
    if errs:
        d = errs[0]
        print(f"first error: code={d.code} message={d.message!r}")
        print(f"  spans: {[(s.file_name, s.byte_start, s.byte_end, s.is_primary) for s in d.spans][:3]}")
    print(f"STEP 4 {'PASS' if ok4 else 'FAIL'}: compile_status={r4.compile.status} "
          f"errors={len(errs)} run_attempted={r4.test is not None}")
    if not ok4:
        failures.append("step 4: broken candidate did not produce structured compile_error diagnostics")

    print("\n" + ("M0 DEMO: ALL STEPS PASS" if not failures else "M0 DEMO: FAILURES: " + "; ".join(failures)))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
