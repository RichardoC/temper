//! Property definitions for the generated crate.
//!
//! PLACEHOLDER: `temper.harness.emit` overwrites this file with a property
//! exercising the emitted candidate. `main.rs` (the temper result protocol
//! harness) knows ONLY the stable entry point below and the value type
//! `Vec<i32>`; strategies, generators, and assertions never leak into
//! `main.rs`.

use crate::candidate::candidate_sort;
use proptest::prelude::*;
use proptest::test_runner::{TestCaseError, TestCaseResult, TestError, TestRunner};

/// Human-readable description of the value strategy used by
/// [`run_property`]. Informational only (echoed in harness diagnostics).
#[allow(dead_code)]
pub const STRATEGY_DESC: &str = "vec(any::<i32>(), 0..64)";

/// The value strategy: `Vec<i32>` of length 0..64.
fn strategy() -> impl Strategy<Value = Vec<i32>> {
    proptest::collection::vec(any::<i32>(), 0..64)
}

/// The property: `candidate_sort(v)` must reorder `v` into sorted order
/// without changing its multiset of elements.
fn property(v: Vec<i32>) -> TestCaseResult {
    let mut got = v.clone();
    candidate_sort(&mut got);

    let mut want = v.clone();
    want.sort_unstable();

    if got == want {
        Ok(())
    } else {
        Err(TestCaseError::fail(format!(
            "candidate_sort(v) != sorted(v): got {:?}, want {:?}",
            got, want
        )))
    }
}

/// Stable entry point called by `main.rs`: run this crate's property under
/// the given runner. All strategy/property detail stays behind this line.
pub fn run_property(runner: &mut TestRunner) -> Result<(), TestError<Vec<i32>>> {
    runner.run(&strategy(), property)
}
