//! Human-authored property for bench/sort (parent-authored, APPROVED).
//! Contract with main.rs: `run_property(runner)` entry point; value type
//! Vec<i32> is free (main.rs is generic over TestError<T>).

use crate::candidate::sort_i32;
use proptest::prelude::*;
use proptest::test_runner::{TestCaseError, TestCaseResult, TestError, TestRunner};

pub const STRATEGY_DESC: &str = "vec(any::<i32>(), 0..64)";

fn strategy() -> impl Strategy<Value = Vec<i32>> {
    proptest::collection::vec(any::<i32>(), 0..64)
}

fn property(v: Vec<i32>) -> TestCaseResult {
    let mut got = v.clone();
    sort_i32(&mut got);

    let mut want = v.clone();
    want.sort_unstable();

    if got == want {
        Ok(())
    } else {
        Err(TestCaseError::fail(format!(
            "sort_i32(v) != sorted(v): got {:?}, want {:?}",
            got, want
        )))
    }
}

pub fn run_property(runner: &mut TestRunner) -> Result<(), TestError<Vec<i32>>> {
    runner.run(&strategy(), property)
}
