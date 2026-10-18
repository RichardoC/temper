//! Human-authored property for bench/wrap-sum (parent-authored, APPROVED).
//! Value type Vec<u64>; the oracle is u64 wrapping arithmetic itself.

use crate::candidate::wrapping_sum;
use proptest::prelude::*;
use proptest::test_runner::{TestCaseError, TestCaseResult, TestError, TestRunner};

pub const STRATEGY_DESC: &str = "vec(u64::arbitrary(), 0..16)";

fn strategy() -> impl Strategy<Value = Vec<u64>> {
    proptest::collection::vec(any::<u64>(), 0..16)
}

fn property(xs: Vec<u64>) -> TestCaseResult {
    let want = xs.iter().fold(0u64, |acc, &x| acc.wrapping_add(x));
    let got = wrapping_sum(&xs);
    if got == want {
        Ok(())
    } else {
        Err(TestCaseError::fail(format!(
            "wrapping_sum({xs:?}) = {got}, want {want}"
        )))
    }
}

pub fn run_property(runner: &mut TestRunner) -> Result<(), TestError<Vec<u64>>> {
    runner.run(&strategy(), property)
}
