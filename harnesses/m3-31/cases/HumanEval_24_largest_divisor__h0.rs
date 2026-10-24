use proptest::prelude::*;
use proptest::test_runner::{TestCaseError, TestCaseResult, TestError, TestRunner};
use crate::candidate::largest_divisor;

pub const STRATEGY_DESC: &str = "integers n >= 2; the result divides n, is smaller than n, and is the largest such";
fn strategy() -> impl Strategy<Value = isize> {
    2isize..2000
}

fn property(n: isize) -> TestCaseResult {
    let got = largest_divisor(n);
    if got >= n {
        return Err(TestCaseError::fail(format!(
            "largest_divisor({}) = {} is not smaller than n", n, got)));
    }
    if got <= 0 || n % got != 0 {
        return Err(TestCaseError::fail(format!(
            "largest_divisor({}) = {} does not divide n", n, got)));
    }
    for d in (got + 1)..n {
        if n % d == 0 {
            return Err(TestCaseError::fail(format!(
                "largest_divisor({}) = {} but {} also divides n", n, got, d)));
        }
    }
    Ok(())
}

pub fn run_property(runner: &mut TestRunner) -> Result<(), TestError<isize>> {
    runner.run(&strategy(), property)
}
