use proptest::prelude::*;
use proptest::test_runner::{TestCaseError, TestCaseResult, TestError, TestRunner};
use crate::candidate::greatest_common_divisor;

pub const STRATEGY_DESC: &str = "positive integers; the result divides both and is the greatest such number";
fn strategy() -> impl Strategy<Value = (isize, isize)> {
    (1isize..500, 1isize..500)
}

fn property(input: (isize, isize)) -> TestCaseResult {
    let (a, b) = input;
    let got = greatest_common_divisor(a, b);
    if got <= 0 {
        return Err(TestCaseError::fail(format!(
            "gcd({}, {}) = {} is not positive", a, b, got)));
    }
    if a % got != 0 || b % got != 0 {
        return Err(TestCaseError::fail(format!(
            "gcd({}, {}) = {} does not divide both", a, b, got)));
    }
    // maximality: no larger divisor of both exists
    for d in (got + 1)..=(a.min(b)) {
        if a % d == 0 && b % d == 0 {
            return Err(TestCaseError::fail(format!(
                "gcd({}, {}) = {} but {} divides both", a, b, got, d)));
        }
    }
    Ok(())
}

pub fn run_property(runner: &mut TestRunner) -> Result<(), TestError<(isize, isize)>> {
    runner.run(&strategy(), property)
}
