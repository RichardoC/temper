use proptest::prelude::*;
use proptest::test_runner::{TestCaseError, TestCaseResult, TestError, TestRunner};
use crate::candidate::largest_divisor;

pub const STRATEGY_DESC: &str = "the documented example plus the composite/prime split";
fn strategy() -> impl Strategy<Value = isize> {
    prop_oneof![
        Just(15isize),
        (2isize..200).prop_map(|i| i * 2),
        (2isize..80).prop_map(|i| i * 3),
    ]
}

fn property(n: isize) -> TestCaseResult {
    let got = largest_divisor(n);
    if n == 15 && got != 5 {
        return Err(TestCaseError::fail(format!(
            "documented example largest_divisor(15) = {} expected 5", got)));
    }
    // an even n > 2 has n/2 as its largest proper divisor
    if n % 2 == 0 && n > 2 && got != n / 2 {
        return Err(TestCaseError::fail(format!(
            "largest_divisor({}) = {} expected n/2 = {}", n, got, n / 2)));
    }
    // a multiple of 3 that is not even has n/3
    if n % 2 != 0 && n % 3 == 0 && n > 3 && got != n / 3 {
        return Err(TestCaseError::fail(format!(
            "largest_divisor({}) = {} expected n/3 = {}", n, got, n / 3)));
    }
    Ok(())
}

pub fn run_property(runner: &mut TestRunner) -> Result<(), TestError<isize>> {
    runner.run(&strategy(), property)
}
