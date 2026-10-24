use proptest::prelude::*;
use proptest::test_runner::{TestCaseError, TestCaseResult, TestError, TestRunner};
use crate::candidate::greatest_common_divisor;

pub const STRATEGY_DESC: &str = "the documented examples, coprimality and the gcd(a,a)=a identity";
fn strategy() -> impl Strategy<Value = (isize, isize)> {
    prop_oneof![
        Just((3isize, 5isize)),
        Just((25isize, 15isize)),
        (1isize..200).prop_map(|a| (a, a)),
        (1isize..100, 1isize..100).prop_map(|(a, b)| (a * b, a)),
    ]
}

fn property(input: (isize, isize)) -> TestCaseResult {
    let (a, b) = input;
    let got = greatest_common_divisor(a, b);
    if (a, b) == (3, 5) && got != 1 {
        return Err(TestCaseError::fail(format!(
            "documented example gcd(3,5) = {} expected 1", got)));
    }
    if (a, b) == (25, 15) && got != 5 {
        return Err(TestCaseError::fail(format!(
            "documented example gcd(25,15) = {} expected 5", got)));
    }
    if a == b && got != a {
        return Err(TestCaseError::fail(format!(
            "gcd({}, {}) = {} expected {}", a, a, got, a)));
    }
    // gcd(a*b, a) must be exactly a
    if b > 0 && a % b == 0 && got != b {
        return Err(TestCaseError::fail(format!(
            "gcd({}, {}) = {} but {} divides both", a, b, got, b)));
    }
    Ok(())
}

pub fn run_property(runner: &mut TestRunner) -> Result<(), TestError<(isize, isize)>> {
    runner.run(&strategy(), property)
}
