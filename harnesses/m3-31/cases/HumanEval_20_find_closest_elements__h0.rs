use proptest::prelude::*;
use proptest::test_runner::{TestCaseError, TestCaseResult, TestError, TestRunner};
use crate::candidate::find_closest_elements;

pub const STRATEGY_DESC: &str = "vectors of at least two f64; the returned pair is ordered and attains the minimum gap";
fn strategy() -> impl Strategy<Value = Vec<f64>> {
    proptest::collection::vec(-500.0f64..500.0, 2..8)
}

fn property(numbers: Vec<f64>) -> TestCaseResult {
    let (a, b) = find_closest_elements(numbers.clone());
    if a > b {
        return Err(TestCaseError::fail(format!(
            "returned ({}, {}) is not ordered smaller-first", a, b)));
    }
    // both values must be present in the input
    if !numbers.iter().any(|x| (*x - a).abs() < 1e-12)
        || !numbers.iter().any(|x| (*x - b).abs() < 1e-12) {
        return Err(TestCaseError::fail(format!(
            "returned ({}, {}) are not both elements of {:?}", a, b, numbers)));
    }
    // the gap must be the minimum pairwise gap
    let mut best = f64::INFINITY;
    let mut sorted = numbers.clone();
    sorted.sort_by(|x, y| x.partial_cmp(y).unwrap());
    for w in sorted.windows(2) {
        let d = (w[1] - w[0]).abs();
        if d < best { best = d; }
    }
    if ((b - a).abs() - best).abs() > 1e-9 * (1.0 + best.abs()) {
        return Err(TestCaseError::fail(format!(
            "returned gap {} but minimum pairwise gap in {:?} is {}",
            (b - a).abs(), numbers, best)));
    }
    Ok(())
}

pub fn run_property(runner: &mut TestRunner) -> Result<(), TestError<Vec<f64>>> {
    runner.run(&strategy(), property)
}
