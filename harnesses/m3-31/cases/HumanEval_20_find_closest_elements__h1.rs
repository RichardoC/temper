use proptest::prelude::*;
use proptest::test_runner::{TestCaseError, TestCaseResult, TestError, TestRunner};
use crate::candidate::find_closest_elements;

pub const STRATEGY_DESC: &str = "the documented examples and duplicate-heavy vectors (gap zero)";
fn strategy() -> impl Strategy<Value = Vec<f64>> {
    prop_oneof![
        Just(vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 2.2]),
        Just(vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 2.0]),
        proptest::collection::vec(
            (0u32..5u32).prop_map(|i| i as f64), 2..8),
    ]
}

fn property(numbers: Vec<f64>) -> TestCaseResult {
    let got = find_closest_elements(numbers.clone());
    if numbers == vec![1.0, 2.0, 3.0, 4.0, 5.0, 2.2] && got != (2.0, 2.2) {
        return Err(TestCaseError::fail(format!(
            "documented example gave {:?} expected (2.0, 2.2)", got)));
    }
    if numbers == vec![1.0, 2.0, 3.0, 4.0, 5.0, 2.0] && got != (2.0, 2.0) {
        return Err(TestCaseError::fail(format!(
            "documented example gave {:?} expected (2.0, 2.0)", got)));
    }
    // a duplicated value forces a zero gap
    let mut sorted = numbers.clone();
    sorted.sort_by(|x, y| x.partial_cmp(y).unwrap());
    if sorted.windows(2).any(|w| (w[1] - w[0]).abs() < 1e-12) {
        if (got.1 - got.0).abs() > 1e-12 {
            return Err(TestCaseError::fail(format!(
                "{:?} contains a duplicate so the closest pair has gap 0, got {:?}",
                numbers, got)));
        }
    }
    Ok(())
}

pub fn run_property(runner: &mut TestRunner) -> Result<(), TestError<Vec<f64>>> {
    runner.run(&strategy(), property)
}
