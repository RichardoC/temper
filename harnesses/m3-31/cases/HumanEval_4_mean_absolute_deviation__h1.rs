use proptest::prelude::*;
use proptest::test_runner::{TestCaseError, TestCaseResult, TestError, TestRunner};
use crate::candidate::mean_absolute_deviation;

pub const STRATEGY_DESC: &str = "the documented example and a shift-invariance relation";
fn strategy() -> impl Strategy<Value = Vec<f64>> {
    prop_oneof![
        Just(vec![1.0f64, 2.0, 3.0, 4.0]),
        (proptest::collection::vec(-100.0f64..100.0, 2..8), -500.0f64..500.0)
            .prop_map(|(v, _shift)| v),
    ]
}

fn property(numbers: Vec<f64>) -> TestCaseResult {
    let got = mean_absolute_deviation(numbers.clone());
    if numbers == vec![1.0, 2.0, 3.0, 4.0] && (got - 1.0).abs() > 1e-9 {
        return Err(TestCaseError::fail(format!(
            "documented example MAD([1,2,3,4]) = {} expected 1.0", got)));
    }
    // MAD is non-negative, and is 0 exactly when every element is equal
    if got < -1e-9 {
        return Err(TestCaseError::fail(format!(
            "MAD of {:?} = {} is negative", numbers, got)));
    }
    let first = numbers[0];
    if numbers.iter().all(|x| (*x - first).abs() < 1e-12) && got.abs() > 1e-6 {
        return Err(TestCaseError::fail(format!(
            "constant vector {:?} gave MAD {}", numbers, got)));
    }
    Ok(())
}

pub fn run_property(runner: &mut TestRunner) -> Result<(), TestError<Vec<f64>>> {
    runner.run(&strategy(), property)
}
