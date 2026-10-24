use proptest::prelude::*;
use proptest::test_runner::{TestCaseError, TestCaseResult, TestError, TestRunner};
use crate::candidate::has_close_elements;

pub const STRATEGY_DESC: &str = "the documentation's worked examples verbatim, plus single-element vectors";
fn strategy() -> impl Strategy<Value = (Vec<f64>, f64)> {
    prop_oneof![
        Just((vec![1.0, 2.0, 3.0], 0.5)),
        Just((vec![1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)),
        (proptest::collection::vec(-10.0f64..10.0, 0..4), 0.0f64..3.0),
    ]
}

fn property(input: (Vec<f64>, f64)) -> TestCaseResult {
    let (numbers, threshold) = input;
    // empty and single-element vectors can never contain a close PAIR
    if numbers.len() < 2 {
        if has_close_elements(numbers.clone(), threshold) {
            return Err(TestCaseError::fail(format!(
                "reported a close pair in a {} element vector", numbers.len())));
        }
        return Ok(());
    }
    let got = has_close_elements(numbers.clone(), threshold);
    // the documented examples fix two decisions exactly
    let documented = match (numbers.as_slice(), threshold) {
        (a, t) if a == [1.0, 2.0, 3.0] && t == 0.5 => Some(false),
        (a, t) if a == [1.0, 2.8, 3.0, 4.0, 5.0, 2.0] && t == 0.3 => Some(true),
        _ => None,
    };
    if let Some(want) = documented {
        if got != want {
            return Err(TestCaseError::fail(format!(
                "documented example {:?} @ {} returned {}", numbers, threshold, got)));
        }
    }
    Ok(())
}

pub fn run_property(runner: &mut TestRunner) -> Result<(), TestError<(Vec<f64>, f64)>> {
    runner.run(&strategy(), property)
}
