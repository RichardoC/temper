use proptest::prelude::*;
use proptest::test_runner::{TestCaseError, TestCaseResult, TestError, TestRunner};
use crate::candidate::has_close_elements;

pub const STRATEGY_DESC: &str = "pairs of f64 with a non-negative threshold, checking the documented examples' decision";
fn strategy() -> impl Strategy<Value = (Vec<f64>, f64)> {
    (proptest::collection::vec(-1000.0f64..1000.0, 0..12), 0.0f64..50.0)
}

fn property(input: (Vec<f64>, f64)) -> TestCaseResult {
    let (numbers, threshold) = input;
    let got = has_close_elements(numbers.clone(), threshold);
    // independent recomputation: some pair is closer than the threshold
    let mut want = false;
    for i in 0..numbers.len() {
        for j in (i + 1)..numbers.len() {
            if (numbers[i] - numbers[j]).abs() < threshold {
                want = true;
            }
        }
    }
    if got == want {
        Ok(())
    } else {
        Err(TestCaseError::fail(format!(
            "has_close_elements({:?}, {}) = {} but pairwise scan says {}",
            numbers, threshold, got, want)))
    }
}

pub fn run_property(runner: &mut TestRunner) -> Result<(), TestError<(Vec<f64>, f64)>> {
    runner.run(&strategy(), property)
}
