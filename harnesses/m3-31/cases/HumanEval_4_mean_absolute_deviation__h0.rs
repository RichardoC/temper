use proptest::prelude::*;
use proptest::test_runner::{TestCaseError, TestCaseResult, TestError, TestRunner};
use crate::candidate::mean_absolute_deviation;

pub const STRATEGY_DESC: &str = "vectors of finite f64; MAD equals the mean absolute deviation from the mean";
fn strategy() -> impl Strategy<Value = Vec<f64>> {
    proptest::collection::vec(-1000.0f64..1000.0, 1..10)
}

fn property(numbers: Vec<f64>) -> TestCaseResult {
    let got = mean_absolute_deviation(numbers.clone());
    let n = numbers.len() as f64;
    let mean = numbers.iter().sum::<f64>() / n;
    let want = numbers.iter().map(|x| (x - mean).abs()).sum::<f64>() / n;
    if (got - want).abs() > 1e-6 * (1.0 + want.abs()) {
        return Err(TestCaseError::fail(format!(
            "MAD of {:?} = {} but mean absolute deviation from mean = {}",
            numbers, got, want)));
    }
    Ok(())
}

pub fn run_property(runner: &mut TestRunner) -> Result<(), TestError<Vec<f64>>> {
    runner.run(&strategy(), property)
}
