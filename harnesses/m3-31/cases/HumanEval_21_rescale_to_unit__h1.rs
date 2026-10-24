use proptest::prelude::*;
use proptest::test_runner::{TestCaseError, TestCaseResult, TestError, TestRunner};
use crate::candidate::rescale_to_unit;

pub const STRATEGY_DESC: &str = "the documented example plus order preservation and the [0,1] range bound";
fn strategy() -> impl Strategy<Value = Vec<f64>> {
    prop_oneof![
        Just(vec![1.0f64, 2.0, 3.0, 4.0, 5.0]),
        proptest::collection::vec(0.0f64..50.0, 2..6).prop_filter(
            "non-constant",
            |v| v.windows(2).any(|w| (w[0] - w[1]).abs() > 1e-6)
                || {
                    let lo = v.iter().cloned().fold(f64::INFINITY, f64::min);
                    let hi = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    (hi - lo).abs() > 1e-6
                }),
    ]
}

fn property(numbers: Vec<f64>) -> TestCaseResult {
    let got = rescale_to_unit(numbers.clone());
    if numbers == vec![1.0, 2.0, 3.0, 4.0, 5.0] {
        let want = vec![0.0f64, 0.25, 0.5, 0.75, 1.0];
        for (i, w) in want.iter().enumerate() {
            if i < got.len() && (got[i] - w).abs() > 1e-9 {
                return Err(TestCaseError::fail(format!(
                    "documented example [{}] = {} expected {}", i, got[i], w)));
            }
        }
        return Ok(());
    }
    // every rescaled value lies in [0, 1]
    for v in &got {
        if *v < -1e-9 || *v > 1.0 + 1e-9 {
            return Err(TestCaseError::fail(format!(
                "rescaled value {} of {:?} is outside [0,1]", v, numbers)));
        }
    }
    // order preservation: if x_i <= x_j then out_i <= out_j
    for i in 0..numbers.len() {
        for j in 0..numbers.len() {
            if i < got.len() && j < got.len()
                && numbers[i] <= numbers[j] && got[i] > got[j] + 1e-9 {
                return Err(TestCaseError::fail(format!(
                    "order changed: {} <= {} but {} > {}",
                    numbers[i], numbers[j], got[i], got[j])));
            }
        }
    }
    Ok(())
}

pub fn run_property(runner: &mut TestRunner) -> Result<(), TestError<Vec<f64>>> {
    runner.run(&strategy(), property)
}
