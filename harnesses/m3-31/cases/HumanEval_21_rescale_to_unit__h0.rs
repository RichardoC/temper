use proptest::prelude::*;
use proptest::test_runner::{TestCaseError, TestCaseResult, TestError, TestRunner};
use crate::candidate::rescale_to_unit;

pub const STRATEGY_DESC: &str = "vectors of at least two f64; min maps to 0, max maps to 1, order is preserved";
fn strategy() -> impl Strategy<Value = Vec<f64>> {
    proptest::collection::vec(-100.0f64..100.0, 2..8)
        .prop_filter("non-constant", |v| {
            let lo = v.iter().cloned().fold(f64::INFINITY, f64::min);
            let hi = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            (hi - lo).abs() > 1e-6
        })
}

fn property(numbers: Vec<f64>) -> TestCaseResult {
    let got = rescale_to_unit(numbers.clone());
    if got.len() != numbers.len() {
        return Err(TestCaseError::fail(format!(
            "rescale_to_unit({:?}) has len {} expected {}",
            numbers, got.len(), numbers.len())));
    }
    let lo = numbers.iter().cloned().fold(f64::INFINITY, f64::min);
    let hi = numbers.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let want: Vec<f64> = numbers.iter().map(|x| (x - lo) / (hi - lo)).collect();
    for (i, w) in want.iter().enumerate() {
        if (got[i] - w).abs() > 1e-6 * (1.0 + w.abs()) {
            return Err(TestCaseError::fail(format!(
                "rescale_to_unit({:?})[{}] = {} expected {}", numbers, i, got[i], w)));
        }
    }
    // the minimum must become 0 and the maximum 1
    let got_lo = got.iter().cloned().fold(f64::INFINITY, f64::min);
    let got_hi = got.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if got_lo.abs() > 1e-9 || (got_hi - 1.0).abs() > 1e-9 {
        return Err(TestCaseError::fail(format!(
            "rescaled range is [{}, {}] expected [0, 1]", got_lo, got_hi)));
    }
    Ok(())
}

pub fn run_property(runner: &mut TestRunner) -> Result<(), TestError<Vec<f64>>> {
    runner.run(&strategy(), property)
}
