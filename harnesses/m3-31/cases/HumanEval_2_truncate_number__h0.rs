use proptest::prelude::*;
use proptest::test_runner::{TestCaseError, TestCaseResult, TestError, TestRunner};
use crate::candidate::truncate_number;

pub const STRATEGY_DESC: &str = "positive non-integral f64: decimal part is in [0,1) and the integer part reconstructs the input";
fn strategy() -> impl Strategy<Value = f64> {
    (0u32..100_000u32, 1u32..1000u32)
        .prop_map(|(whole, frac)| whole as f64 + (frac as f64) / 1000.0)
        .prop_filter("strictly positive", |x| *x > 0.0)
}

fn property(x: f64) -> TestCaseResult {
    let got = truncate_number(x);
    let whole = x.trunc();
    let want = x - whole;
    if !(0.0..1.0).contains(&got) {
        return Err(TestCaseError::fail(format!(
            "decimal part of {} = {} is outside [0,1)", x, got)));
    }
    if (got - want).abs() > 1e-9 {
        return Err(TestCaseError::fail(format!(
            "decimal part of {} = {} but x - trunc(x) = {}", x, got, want)));
    }
    if ((whole + got) - x).abs() > 1e-9 {
        return Err(TestCaseError::fail(format!(
            "integer part + decimal part {} != {}", whole + got, x)));
    }
    Ok(())
}

pub fn run_property(runner: &mut TestRunner) -> Result<(), TestError<f64>> {
    runner.run(&strategy(), property)
}
