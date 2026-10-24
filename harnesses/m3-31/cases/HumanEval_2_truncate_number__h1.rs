use proptest::prelude::*;
use proptest::test_runner::{TestCaseError, TestCaseResult, TestError, TestRunner};
use crate::candidate::truncate_number;

pub const STRATEGY_DESC: &str = "the documented example, integral inputs (decimal part 0) and small decimals";
fn strategy() -> impl Strategy<Value = f64> {
    prop_oneof![
        Just(3.5),
        (1u32..50u32).prop_map(|i| i as f64),
        (1u32..1000u32).prop_map(|i| (i as f64) / 1000.0),
    ]
}

fn property(x: f64) -> TestCaseResult {
    let got = truncate_number(x);
    if x == 3.5 && (got - 0.5).abs() > 1e-9 {
        return Err(TestCaseError::fail(format!(
            "documented example truncate_number(3.5) = {} expected 0.5", got)));
    }
    if x.fract() == 0.0 && got.abs() > 1e-9 {
        return Err(TestCaseError::fail(format!(
            "integral input {} gave non-zero decimal part {}", x, got)));
    }
    if got < 0.0 || got >= 1.0 {
        return Err(TestCaseError::fail(format!(
            "decimal part of {} = {} outside [0,1)", x, got)));
    }
    Ok(())
}

pub fn run_property(runner: &mut TestRunner) -> Result<(), TestError<f64>> {
    runner.run(&strategy(), property)
}
