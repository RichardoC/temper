use proptest::prelude::*;
use proptest::test_runner::{TestCaseError, TestCaseResult, TestError, TestRunner};
use crate::candidate::string_sequence;

pub const STRATEGY_DESC: &str = "non-negative n; the result is the space-delimited run 0..=n";
fn strategy() -> impl Strategy<Value = isize> {
    0isize..30
}

fn property(n: isize) -> TestCaseResult {
    let got = string_sequence(n);
    let want: String = (0..=n).map(|i| i.to_string()).collect::<Vec<_>>().join(" ");
    if got != want {
        return Err(TestCaseError::fail(format!(
            "string_sequence({}) = {:?} expected {:?}", n, got, want)));
    }
    Ok(())
}

pub fn run_property(runner: &mut TestRunner) -> Result<(), TestError<isize>> {
    runner.run(&strategy(), property)
}
