use proptest::prelude::*;
use proptest::test_runner::{TestCaseError, TestCaseResult, TestError, TestRunner};
use crate::candidate::strlen;

pub const STRATEGY_DESC: &str = "arbitrary strings; the length is the character count";
fn strategy() -> impl Strategy<Value = String> {
    ".{0,12}"
}

fn property(s: String) -> TestCaseResult {
    let got = strlen(s.clone());
    let want = s.chars().count() as isize;
    if got != want {
        return Err(TestCaseError::fail(format!(
            "strlen({:?}) = {} expected {}", s, got, want)));
    }
    Ok(())
}

pub fn run_property(runner: &mut TestRunner) -> Result<(), TestError<String>> {
    runner.run(&strategy(), property)
}
