use proptest::prelude::*;
use proptest::test_runner::{TestCaseError, TestCaseResult, TestError, TestRunner};
use crate::candidate::make_palindrome;

pub const STRATEGY_DESC: &str = "arbitrary strings; the result must start with the input and be a palindrome";
fn strategy() -> impl Strategy<Value = String> {
    "[a-c]{0,8}"
}

fn property(s: String) -> TestCaseResult {
    let got = make_palindrome(s.clone());
    if !got.starts_with(&s) {
        return Err(TestCaseError::fail(format!(
            "make_palindrome({:?}) = {:?} does not start with the input", s, got)));
    }
    let rev: String = got.chars().rev().collect();
    if got != rev {
        return Err(TestCaseError::fail(format!(
            "make_palindrome({:?}) = {:?} is not a palindrome", s, got)));
    }
    Ok(())
}

pub fn run_property(runner: &mut TestRunner) -> Result<(), TestError<String>> {
    runner.run(&strategy(), property)
}
