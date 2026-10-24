use proptest::prelude::*;
use proptest::test_runner::{TestCaseError, TestCaseResult, TestError, TestRunner};
use crate::candidate::string_sequence;

pub const STRATEGY_DESC: &str = "the documented examples, and the token count equals n+1";
fn strategy() -> impl Strategy<Value = isize> {
    prop_oneof![Just(0isize), Just(5isize), 0isize..12]
}

fn property(n: isize) -> TestCaseResult {
    let got = string_sequence(n);
    if n == 0 && got != "0" {
        return Err(TestCaseError::fail(format!(
            "string_sequence(0) = {:?} expected "0"", got)));
    }
    if n == 5 && got != "0 1 2 3 4 5" {
        return Err(TestCaseError::fail(format!(
            "string_sequence(5) = {:?} expected "0 1 2 3 4 5"", got)));
    }
    let tokens: Vec<&str> = got.split(' ').collect();
    if tokens.len() != (n as usize) + 1 {
        return Err(TestCaseError::fail(format!(
            "string_sequence({}) has {} tokens expected {}", n, tokens.len(), n + 1)));
    }
    if tokens[0] != "0" {
        return Err(TestCaseError::fail(format!(
            "string_sequence({}) starts with {:?} expected "0"", n, tokens[0])));
    }
    if tokens.last() != Some(&(n.to_string().as_str())) {
        return Err(TestCaseError::fail(format!(
            "string_sequence({}) ends with {:?} expected {}", n, tokens.last(), n)));
    }
    Ok(())
}

pub fn run_property(runner: &mut TestRunner) -> Result<(), TestError<isize>> {
    runner.run(&strategy(), property)
}
