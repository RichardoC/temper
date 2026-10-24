use proptest::prelude::*;
use proptest::test_runner::{TestCaseError, TestCaseResult, TestError, TestRunner};
use crate::candidate::all_prefixes;

pub const STRATEGY_DESC: &str = "the documented example plus a monotone-length relation";
fn strategy() -> impl Strategy<Value = String> {
    prop_oneof![
        Just(String::from("abc")),
        "[xy]{1,6}",
        Just(String::from("")),
    ]
}

fn property(s: String) -> TestCaseResult {
    let got = all_prefixes(s.clone());
    if s == "abc" {
        let want = vec![String::from("a"), String::from("ab"), String::from("abc")];
        if got != want {
            return Err(TestCaseError::fail(format!(
                "documented example gave {:?} expected {:?}", got, want)));
        }
        return Ok(());
    }
    if s.is_empty() && !got.is_empty() {
        return Err(TestCaseError::fail(format!(
            "empty string gave {:?} expected empty", got)));
    }
    // lengths must be strictly increasing by exactly one, starting at 1
    for (i, p) in got.iter().enumerate() {
        if p.chars().count() != i + 1 {
            return Err(TestCaseError::fail(format!(
                "prefix {} has length {} expected {}", i, p.chars().count(), i + 1)));
        }
    }
    // every prefix must be a prefix of the input
    for p in &got {
        if !s.starts_with(p) {
            return Err(TestCaseError::fail(format!(
                "{:?} is not a prefix of {:?}", p, s)));
        }
    }
    Ok(())
}

pub fn run_property(runner: &mut TestRunner) -> Result<(), TestError<String>> {
    runner.run(&strategy(), property)
}
