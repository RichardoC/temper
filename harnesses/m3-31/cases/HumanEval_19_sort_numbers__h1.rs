use proptest::prelude::*;
use proptest::test_runner::{TestCaseError, TestCaseResult, TestError, TestRunner};
use crate::candidate::sort_numbers;

pub const STRATEGY_DESC: &str = "the documented example plus an explicit multiset-preservation check";
fn strategy() -> impl Strategy<Value = String> {
    prop_oneof![
        Just(String::from("three one five")),
        proptest::collection::vec(
            proptest::sample::select(vec!["zero", "one", "four", "nine"]),
            1..5usize)
            .prop_map(|w| w.join(" ")),
    ]
}

fn property(s: String) -> TestCaseResult {
    let got = sort_numbers(s.clone());
    if s == "three one five" && got != "one three five" {
        return Err(TestCaseError::fail(format!(
            "documented example gave {:?} expected "one three five"", got)));
    }
    // multiset preservation: the same words, just reordered
    let mut a: Vec<&str> = s.split_whitespace().collect();
    let mut b: Vec<&str> = got.split_whitespace().collect();
    a.sort_unstable();
    b.sort_unstable();
    if a != b {
        return Err(TestCaseError::fail(format!(
            "sort_numbers({:?}) = {:?} changed the word multiset", s, got)));
    }
    Ok(())
}

pub fn run_property(runner: &mut TestRunner) -> Result<(), TestError<String>> {
    runner.run(&strategy(), property)
}
