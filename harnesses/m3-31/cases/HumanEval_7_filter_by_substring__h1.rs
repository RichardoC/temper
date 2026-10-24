use proptest::prelude::*;
use proptest::test_runner::{TestCaseError, TestCaseResult, TestError, TestRunner};
use crate::candidate::filter_by_substring;

pub const STRATEGY_DESC: &str = "the documented example, an empty vector, and a no-match needle";
fn strategy() -> impl Strategy<Value = (Vec<String>, String)> {
    prop_oneof![
        Just((vec![], String::from("a"))),
        Just((vec![
            String::from("abc"), String::from("bacd"),
            String::from("cde"), String::from("array")], String::from("a"))),
        (proptest::collection::vec("[xyz]{0,5}", 0..6), Just(String::from("a"))),
    ]
}

fn property(input: (Vec<String>, String)) -> TestCaseResult {
    let (strings, needle) = input;
    let got = filter_by_substring(strings.clone(), needle.clone());
    if strings.is_empty() && !got.is_empty() {
        return Err(TestCaseError::fail(format!(
            "empty input gave {:?} expected empty", got)));
    }
    if needle == "a" && strings.iter().all(|s| !s.contains('a')) {
        if !got.is_empty() {
            return Err(TestCaseError::fail(format!(
                "no string contains the needle yet got {:?}", got)));
        }
        return Ok(());
    }
    if strings.iter().any(|s| s == "abc") && needle == "a" {
        if !got.contains(&String::from("abc")) {
            return Err(TestCaseError::fail(format!(
                "documented example dropped "abc" -> {:?}", got)));
        }
    }
    for s in &got {
        if !s.contains(&needle) {
            return Err(TestCaseError::fail(format!(
                "returned {:?} which does not contain {:?}", s, needle)));
        }
    }
    Ok(())
}

pub fn run_property(runner: &mut TestRunner) -> Result<(), TestError<(Vec<String>, String)>> {
    runner.run(&strategy(), property)
}
