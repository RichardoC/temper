use proptest::prelude::*;
use proptest::test_runner::{TestCaseError, TestCaseResult, TestError, TestRunner};
use crate::candidate::how_many_times;

pub const STRATEGY_DESC: &str = "the documented examples and the empty-substring boundary";
fn strategy() -> impl Strategy<Value = (String, String)> {
    prop_oneof![
        Just((String::from(""), String::from("a"))),
        Just((String::from("aaa"), String::from("a"))),
        Just((String::from("aaaa"), String::from("aa"))),
        ("[ab]{0,8}", "[ab]{1,2}"),
    ]
}

fn property(input: (String, String)) -> TestCaseResult {
    let (s, sub) = input;
    let got = how_many_times(s.clone(), sub.clone());
    if s.is_empty() && got != 0 {
        return Err(TestCaseError::fail(format!(
            "empty haystack gave {} expected 0", got)));
    }
    if s == "aaa" && sub == "a" && got != 3 {
        return Err(TestCaseError::fail(format!(
            "documented example (aaa, a) = {} expected 3", got)));
    }
    if s == "aaaa" && sub == "aa" && got != 3 {
        return Err(TestCaseError::fail(format!(
            "documented example (aaaa, aa) = {} expected 3 (overlapping)", got)));
    }
    if !s.is_empty() && !sub.is_empty() {
        // an occurrence count can never exceed the haystack length
        if got > s.chars().count() as isize {
            return Err(TestCaseError::fail(format!(
                "({:?}, {:?}) = {} exceeds the haystack length", s, sub, got)));
        }
        if got < 0 {
            return Err(TestCaseError::fail(format!(
                "({:?}, {:?}) = {} is negative", s, sub, got)));
        }
    }
    Ok(())
}

pub fn run_property(runner: &mut TestRunner) -> Result<(), TestError<(String, String)>> {
    runner.run(&strategy(), property)
}
