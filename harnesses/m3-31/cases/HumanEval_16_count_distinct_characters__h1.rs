use proptest::prelude::*;
use proptest::test_runner::{TestCaseError, TestCaseResult, TestError, TestRunner};
use crate::candidate::count_distinct_characters;

pub const STRATEGY_DESC: &str = "the documented examples and the case-invariance relation";
fn strategy() -> impl Strategy<Value = String> {
    prop_oneof![
        Just(String::from("xyzXYZ")),
        Just(String::from("Jerry")),
        "[a-eA-E]{0,8}",
        Just(String::from("")),
    ]
}

fn property(s: String) -> TestCaseResult {
    let got = count_distinct_characters(s.clone());
    if s == "xyzXYZ" && got != 3 {
        return Err(TestCaseError::fail(format!(
            "documented example xyzXYZ = {} expected 3", got)));
    }
    if s == "Jerry" && got != 4 {
        return Err(TestCaseError::fail(format!(
            "documented example Jerry = {} expected 4", got)));
    }
    if s.is_empty() && got != 0 {
        return Err(TestCaseError::fail(format!(
            "empty string gave {} expected 0", got)));
    }
    // case invariance: the count must not change if the case is flipped
    let flipped: String = s.chars()
        .map(|c| if c.is_ascii_lowercase() { c.to_ascii_uppercase() } else { c.to_ascii_lowercase() })
        .collect();
    let flipped_count = count_distinct_characters(flipped.clone());
    if flipped_count != got {
        return Err(TestCaseError::fail(format!(
            "{:?} counts {} but case-flipped {:?} counts {}",
            s, got, flipped, flipped_count)));
    }
    Ok(())
}

pub fn run_property(runner: &mut TestRunner) -> Result<(), TestError<String>> {
    runner.run(&strategy(), property)
}
