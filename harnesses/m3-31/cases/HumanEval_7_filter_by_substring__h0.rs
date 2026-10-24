use proptest::prelude::*;
use proptest::test_runner::{TestCaseError, TestCaseResult, TestError, TestRunner};
use crate::candidate::filter_by_substring;

pub const STRATEGY_DESC: &str = "vectors of strings and a filter; the result is exactly the subsequence containing it";
fn strategy() -> impl Strategy<Value = (Vec<String>, String)> {
    (
        proptest::collection::vec("[a-c]{0,4}", 0..8),
        "[a-c]{1,2}",
    )
}

fn property(input: (Vec<String>, String)) -> TestCaseResult {
    let (strings, needle) = input;
    let got = filter_by_substring(strings.clone(), needle.clone());
    let want: Vec<String> = strings.iter().filter(|s| s.contains(&needle)).cloned().collect();
    if got != want {
        return Err(TestCaseError::fail(format!(
            "filter_by_substring({:?}, {:?}) = {:?} expected {:?}",
            strings, needle, got, want)));
    }
    Ok(())
}

pub fn run_property(runner: &mut TestRunner) -> Result<(), TestError<(Vec<String>, String)>> {
    runner.run(&strategy(), property)
}
