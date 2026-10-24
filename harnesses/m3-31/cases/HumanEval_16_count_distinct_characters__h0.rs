use proptest::prelude::*;
use proptest::test_runner::{TestCaseError, TestCaseResult, TestError, TestRunner};
use crate::candidate::count_distinct_characters;

pub const STRATEGY_DESC: &str = "ASCII strings; the count is case-insensitive distinct characters";
fn strategy() -> impl Strategy<Value = String> {
    "[a-zA-Z]{0,10}"
}

fn property(s: String) -> TestCaseResult {
    let got = count_distinct_characters(s.clone());
    let mut seen: Vec<char> = s.chars().map(|c| c.to_ascii_lowercase()).collect();
    seen.sort_unstable();
    seen.dedup();
    let want = seen.len() as isize;
    if got != want {
        return Err(TestCaseError::fail(format!(
            "count_distinct_characters({:?}) = {} expected {} (case-insensitive)",
            s, got, want)));
    }
    Ok(())
}

pub fn run_property(runner: &mut TestRunner) -> Result<(), TestError<String>> {
    runner.run(&strategy(), property)
}
