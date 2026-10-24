use proptest::prelude::*;
use proptest::test_runner::{TestCaseError, TestCaseResult, TestError, TestRunner};
use crate::candidate::longest;

pub const STRATEGY_DESC: &str = "vectors of strings; the result is the longest, first on ties, None on empty";
fn strategy() -> impl Strategy<Value = Vec<String>> {
    proptest::collection::vec("[a-c]{0,5}", 0..8)
}

fn property(strings: Vec<String>) -> TestCaseResult {
    let got = longest(strings.clone());
    if strings.is_empty() {
        if got.is_some() {
            return Err(TestCaseError::fail(format!(
                "empty vector gave {:?} expected None", got)));
        }
        return Ok(());
    }
    let best_len = strings.iter().map(|s| s.chars().count()).max().unwrap();
    let want = strings.iter().find(|s| s.chars().count() == best_len).cloned();
    if got != want {
        return Err(TestCaseError::fail(format!(
            "longest({:?}) = {:?} expected {:?}", strings, got, want)));
    }
    Ok(())
}

pub fn run_property(runner: &mut TestRunner) -> Result<(), TestError<Vec<String>>> {
    runner.run(&strategy(), property)
}
