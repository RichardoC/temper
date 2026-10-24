use proptest::prelude::*;
use proptest::test_runner::{TestCaseError, TestCaseResult, TestError, TestRunner};
use crate::candidate::all_prefixes;

pub const STRATEGY_DESC: &str = "arbitrary strings; prefix i is exactly the first i+1 characters, shortest to longest";
fn strategy() -> impl Strategy<Value = String> {
    "[a-c]{0,8}"
}

fn property(s: String) -> TestCaseResult {
    let got = all_prefixes(s.clone());
    let chars: Vec<char> = s.chars().collect();
    if got.len() != chars.len() {
        return Err(TestCaseError::fail(format!(
            "all_prefixes({:?}) has {} entries expected {}", s, got.len(), chars.len())));
    }
    for (i, p) in got.iter().enumerate() {
        let want: String = chars[..=i].iter().collect();
        if *p != want {
            return Err(TestCaseError::fail(format!(
                "all_prefixes({:?})[{}] = {:?} expected {:?}", s, i, p, want)));
        }
    }
    Ok(())
}

pub fn run_property(runner: &mut TestRunner) -> Result<(), TestError<String>> {
    runner.run(&strategy(), property)
}
