use proptest::prelude::*;
use proptest::test_runner::{TestCaseError, TestCaseResult, TestError, TestRunner};
use crate::candidate::parse_nested_parens;

pub const STRATEGY_DESC: &str = "space-separated balanced groups; the count matches and each depth is the group's max nesting";
fn strategy() -> impl Strategy<Value = String> {
    proptest::collection::vec(
        proptest::sample::select(vec!["()", "(()())", "((()))", "((())()())"]),
        1..4usize)
        .prop_map(|g| g.join(" "))
}

fn property(s: String) -> TestCaseResult {
    let got = parse_nested_parens(s.clone());
    let groups: Vec<&str> = s.split_whitespace().collect();
    if got.len() != groups.len() {
        return Err(TestCaseError::fail(format!(
            "{} groups but {} depths returned for {:?}", groups.len(), got.len(), s)));
    }
    for (i, g) in groups.iter().enumerate() {
        let mut depth = 0i64;
        let mut maxd = 0i64;
        for ch in g.chars() {
            if ch == '(' { depth += 1; maxd = maxd.max(depth); }
            if ch == ')' { depth -= 1; }
        }
        if got[i] != maxd {
            return Err(TestCaseError::fail(format!(
                "group {:?} has max nesting {} but got {}", g, maxd, got[i])));
        }
    }
    Ok(())
}

pub fn run_property(runner: &mut TestRunner) -> Result<(), TestError<String>> {
    runner.run(&strategy(), property)
}
