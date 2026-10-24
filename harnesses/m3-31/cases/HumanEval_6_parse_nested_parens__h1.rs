use proptest::prelude::*;
use proptest::test_runner::{TestCaseError, TestCaseResult, TestError, TestRunner};
use crate::candidate::parse_nested_parens;

pub const STRATEGY_DESC: &str = "the documented example verbatim plus single-level groups";
fn strategy() -> impl Strategy<Value = String> {
    prop_oneof![
        Just(String::from("(()()) ((())) () ((())()())")),
        proptest::collection::vec(
            proptest::sample::select(vec!["()", "(())", "((()))"]), 1..4usize)
            .prop_map(|g| g.join(" ")),
    ]
}

fn property(s: String) -> TestCaseResult {
    let got = parse_nested_parens(s.clone());
    if s == "(()()) ((())) () ((())()())" {
        let want = vec![2isize, 3, 1, 3];
        if got != want {
            return Err(TestCaseError::fail(format!(
                "documented example gave {:?} expected {:?}", got, want)));
        }
        return Ok(());
    }
    // "()" alone has exactly one level of nesting
    if s == "()" && got != vec![1isize] {
        return Err(TestCaseError::fail(format!(
            "single () gave {:?} expected [1]", got)));
    }
    // every reported depth is at least 1 for a non-empty paren group
    for (i, g) in s.split_whitespace().enumerate() {
        if i < got.len() && !g.is_empty() && g.contains('(') && got[i] < 1 {
            return Err(TestCaseError::fail(format!(
                "group {:?} reported depth {}", g, got[i])));
        }
    }
    Ok(())
}

pub fn run_property(runner: &mut TestRunner) -> Result<(), TestError<String>> {
    runner.run(&strategy(), property)
}
