use proptest::prelude::*;
use proptest::test_runner::{TestCaseError, TestCaseResult, TestError, TestRunner};
use crate::candidate::separate_paren_groups;

pub const STRATEGY_DESC: &str = "the documented example verbatim plus a nesting-depth check per group";
fn strategy() -> impl Strategy<Value = String> {
    prop_oneof![
        Just(String::from("( ) (( )) (( )( ))")),
        proptest::collection::vec(
            proptest::sample::select(vec!["()", "(())", "((()))", "()()"]),
            1..3usize)
            .prop_map(|g| g.join(" ")),
    ]
}

fn property(s: String) -> TestCaseResult {
    let out = separate_paren_groups(s.clone());
    if s == "( ) (( )) (( )( ))" {
        let want = vec![String::from("()"), String::from("(())"), String::from("(()())")];
        if out != want {
            return Err(TestCaseError::fail(format!(
                "documented example returned {:?}, expected {:?}", out, want)));
        }
        return Ok(());
    }
    // every returned group must be a balanced nesting (max depth >= 1, ends at 0)
    for part in &out {
        let mut depth = 0i64;
        let mut maxd = 0i64;
        for ch in part.chars() {
            if ch == '(' { depth += 1; maxd = maxd.max(depth); }
            if ch == ')' { depth -= 1; }
        }
        if depth != 0 || maxd < 1 {
            return Err(TestCaseError::fail(format!(
                "group {:?}: depth ends {} max {}", part, depth, maxd)));
        }
    }
    Ok(())
}

pub fn run_property(runner: &mut TestRunner) -> Result<(), TestError<String>> {
    runner.run(&strategy(), property)
}
