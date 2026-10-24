use proptest::prelude::*;
use proptest::test_runner::{TestCaseError, TestCaseResult, TestError, TestRunner};
use crate::candidate::separate_paren_groups;

pub const STRATEGY_DESC: &str = "space-joined balanced paren groups; each returned group must be balanced and conserve parentheses";
fn strategy() -> impl Strategy<Value = String> {
    proptest::collection::vec(
        proptest::sample::select(vec!["()", "(())", "()()", "(()())", "(())()"]),
        0..4usize,
    )
    .prop_map(|groups| groups.join(" "))
}

fn property(s: String) -> TestCaseResult {
    let out = separate_paren_groups(s.clone());
    for part in &out {
        let mut depth = 0i64;
        for ch in part.chars() {
            match ch {
                '(' => depth += 1,
                ')' => depth -= 1,
                _ => {}
            }
            if depth < 0 {
                return Err(TestCaseError::fail(format!(
                    "negative depth inside group {:?}", part)));
            }
        }
        if depth != 0 {
            return Err(TestCaseError::fail(format!(
                "group {:?} is not balanced", part)));
        }
        if part.contains(' ') {
            return Err(TestCaseError::fail(format!(
                "group {:?} kept a space though spaces must be ignored", part)));
        }
    }
    let strip = |t: &str| -> String {
        t.chars().filter(|c| *c == '(' || *c == ')').collect()
    };
    if strip(&out.concat()) != strip(&s) {
        return Err(TestCaseError::fail(format!(
            "parentheses not conserved: input {:?} -> output {:?}", s, out)));
    }
    Ok(())
}

pub fn run_property(runner: &mut TestRunner) -> Result<(), TestError<String>> {
    runner.run(&strategy(), property)
}
