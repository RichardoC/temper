use proptest::prelude::*;
use proptest::test_runner::{TestCaseError, TestCaseResult, TestError, TestRunner};
use crate::candidate::make_palindrome;

pub const STRATEGY_DESC: &str = "the documented examples plus minimality: no shorter prefix extension is a palindrome";
fn strategy() -> impl Strategy<Value = String> {
    prop_oneof![
        Just(String::from("")),
        Just(String::from("cat")),
        Just(String::from("cata")),
        "[a-b]{0,5}",
    ]
}

fn property(s: String) -> TestCaseResult {
    let got = make_palindrome(s.clone());
    let documented = match s.as_str() {
        "" => Some(String::from("")),
        "cat" => Some(String::from("catac")),
        "cata" => Some(String::from("catac")),
        _ => None,
    };
    if let Some(want) = documented {
        if got != want {
            return Err(TestCaseError::fail(format!(
                "documented example make_palindrome({:?}) = {:?} expected {:?}",
                s, got, want)));
        }
        return Ok(());
    }
    // minimality: the extension appended must be the shortest that works
    if got.starts_with(&s) {
        let ext = &got[s.len()..];
        for k in 0..ext.chars().count() {
            let shorter: String = ext.chars().take(k).collect();
            let candidate = format!("{}{}", s, shorter);
            let rev: String = candidate.chars().rev().collect();
            if candidate == rev {
                return Err(TestCaseError::fail(format!(
                    "make_palindrome({:?}) = {:?} but {:?} is already a palindrome",
                    s, got, candidate)));
            }
        }
    }
    Ok(())
}

pub fn run_property(runner: &mut TestRunner) -> Result<(), TestError<String>> {
    runner.run(&strategy(), property)
}
