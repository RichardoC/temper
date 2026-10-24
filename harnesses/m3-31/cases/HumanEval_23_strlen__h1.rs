use proptest::prelude::*;
use proptest::test_runner::{TestCaseError, TestCaseResult, TestError, TestRunner};
use crate::candidate::strlen;

pub const STRATEGY_DESC: &str = "the documented examples plus the concatenation relation";
fn strategy() -> impl Strategy<Value = String> {
    prop_oneof![
        Just(String::from("")),
        Just(String::from("abc")),
        "[a-z]{0,6}",
    ]
}

fn property(s: String) -> TestCaseResult {
    let got = strlen(s.clone());
    if s.is_empty() && got != 0 {
        return Err(TestCaseError::fail(format!(
            "strlen("") = {} expected 0", got)));
    }
    if s == "abc" && got != 3 {
        return Err(TestCaseError::fail(format!(
            "documented example strlen("abc") = {} expected 3", got)));
    }
    // concatenation: len(a + b) == len(a) + len(b)
    let a = s.clone();
    let b = String::from("zz");
    let ab = strlen(format!("{}{}", a, b));
    if ab != got + 2 {
        return Err(TestCaseError::fail(format!(
            "strlen({:?}) = {} but strlen({:?} + "zz") = {}", s, got, s, ab)));
    }
    Ok(())
}

pub fn run_property(runner: &mut TestRunner) -> Result<(), TestError<String>> {
    runner.run(&strategy(), property)
}
