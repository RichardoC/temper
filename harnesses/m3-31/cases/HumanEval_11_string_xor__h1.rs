use proptest::prelude::*;
use proptest::test_runner::{TestCaseError, TestCaseResult, TestError, TestRunner};
use crate::candidate::string_xor;

pub const STRATEGY_DESC: &str = "the documented example and the x XOR x = 0 identity";
fn strategy() -> impl Strategy<Value = (String, String)> {
    prop_oneof![
        Just((String::from("010"), String::from("110"))),
        "[01]{0,6}".prop_map(|a| (a.clone(), a)),
        ("[01]{0,4}", "[01]{0,4}").prop_map(|(a, b)| {
            let n = a.len().min(b.len());
            (a[..n].to_string(), b[..n].to_string())
        }),
    ]
}

fn property(input: (String, String)) -> TestCaseResult {
    let (a, b) = input;
    let got = string_xor(a.clone(), b.clone());
    if a == "010" && b == "110" && got != "100" {
        return Err(TestCaseError::fail(format!(
            "documented example gave {:?} expected "100"", got)));
    }
    if a == b && got.chars().any(|c| c != '0') {
        return Err(TestCaseError::fail(format!(
            "x XOR x should be all zeros, got {:?}", got)));
    }
    if got.chars().any(|c| c != '0' && c != '1') {
        return Err(TestCaseError::fail(format!(
            "result {:?} contains a non-binary character", got)));
    }
    Ok(())
}

pub fn run_property(runner: &mut TestRunner) -> Result<(), TestError<(String, String)>> {
    runner.run(&strategy(), property)
}
