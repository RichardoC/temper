use proptest::prelude::*;
use proptest::test_runner::{TestCaseError, TestCaseResult, TestError, TestRunner};
use crate::candidate::string_xor;

pub const STRATEGY_DESC: &str = "equal-length binary strings; XOR is character-wise and length is preserved";
fn strategy() -> impl Strategy<Value = (String, String)> {
    ("[01]{0,8}", "[01]{0,8}").prop_map(|(a, b)| {
        let n = a.len().min(b.len());
        (a[..n].to_string(), b[..n].to_string())
    })
}

fn property(input: (String, String)) -> TestCaseResult {
    let (a, b) = input;
    let got = string_xor(a.clone(), b.clone());
    let want: String = a.chars().zip(b.chars())
        .map(|(x, y)| if x == y { '0' } else { '1' })
        .collect();
    if got != want {
        return Err(TestCaseError::fail(format!(
            "string_xor({:?}, {:?}) = {:?} expected {:?}", a, b, got, want)));
    }
    if got.len() != a.len() {
        return Err(TestCaseError::fail(format!(
            "length changed: {} -> {}", a.len(), got.len())));
    }
    Ok(())
}

pub fn run_property(runner: &mut TestRunner) -> Result<(), TestError<(String, String)>> {
    runner.run(&strategy(), property)
}
