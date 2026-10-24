use proptest::prelude::*;
use proptest::test_runner::{TestCaseError, TestCaseResult, TestError, TestRunner};
use crate::candidate::parse_music;

pub const STRATEGY_DESC: &str = "note strings built from the documented legend; each symbol maps to its beat count";
fn strategy() -> impl Strategy<Value = String> {
    proptest::collection::vec(
        proptest::sample::select(vec!["o", "o|", ".|"]), 1..8usize)
        .prop_map(|n| n.join(" "))
}

fn property(s: String) -> TestCaseResult {
    let got = parse_music(s.clone());
    let want: Vec<isize> = s.split_whitespace()
        .map(|tok| match tok {
            "o" => 4,
            "o|" => 2,
            ".|" => 1,
            _ => 0,
        })
        .collect();
    if got != want {
        return Err(TestCaseError::fail(format!(
            "parse_music({:?}) = {:?} expected {:?}", s, got, want)));
    }
    Ok(())
}

pub fn run_property(runner: &mut TestRunner) -> Result<(), TestError<String>> {
    runner.run(&strategy(), property)
}
