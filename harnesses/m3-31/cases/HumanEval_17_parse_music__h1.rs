use proptest::prelude::*;
use proptest::test_runner::{TestCaseError, TestCaseResult, TestError, TestRunner};
use crate::candidate::parse_music;

pub const STRATEGY_DESC: &str = "the documented example verbatim plus the length relation";
fn strategy() -> impl Strategy<Value = String> {
    prop_oneof![
        Just(String::from("o o| .| o| o| .| .| .| .| o o")),
        proptest::collection::vec(
            proptest::sample::select(vec!["o", "o|", ".|"]), 1..6usize)
            .prop_map(|n| n.join(" ")),
    ]
}

fn property(s: String) -> TestCaseResult {
    let got = parse_music(s.clone());
    if s == "o o| .| o| o| .| .| .| .| o o" {
        let want = vec![4isize, 2, 1, 2, 2, 1, 1, 1, 1, 4, 4];
        if got != want {
            return Err(TestCaseError::fail(format!(
                "documented example gave {:?} expected {:?}", got, want)));
        }
        return Ok(());
    }
    // one beat count per whitespace-separated note
    let n_notes = s.split_whitespace().count();
    if got.len() != n_notes {
        return Err(TestCaseError::fail(format!(
            "{:?} has {} notes but {} beats returned", s, n_notes, got.len())));
    }
    // every beat count must be one of the three documented values
    for b in &got {
        if !(*b == 1 || *b == 2 || *b == 4) {
            return Err(TestCaseError::fail(format!(
                "beat count {} is not 1, 2 or 4 in {:?}", b, got)));
        }
    }
    Ok(())
}

pub fn run_property(runner: &mut TestRunner) -> Result<(), TestError<String>> {
    runner.run(&strategy(), property)
}
