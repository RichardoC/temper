use proptest::prelude::*;
use proptest::test_runner::{TestCaseError, TestCaseResult, TestError, TestRunner};
use crate::candidate::sort_numbers;

pub const STRATEGY_DESC: &str = "space-delimited number words; the output is the same multiset in numeric order";
fn strategy() -> impl Strategy<Value = String> {
    proptest::collection::vec(
        proptest::sample::select(vec![
            "zero", "one", "two", "three", "four",
            "five", "six", "seven", "eight", "nine",
        ]), 1..6usize)
        .prop_map(|w| w.join(" "))
}

fn property(s: String) -> TestCaseResult {
    let got = sort_numbers(s.clone());
    let value = |w: &str| -> isize {
        match w {
            "zero" => 0, "one" => 1, "two" => 2, "three" => 3, "four" => 4,
            "five" => 5, "six" => 6, "seven" => 7, "eight" => 8, "nine" => 9,
            _ => -1,
        }
    };
    let mut want: Vec<&str> = s.split_whitespace().collect();
    want.sort_by_key(|w| value(w));
    let got_words: Vec<&str> = got.split_whitespace().collect();
    if got_words != want {
        return Err(TestCaseError::fail(format!(
            "sort_numbers({:?}) = {:?} expected {:?}", s, got, want)));
    }
    Ok(())
}

pub fn run_property(runner: &mut TestRunner) -> Result<(), TestError<String>> {
    runner.run(&strategy(), property)
}
