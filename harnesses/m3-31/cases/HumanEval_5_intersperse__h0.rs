use proptest::prelude::*;
use proptest::test_runner::{TestCaseError, TestCaseResult, TestError, TestRunner};
use crate::candidate::intersperse;

pub const STRATEGY_DESC: &str = "vectors with a separator; length and interleaving are fully determined";
fn strategy() -> impl Strategy<Value = (Vec<isize>, isize)> {
    (proptest::collection::vec(-100isize..100, 0..8), -100isize..100)
}

fn property(input: (Vec<isize>, isize)) -> TestCaseResult {
    let (numbers, delim) = input;
    let got = intersperse(numbers.clone(), delim);
    let want_len = if numbers.len() <= 1 { numbers.len() } else { 2 * numbers.len() - 1 };
    if got.len() != want_len {
        return Err(TestCaseError::fail(format!(
            "intersperse({:?}, {}) has len {} expected {}",
            numbers, delim, got.len(), want_len)));
    }
    if numbers.len() > 1 {
        for (i, x) in numbers.iter().enumerate() {
            if got[2 * i] != *x {
                return Err(TestCaseError::fail(format!(
                    "element {} should be {} in {:?}", i, x, got)));
            }
            if i + 1 < numbers.len() && got[2 * i + 1] != delim {
                return Err(TestCaseError::fail(format!(
                    "separator between {} and {} should be {} in {:?}",
                    i, i + 1, delim, got)));
            }
        }
    } else if got != numbers {
        return Err(TestCaseError::fail(format!(
            "len<=1 input {:?} changed to {:?}", numbers, got)));
    }
    Ok(())
}

pub fn run_property(runner: &mut TestRunner) -> Result<(), TestError<(Vec<isize>, isize)>> {
    runner.run(&strategy(), property)
}
