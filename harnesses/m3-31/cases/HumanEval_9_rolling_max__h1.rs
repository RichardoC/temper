use proptest::prelude::*;
use proptest::test_runner::{TestCaseError, TestCaseResult, TestError, TestRunner};
use crate::candidate::rolling_max;

pub const STRATEGY_DESC: &str = "the documented example, monotone inputs (output equals input) and a decreasing run";
fn strategy() -> impl Strategy<Value = Vec<isize>> {
    prop_oneof![
        Just(vec![1isize, 2, 3, 2, 3, 4, 2]),
        proptest::collection::vec(0isize..10, 0..6)
            .prop_map(|mut v| { v.sort(); v }),
        proptest::collection::vec(0isize..10, 1..6)
            .prop_map(|mut v| { v.sort(); v.reverse(); v }),
    ]
}

fn property(numbers: Vec<isize>) -> TestCaseResult {
    let got = rolling_max(numbers.clone());
    if numbers == vec![1, 2, 3, 2, 3, 4, 2] {
        let want = vec![1isize, 2, 3, 3, 3, 4, 4];
        if got != want {
            return Err(TestCaseError::fail(format!(
                "documented example gave {:?} expected {:?}", got, want)));
        }
        return Ok(());
    }
    if numbers.is_empty() && !got.is_empty() {
        return Err(TestCaseError::fail(format!(
            "empty input gave {:?} expected empty", got)));
    }
    // sorted ascending input is its own rolling max
    if numbers.windows(2).all(|w| w[0] <= w[1]) && got != numbers {
        return Err(TestCaseError::fail(format!(
            "sorted input {:?} gave {:?}", numbers, got)));
    }
    // sorted descending input has every entry equal to the first
    if numbers.len() > 1 && numbers.windows(2).all(|w| w[0] >= w[1]) {
        if got.iter().any(|x| *x != numbers[0]) {
            return Err(TestCaseError::fail(format!(
                "descending input {:?} gave {:?} not all {}", numbers, got, numbers[0])));
        }
    }
    Ok(())
}

pub fn run_property(runner: &mut TestRunner) -> Result<(), TestError<Vec<isize>>> {
    runner.run(&strategy(), property)
}
