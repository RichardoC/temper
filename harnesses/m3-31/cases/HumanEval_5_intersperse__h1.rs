use proptest::prelude::*;
use proptest::test_runner::{TestCaseError, TestCaseResult, TestError, TestRunner};
use crate::candidate::intersperse;

pub const STRATEGY_DESC: &str = "the documented examples and the removal-of-separators round trip";
fn strategy() -> impl Strategy<Value = (Vec<isize>, isize)> {
    prop_oneof![
        Just((vec![], 4isize)),
        Just((vec![1isize, 2, 3], 4isize)),
        (proptest::collection::vec(0isize..10, 0..5), 100isize..200),
    ]
}

fn property(input: (Vec<isize>, isize)) -> TestCaseResult {
    let (numbers, delim) = input;
    let got = intersperse(numbers.clone(), delim);
    if numbers.is_empty() {
        if !got.is_empty() {
            return Err(TestCaseError::fail(format!(
                "empty input gave {:?} expected empty", got)));
        }
        return Ok(());
    }
    if numbers == vec![1, 2, 3] && delim == 4 {
        let want = vec![1isize, 4, 2, 4, 3];
        if got != want {
            return Err(TestCaseError::fail(format!(
                "documented example gave {:?} expected {:?}", got, want)));
        }
        return Ok(());
    }
    // dropping the separators must recover the input exactly
    let recovered: Vec<isize> = got.iter().copied().filter(|x| *x != delim).collect();
    if delim < 0 || delim > 9 || !numbers.contains(&delim) {
        if recovered != numbers {
            return Err(TestCaseError::fail(format!(
                "separator removal gave {:?} not {:?}", recovered, numbers)));
        }
    }
    Ok(())
}

pub fn run_property(runner: &mut TestRunner) -> Result<(), TestError<(Vec<isize>, isize)>> {
    runner.run(&strategy(), property)
}
