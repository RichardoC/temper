use proptest::prelude::*;
use proptest::test_runner::{TestCaseError, TestCaseResult, TestError, TestRunner};
use crate::candidate::rolling_max;

pub const STRATEGY_DESC: &str = "integer vectors; each output is the prefix maximum and the length is preserved";
fn strategy() -> impl Strategy<Value = Vec<isize>> {
    proptest::collection::vec(-1000isize..1000, 0..10)
}

fn property(numbers: Vec<isize>) -> TestCaseResult {
    let got = rolling_max(numbers.clone());
    if got.len() != numbers.len() {
        return Err(TestCaseError::fail(format!(
            "rolling_max({:?}) has len {} expected {}",
            numbers, got.len(), numbers.len())));
    }
    let mut running = isize::MIN;
    for (i, x) in numbers.iter().enumerate() {
        running = running.max(*x);
        if got[i] != running {
            return Err(TestCaseError::fail(format!(
                "rolling_max({:?})[{}] = {} expected prefix max {}",
                numbers, i, got[i], running)));
        }
    }
    Ok(())
}

pub fn run_property(runner: &mut TestRunner) -> Result<(), TestError<Vec<isize>>> {
    runner.run(&strategy(), property)
}
