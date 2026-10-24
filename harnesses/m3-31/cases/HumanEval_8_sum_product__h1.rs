use proptest::prelude::*;
use proptest::test_runner::{TestCaseError, TestCaseResult, TestError, TestRunner};
use crate::candidate::sum_product;

pub const STRATEGY_DESC: &str = "the documented examples and a zero-in-vector product identity";
fn strategy() -> impl Strategy<Value = Vec<isize>> {
    prop_oneof![
        Just(vec![]),
        Just(vec![1isize, 2, 3, 4]),
        proptest::collection::vec(1isize..5, 0..5)
            .prop_map(|mut v| { v.push(0); v }),
    ]
}

fn property(numbers: Vec<isize>) -> TestCaseResult {
    let got = sum_product(numbers.clone());
    if numbers.is_empty() && got != (0, 1) {
        return Err(TestCaseError::fail(format!(
            "empty vector gave {:?} expected (0, 1)", got)));
    }
    if numbers == vec![1, 2, 3, 4] && got != (10, 24) {
        return Err(TestCaseError::fail(format!(
            "documented example gave {:?} expected (10, 24)", got)));
    }
    // any zero makes the product zero, and the product is then independent of the rest
    if numbers.contains(&0) && got.1 != 0 {
        return Err(TestCaseError::fail(format!(
            "{:?} contains a zero but product is {}", numbers, got.1)));
    }
    Ok(())
}

pub fn run_property(runner: &mut TestRunner) -> Result<(), TestError<Vec<isize>>> {
    runner.run(&strategy(), property)
}
