use proptest::prelude::*;
use proptest::test_runner::{TestCaseError, TestCaseResult, TestError, TestRunner};
use crate::candidate::sum_product;

pub const STRATEGY_DESC: &str = "integer vectors; sum and product with the documented empty-vector identities";
fn strategy() -> impl Strategy<Value = Vec<isize>> {
    proptest::collection::vec(-6isize..6, 0..6)
}

fn property(numbers: Vec<isize>) -> TestCaseResult {
    let got = sum_product(numbers.clone());
    let want_sum: isize = numbers.iter().sum();
    let want_prod: isize = numbers.iter().product();
    if got != (want_sum, want_prod) {
        return Err(TestCaseError::fail(format!(
            "sum_product({:?}) = {:?} expected {:?}",
            numbers, got, (want_sum, want_prod))));
    }
    Ok(())
}

pub fn run_property(runner: &mut TestRunner) -> Result<(), TestError<Vec<isize>>> {
    runner.run(&strategy(), property)
}
