use proptest::prelude::*;
use proptest::test_runner::{TestCaseError, TestCaseResult, TestError, TestRunner};
use crate::candidate::below_zero;

pub const STRATEGY_DESC: &str = "operation vectors; the running-prefix minimum decides the documented answer";
fn strategy() -> impl Strategy<Value = Vec<isize>> {
    proptest::collection::vec(-100isize..100, 0..10)
}

fn property(ops: Vec<isize>) -> TestCaseResult {
    let got = below_zero(ops.clone());
    let mut balance: isize = 0;
    let mut went_negative = false;
    for op in &ops {
        balance += *op;
        if balance < 0 { went_negative = true; }
    }
    if got == went_negative {
        Ok(())
    } else {
        Err(TestCaseError::fail(format!(
            "below_zero({:?}) = {} but running balance went negative = {}",
            ops, got, went_negative)))
    }
}

pub fn run_property(runner: &mut TestRunner) -> Result<(), TestError<Vec<isize>>> {
    runner.run(&strategy(), property)
}
