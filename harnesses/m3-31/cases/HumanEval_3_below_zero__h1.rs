use proptest::prelude::*;
use proptest::test_runner::{TestCaseError, TestCaseResult, TestError, TestRunner};
use crate::candidate::below_zero;

pub const STRATEGY_DESC: &str = "the documented examples, all-nonnegative vectors, and vectors that dip only in the middle";
fn strategy() -> impl Strategy<Value = Vec<isize>> {
    prop_oneof![
        Just(vec![1isize, 2, 3]),
        Just(vec![1isize, 2, -4, 5]),
        proptest::collection::vec(0isize..100, 0..8),
        proptest::collection::vec(0isize..50, 0..4)
            .prop_map(|mut v| { v.push(-1000); v.push(5); v }),
    ]
}

fn property(ops: Vec<isize>) -> TestCaseResult {
    let got = below_zero(ops.clone());
    match ops.as_slice() {
        [1, 2, 3] if got => {
            return Err(TestCaseError::fail("documented example [1,2,3] must be false"))
        }
        [1, 2, -4, 5] if !got => {
            return Err(TestCaseError::fail("documented example [1,2,-4,5] must be true"))
        }
        _ => {}
    }
    // all non-negative operations can never take the balance below zero
    if ops.iter().all(|x| *x >= 0) && got {
        return Err(TestCaseError::fail(format!(
            "{:?} has no negative operation yet returned true", ops)));
    }
    // an empty vector leaves the balance at zero, which is not below zero
    if ops.is_empty() && got {
        return Err(TestCaseError::fail("empty vector returned true"));
    }
    Ok(())
}

pub fn run_property(runner: &mut TestRunner) -> Result<(), TestError<Vec<isize>>> {
    runner.run(&strategy(), property)
}
