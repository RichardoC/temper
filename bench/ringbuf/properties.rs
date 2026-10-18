//! Human-authored property for bench/ringbuf (parent-authored, APPROVED).
//! Value type (usize, Vec<i64>): capacity + push sequence.

use crate::candidate::{ring_new, ring_pop, ring_push};
use proptest::prelude::*;
use proptest::test_runner::{TestCaseError, TestCaseResult, TestError, TestRunner};

pub const STRATEGY_DESC: &str = "(1usize..=8, vec(any::<i64>(), 0..24))";

fn strategy() -> impl Strategy<Value = (usize, Vec<i64>)> {
    (1usize..=8, proptest::collection::vec(any::<i64>(), 0..24))
}

fn property((cap, vals): (usize, Vec<i64>)) -> TestCaseResult {
    // After pushing n values into a capacity-c buffer, the buffer must hold
    // the LAST min(n, c) values and pop them in FIFO order; draining must
    // end empty; popping an empty buffer must be None.
    let keep = vals.len().min(cap);
    let want: Vec<i64> = vals[vals.len() - keep..].to_vec();

    let mut rb = ring_new(cap);
    if let Some(x) = ring_pop(&mut rb) {
        return Err(TestCaseError::fail(format!(
            "pop on empty buffer returned Some({x})"
        )));
    }

    for &v in &vals {
        ring_push(&mut rb, v);
    }

    for (i, &w) in want.iter().enumerate() {
        match ring_pop(&mut rb) {
            Some(got) if got == w => {}
            other => {
                return Err(TestCaseError::fail(format!(
                    "FIFO order violated at pop {i}: got {other:?}, want Some({w}) (cap={cap}, vals={vals:?})"
                )))
            }
        }
    }

    if let Some(x) = ring_pop(&mut rb) {
        return Err(TestCaseError::fail(format!(
            "buffer not empty after draining: Some({x})"
        )));
    }
    Ok(())
}

pub fn run_property(runner: &mut TestRunner) -> Result<(), TestError<(usize, Vec<i64>)>> {
    runner.run(&strategy(), property)
}
