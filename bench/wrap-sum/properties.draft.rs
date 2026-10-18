//! Property-based tests for `crate::candidate::wrapping_sum`.
//!
//! Specification under test: `wrapping_sum(xs)` returns the sum of all
//! elements of `xs` reduced modulo 2^64 (i.e. computed with wrapping u64
//! arithmetic), never panics, and maps the empty slice to 0.

use crate::candidate::wrapping_sum;
use proptest::collection::vec;
use proptest::prelude::*;
use proptest::test_runner::{TestCaseError, TestCaseResult, TestError, TestRunner};
use std::panic::{catch_unwind, AssertUnwindSafe};

/// Human-readable description of the input strategy used by `run_property`.
pub const STRATEGY_DESC: &str = "Weighted union (prop_oneof!) of Vec<u64> generators: \
    weight 1: the empty vector; \
    weight 2: singleton vectors of arbitrary u64; \
    weight 4: length 2..=16 vectors of boundary values {0, 1, u64::MAX, \
    u64::MAX-1, u64::MAX/2, u64::MAX/2+1, 2^63, 2^63-1} for dense wrap-around; \
    weight 2: two-element vectors [a, u64::MAX], wrapping for every a != 0; \
    weight 2: all-u64::MAX vectors of length 1..=33; \
    weight 6: length 0..=128 vectors of boundary-or-arbitrary u64 elements; \
    weight 2: length 129..=512 vectors of the same elements.";

/// Element strategy restricted to values on the wrap-around boundaries, where
/// partial sums are most likely to cross a multiple of 2^64.
fn boundary_elem() -> impl Strategy<Value = u64> {
    prop_oneof![
        1 => Just(0u64),
        1 => Just(1u64),
        1 => Just(u64::MAX),
        1 => Just(u64::MAX - 1),
        1 => Just(u64::MAX / 2),
        1 => Just(u64::MAX / 2 + 1),
        1 => Just(1u64 << 63),
        1 => Just((1u64 << 63) - 1),
    ]
}

/// Element strategy mixing fully arbitrary u64s with boundary values.
fn mixed_elem() -> impl Strategy<Value = u64> {
    prop_oneof![
        3 => boundary_elem(),
        4 => any::<u64>(),
    ]
}

/// Input strategy stressing `wrapping_sum` with empty, singleton, wrap-dense,
/// guaranteed-overflow, all-max, general, and long slices.
fn strategy() -> impl Strategy<Value = Vec<u64>> {
    prop_oneof![
        // Empty slice: must sum to 0.
        1 => Just(Vec::<u64>::new()),
        // Singleton slices.
        2 => vec(any::<u64>(), 1..=1),
        // Small slices dense with boundary values (wrap-around pressure).
        4 => vec(boundary_elem(), 2..=16),
        // Two-element slices [a, u64::MAX]: wrap for every a != 0.
        2 => any::<u64>().prop_map(|a| vec![a, u64::MAX]),
        // Slices consisting solely of u64::MAX.
        2 => vec(Just(u64::MAX), 1..=33),
        // General arbitrary content and length.
        6 => vec(mixed_elem(), 0..=128),
        // Longer slices to exercise sustained accumulation.
        2 => vec(mixed_elem(), 129..=512),
    ]
}

/// The property: `wrapping_sum(xs)` equals the exact mathematical sum of the
/// elements reduced modulo 2^64, and it never panics.
fn property(xs: Vec<u64>) -> TestCaseResult {
    // Independent oracle: accumulate exactly in u128, then keep the low 64
    // bits. A u128 accumulator cannot itself overflow short of ~2^64 maximal
    // elements, so the oracle needs no wrapping arithmetic of its own.
    let exact: u128 = xs.iter().map(|&x| u128::from(x)).sum();
    let expected = exact as u64; // truncation == reduction modulo 2^64

    // The specification forbids panics. Catch any unwind (the canonical bug
    // is plain `+`, which panics on overflow in debug builds) so that a panic
    // is reported as a readable, shrinkable property failure.
    let got = match catch_unwind(AssertUnwindSafe(|| wrapping_sum(&xs))) {
        Ok(value) => value,
        Err(payload) => {
            let panic_msg = payload
                .downcast_ref::<&str>()
                .copied()
                .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
                .unwrap_or("<non-string panic payload>");
            return Err(TestCaseError::fail(format!(
                "wrapping_sum panicked on input {:?} with message {:?} \
                 (exact sum {}); the specification requires wrapping \
                 arithmetic and forbids panics",
                xs, panic_msg, exact
            )));
        }
    };

    if got != expected {
        return Err(TestCaseError::fail(format!(
            "wrapping_sum({:?}) returned {}, but the sum modulo 2^64 is {} \
             (exact sum {})",
            xs, got, expected, exact
        )));
    }

    Ok(())
}

/// Stable entry point: run the property against the candidate with the given
/// runner.
pub fn run_property(runner: &mut TestRunner) -> Result<(), TestError<Vec<u64>>> {
    runner.run(&strategy(), property)
}
