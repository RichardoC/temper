//! Candidate code under test.
//!
//! PLACEHOLDER: `temper.harness.emit` overwrites this file with the
//! LM-generated candidate. The boundary contract is only the set of
//! `pub` function signatures referenced by `properties.rs`; everything
//! else here is free for the emitter to replace.

/// Sorts `v` in ascending order.
///
/// The placeholder is a known-good reference implementation so that the
/// template builds and passes its own property before emission.
pub fn candidate_sort(v: &mut Vec<i32>) {
    v.sort_unstable();
}
