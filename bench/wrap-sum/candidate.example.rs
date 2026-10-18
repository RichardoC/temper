//! Reference (correct) implementation for bench/wrap-sum.

pub fn wrapping_sum(xs: &[u64]) -> u64 {
    xs.iter().fold(0u64, |acc, &x| acc.wrapping_add(x))
}
