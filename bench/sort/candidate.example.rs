//! Reference (correct) implementation for bench/sort.

pub fn sort_i32(v: &mut Vec<i32>) {
    v.sort_unstable();
}
