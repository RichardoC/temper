//! MultiPL-E-container scoring tests for bench/sort (M2.0).
//!
//! Deterministic port of the APPROVED held-in property
//! (bench/sort/properties.rs: `sort_i32(v) == sorted(v)`, multiset preserved)
//! to fixed cases, because the scoring container is offline and carries no
//! proptest. properties.rs remains the held-in verifier for temper's own
//! loop; this file only feeds the external scorer and MUST assert the same
//! contract. The first two data cases are the shrunk failing seeds recorded
//! in bench/sort/corpus (runs 20261018T203659Z p0/p2).

fn main() {
    let cases: Vec<Vec<i32>> = vec![
        vec![],
        vec![0],
        vec![1, 2, 3],
        vec![3, 2, 1],
        vec![279524015, 0],                 // shrunk failing seed (p0)
        vec![0, -1],                        // shrunk failing seed (p2)
        vec![5, -1, 5, 0, -1, 5],           // duplicates + negatives
        vec![-2147483648, 2147483647, 0],   // i32 extremes
        vec![9, 8, 7, 6, 5, 4, 3, 2, 1, 0], // reverse-sorted
    ];
    for v in cases {
        let mut got = v.clone();
        sort_i32(&mut got);
        let mut want = v.clone();
        want.sort_unstable();
        assert_eq!(got, want, "sort_i32({:?}) -> {:?}, want {:?}", v, got, want);
    }
}
