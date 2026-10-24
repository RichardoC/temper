use proptest::prelude::*;
use proptest::test_runner::{TestCaseError, TestCaseResult, TestError, TestRunner};
use crate::candidate::factorize;

pub const STRATEGY_DESC: &str = "the documented examples plus the count-of-a-prime-power relation";
fn strategy() -> impl Strategy<Value = isize> {
    prop_oneof![
        Just(8isize),
        Just(25isize),
        Just(70isize),
        (2isize..13).prop_map(|p| p * p),
        (2isize..7).prop_map(|p| p * p * p),
    ]
}

fn property(n: isize) -> TestCaseResult {
    let got = factorize(n);
    let documented: Option<Vec<isize>> = match n {
        8 => Some(vec![2, 2, 2]),
        25 => Some(vec![5, 5]),
        70 => Some(vec![2, 5, 7]),
        _ => None,
    };
    if let Some(want) = documented {
        if got != want {
            return Err(TestCaseError::fail(format!(
                "documented example factorize({}) = {:?} expected {:?}",
                n, got, want)));
        }
        return Ok(());
    }
    // a perfect square p*p factorizes to exactly [p, p]
    let mut r = 2;
    while r * r <= n {
        if r * r == n {
            if got != vec![r, r] {
                return Err(TestCaseError::fail(format!(
                    "factorize({}) = {:?} expected [{}, {}]", n, got, r, r)));
            }
        }
        r += 1;
    }
    // a perfect cube p*p*p factorizes to exactly [p, p, p]
    let mut c = 2;
    while c * c * c <= n {
        if c * c * c == n {
            if got != vec![c, c, c] {
                return Err(TestCaseError::fail(format!(
                    "factorize({}) = {:?} expected [{}, {}, {}]", n, got, c, c, c)));
            }
        }
        c += 1;
    }
    Ok(())
}

pub fn run_property(runner: &mut TestRunner) -> Result<(), TestError<isize>> {
    runner.run(&strategy(), property)
}
