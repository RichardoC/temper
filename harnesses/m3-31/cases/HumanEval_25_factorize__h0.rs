use proptest::prelude::*;
use proptest::test_runner::{TestCaseError, TestCaseResult, TestError, TestRunner};
use crate::candidate::factorize;

pub const STRATEGY_DESC: &str = "integers n >= 2; the factors multiply back to n, are ascending and are all prime";
fn strategy() -> impl Strategy<Value = isize> {
    2isize..3000
}

fn is_prime(m: isize) -> bool {
    if m < 2 { return false; }
    let mut d = 2;
    while d * d <= m {
        if m % d == 0 { return false; }
        d += 1;
    }
    true
}

fn property(n: isize) -> TestCaseResult {
    let got = factorize(n);
    let product: isize = got.iter().product();
    if product != n {
        return Err(TestCaseError::fail(format!(
            "factorize({}) = {:?} whose product is {} != n", n, got, product)));
    }
    for w in got.windows(2) {
        if w[0] > w[1] {
            return Err(TestCaseError::fail(format!(
                "factorize({}) = {:?} is not ascending", n, got)));
        }
    }
    for f in &got {
        if !is_prime(*f) {
            return Err(TestCaseError::fail(format!(
                "factorize({}) = {:?} contains non-prime {}", n, got, f)));
        }
    }
    Ok(())
}

pub fn run_property(runner: &mut TestRunner) -> Result<(), TestError<isize>> {
    runner.run(&strategy(), property)
}
