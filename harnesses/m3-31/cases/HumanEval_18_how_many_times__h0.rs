use proptest::prelude::*;
use proptest::test_runner::{TestCaseError, TestCaseResult, TestError, TestRunner};
use crate::candidate::how_many_times;

pub const STRATEGY_DESC: &str = "overlapping occurrence counting via an independent sliding-window scan";
fn strategy() -> impl Strategy<Value = (String, String)> {
    ("[ab]{0,10}", "[ab]{1,3}")
}

fn property(input: (String, String)) -> TestCaseResult {
    let (s, sub) = input;
    let got = how_many_times(s.clone(), sub.clone());
    let sc: Vec<char> = s.chars().collect();
    let uc: Vec<char> = sub.chars().collect();
    let mut want = 0isize;
    if uc.len() <= sc.len() {
        for i in 0..=(sc.len() - uc.len()) {
            if sc[i..i + uc.len()] == uc[..] {
                want += 1;
            }
        }
    }
    if got != want {
        return Err(TestCaseError::fail(format!(
            "how_many_times({:?}, {:?}) = {} expected {} (overlapping)",
            s, sub, got, want)));
    }
    Ok(())
}

pub fn run_property(runner: &mut TestRunner) -> Result<(), TestError<(String, String)>> {
    runner.run(&strategy(), property)
}
