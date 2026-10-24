use proptest::prelude::*;
use proptest::test_runner::{TestCaseError, TestCaseResult, TestError, TestRunner};
use crate::candidate::longest;

pub const STRATEGY_DESC: &str = "the documented examples and the first-on-tie rule made explicit";
fn strategy() -> impl Strategy<Value = Vec<String>> {
    prop_oneof![
        Just(vec![]),
        Just(vec![String::from("a"), String::from("b"), String::from("c")]),
        Just(vec![String::from("a"), String::from("bb"), String::from("ccc")]),
        proptest::collection::vec("[xy]{1,4}", 1..6),
    ]
}

fn property(strings: Vec<String>) -> TestCaseResult {
    let got = longest(strings.clone());
    if strings.is_empty() && got.is_some() {
        return Err(TestCaseError::fail("empty vector must give None"));
    }
    let all_same_len = !strings.is_empty()
        && strings.windows(2).all(|w| w[0].len() == w[1].len());
    if all_same_len && got != Some(strings[0].clone()) {
        return Err(TestCaseError::fail(format!(
            "all lengths equal, expected the FIRST {:?} but got {:?}",
            strings[0], got)));
    }
    if let Some(ref s) = got {
        let best = strings.iter().map(|x| x.len()).max().unwrap();
        if s.len() != best {
            return Err(TestCaseError::fail(format!(
                "returned {:?} (len {}) but the longest is {}", s, s.len(), best)));
        }
    }
    Ok(())
}

pub fn run_property(runner: &mut TestRunner) -> Result<(), TestError<Vec<String>>> {
    runner.run(&strategy(), property)
}
