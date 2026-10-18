//! temper result protocol v1 harness.
//!
//! Runs the property defined in `properties.rs` under proptest 1.12.0's
//! `test_runner` APIs directly (no `proptest!` macro) and reports the outcome
//! as EXACTLY ONE JSON line on stdout, hand-serialized (no serde).
//!
//! Environment:
//!   TEMPER_SEED_HEX      optional. Either a proptest persisted-seed string
//!                        (e.g. `cc <64 hex chars>`, as reported by a previous
//!                        failing run) for seed-exact replay, or a plain hex
//!                        u64 for `RngSeed::Fixed`. Unset => random seed.
//!   TEMPER_CASES         optional, default 256. Successful cases required.
//!   TEMPER_TIMEOUT_MS    optional, default 2000. Echoed for the outer runner;
//!                        the per-case `timeout` Config field requires the
//!                        non-vendored `fork`/`timeout` features, so wall-clock
//!                        enforcement is done by `temper.harness.runner` via
//!                        subprocess timeout.
//!   TEMPER_STRATEGY_HASH optional. sha256 of properties.rs, echoed in output.
//!
//! Output protocol v1 (one line):
//!   pass : {"v":1,"status":"pass","successes":N,"strategy_hash":"H"}
//!   fail : adds "reason","minimal_input","seed_algorithm","seed",
//!          "successes","local_rejects","global_rejects"
//!   abort: {"v":1,"status":"abort","reason",...counts...} — generation gave
//!          up (over-constrained strategy), NOT a property violation.
//!   error: {"v":1,"status":"error","reason":"..."} harness-side problem.
//!
//! Exit codes: 0 pass, 1 fail, 2 abort, 3 error.

use std::collections::BTreeSet;
use std::env;
use std::process::exit;

use proptest::test_runner::{
    Config, MapFailurePersistence, PersistedSeed, RngAlgorithm, RngSeed, TestError, TestRunner,
};

mod candidate;
mod properties;

/// `source_file` key under which failing seeds are persisted into our
/// in-memory `MapFailurePersistence`. Must be `Some` — the map silently drops
/// seeds when `source_file` is `None`.
const SOURCE_KEY: &str = "temper";

/// Default max shrink iterations (bounded so a pathological shrink cannot
/// spin forever; still generous for collection strategies).
const MAX_SHRINK_ITERS: u32 = 10_000;

fn main() {
    exit(run());
}

fn run() -> i32 {
    // ---- environment ------------------------------------------------------
    let strategy_hash = env::var("TEMPER_STRATEGY_HASH").unwrap_or_default();
    let cases: u32 = match env::var("TEMPER_CASES") {
        Ok(s) => match s.trim().parse() {
            Ok(n) => n,
            Err(_) => return emit_error(&format!("TEMPER_CASES not a u32: {s:?}")),
        },
        Err(_) => 256,
    };
    let timeout_ms: u32 = match env::var("TEMPER_TIMEOUT_MS") {
        Ok(s) => match s.trim().parse() {
            Ok(n) => n,
            Err(_) => return emit_error(&format!("TEMPER_TIMEOUT_MS not a u32: {s:?}")),
        },
        Err(_) => 2000,
    };
    let seed_hex = env::var("TEMPER_SEED_HEX").ok();
    eprintln!(
        "temper-harness: cases={cases} timeout_ms={timeout_ms} (wall-clock enforced by outer runner) strategy_hash={strategy_hash:?}"
    );

    // ---- configuration ----------------------------------------------------
    let mut config = Config::default();
    config.cases = cases;
    config.max_shrink_iters = MAX_SHRINK_ITERS;
    config.rng_algorithm = RngAlgorithm::ChaCha;
    config.source_file = Some(SOURCE_KEY);
    config.result_cache = proptest::test_runner::noop_result_cache;

    let mut mfp = MapFailurePersistence::default();
    match seed_hex.as_deref().map(str::trim).filter(|s| !s.is_empty()) {
        None => {} // random seed
        Some(s) => match s.split_once(' ') {
            // Persisted-seed form: replay through proptest's native
            // persisted-failure path (the runner re-seeds directly, reproducing
            // the exact failing input before any fresh case is generated).
            Some((_alg_key, _)) => match s.parse::<PersistedSeed>() {
                Ok(seed) => {
                    mfp.map
                        .entry(SOURCE_KEY)
                        .or_insert_with(BTreeSet::new)
                        .insert(seed);
                }
                Err(_) => {
                    return emit_error(&format!(
                        "TEMPER_SEED_HEX looks like a persisted seed but does not parse: {s:?}"
                    ))
                }
            },
            // Plain hex u64 form: fixed RNG seed (deterministic, but does NOT
            // reproduce a previously persisted failing input bit-for-bit).
            None => match u64::from_str_radix(s.trim_start_matches("0x"), 16) {
                Ok(v) => config.rng_seed = RngSeed::Fixed(v),
                Err(_) => {
                    return emit_error(&format!(
                        "TEMPER_SEED_HEX not a persisted seed nor hex u64: {s:?}"
                    ))
                }
            },
        },
    }
    config.failure_persistence = Some(Box::new(mfp));

    // ---- run --------------------------------------------------------------
    let mut runner = TestRunner::new(config);
    let result = properties::run_property(&mut runner);

    // Counters are private in proptest 1.12.0; the public Display impl is the
    // sanctioned way to read them outside the crate.
    let (successes, local_rejects, global_rejects) =
        parse_counts(&format!("{}", runner));
    let seed = persisted_seed_of(&runner);

    match result {
        Ok(()) => {
            println!(
                "{{\"v\":1,\"status\":\"pass\",\"successes\":{},\"strategy_hash\":\"{}\"}}",
                successes,
                json_escape(&strategy_hash)
            );
            0
        }
        Err(TestError::Fail(reason, minimal)) => {
            let (seed_alg, seed_str) = match &seed {
                Some(s) => (s.split(' ').next().unwrap_or(""), s.as_str()),
                None => ("", ""),
            };
            println!(
                concat!(
                    "{{\"v\":1,\"status\":\"fail\",",
                    "\"reason\":\"{r}\",",
                    "\"minimal_input\":\"{m}\",",
                    "\"seed_algorithm\":\"{sa}\",",
                    "\"seed\":\"{sd}\",",
                    "\"successes\":{s},\"local_rejects\":{lr},\"global_rejects\":{gr},",
                    "\"strategy_hash\":\"{sh}\"}}"
                ),
                r = json_escape(&reason.to_string()),
                m = json_escape(&format!("{:?}", minimal)),
                sa = json_escape(seed_alg),
                sd = json_escape(seed_str),
                s = successes,
                lr = local_rejects,
                gr = global_rejects,
                sh = json_escape(&strategy_hash),
            );
            1
        }
        Err(TestError::Abort(reason)) => {
            println!(
                concat!(
                    "{{\"v\":1,\"status\":\"abort\",",
                    "\"reason\":\"{r}\",",
                    "\"successes\":{s},\"local_rejects\":{lr},\"global_rejects\":{gr},",
                    "\"strategy_hash\":\"{sh}\"}}"
                ),
                r = json_escape(&reason.to_string()),
                s = successes,
                lr = local_rejects,
                gr = global_rejects,
                sh = json_escape(&strategy_hash),
            );
            2
        }
    }
}

/// Emit an `error` protocol line and return the harness-error exit code.
fn emit_error(reason: &str) -> i32 {
    eprintln!("temper-harness: error: {}", reason);
    println!(
        "{{\"v\":1,\"status\":\"error\",\"reason\":\"{}\"}}",
        json_escape(reason)
    );
    3
}

/// Recover the persisted failing seed (if any) through the public downcast
/// hook of the `FailurePersistence` trait.
fn persisted_seed_of(runner: &TestRunner) -> Option<String> {
    let fp = runner.config().failure_persistence.as_ref()?;
    let mfp = fp.as_any().downcast_ref::<MapFailurePersistence>()?;
    mfp.map
        .values()
        .flat_map(|set| set.iter())
        .last()
        .map(|seed| seed.to_string()) // PersistedSeed Display == persistence format
}

/// Parse the success/local-reject/global-reject counters out of the
/// `Display` implementation of `TestRunner` (fields are private otherwise).
fn parse_counts(display: &str) -> (u32, u32, u32) {
    fn grab(s: &str, key: &str) -> u32 {
        s.lines()
            .find_map(|line| line.trim().strip_prefix(key))
            .and_then(|rest| rest.trim().parse().ok())
            .unwrap_or(0)
    }
    (
        grab(display, "successes:"),
        grab(display, "local rejects:"),
        grab(display, "global rejects:"),
    )
}

/// Escape a string for embedding in a JSON string literal (RFC 8259 subset:
/// quotes, backslash, and control characters).
fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}
