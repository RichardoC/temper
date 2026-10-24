# temper — M0 report

Milestone: **M0 — Rust execution harness** (no GPU/model code).
Machine: M3 Max, macOS (arm64), rustup shims at `/opt/homebrew/opt/rustup/bin`,
python3 `/usr/bin/python3` (stdlib only), hyperfine 2.0.0, sandbox-exec verified.

## What was built

### Inherited (verified, not redone — commits `1a9543b`, `00e52de`)
- `rust-template/`: candidate-crate template pinned to proptest `=1.12.0`
  (vendored, committed `vendor/`, 11 MB), offline source-replacement in
  `.cargo/config.toml`, warm `target/`. `cargo build --offline` succeeds as
  committed (0.1 s warm). **No fixup commit was needed.**
- `rust-template/src/main.rs` implements **temper result protocol v1** exactly
  as specced: env `TEMPER_SEED_HEX` (persisted-seed `cc <64 hex>` form via
  `PersistedSeed` parse, or plain hex u64 → `RngSeed::Fixed`), `TEMPER_CASES`
  (default 256), `TEMPER_TIMEOUT_MS` (default 2000, echoed; wall clock enforced
  by the outer runner since proptest's per-case `timeout` needs non-vendored
  features), `TEMPER_STRATEGY_HASH`; `TestRunner::run` used directly (no
  `proptest!` macro); `MapFailurePersistence` with `source_file = Some("temper")`
  (required — the map silently drops seeds when `source_file` is `None`);
  exactly ONE hand-serialized JSON line on stdout with `v/status
  pass|fail|abort`, `minimal_input` + `seed` on fail, counts, `strategy_hash`.
  Verified live: pass (exit 0), fail (panic caught by proptest, shrunk to
  `[0]`, exit 1, seed reported), abort unexercised (no over-constrained
  strategy in scope), seed-exact replay reproduces the identical minimal input.

### New in this session
- **`47264cc` — `temper/` python harness package**
  - `harness/emit.py` — copies template layout (Cargo.toml, Cargo.lock,
    `.cargo/`, `vendor/`, `src/`; never `target/`) into reused workdir
    `temper/.work/gen/`, writes `candidate.rs` + `properties.rs`,
    `strategy_hash = sha256(properties.rs)`.
  - `harness/cache.py` — hardlink registry-dep artifacts (`deps/`,
    `.fingerprint/`, `build/`) from warm `rust-template/target` into
    `.work/gen/target` via `os.link`, mtimes preserved with `os.utime`
    (copy fallback if cross-device). 143 links per fresh emit.
  - `harness/sandbox.py` — backends `seatbelt` (DEFAULT on macOS) / `bwrap`
    (untested, Linux parity) / `none` (passthrough + loud warning). Selector
    env `TEMPER_SANDBOX`. ONE `sandbox-exec -f <profile> -D WORKDIR=<abs>`
    wraps each whole cargo call and the test binary (inheritance covers rustc,
    build scripts, test binary). Profile: deny-default; allows
    `file-map-executable` (global — dyld breaks without it),
    `process-exec*`, `process-fork`, `process-info*`, `sysctl-read`,
    `system-info`; reads: `/usr /System /Library /private/etc /dev`,
    `/opt/homebrew` (shims), `~/.rustup` + `~/.cargo` (toolchain/config),
    `/private/tmp`, SDK path via `xcrun --show-sdk-path`, workdir, plus
    workdir **ancestor dirs**; writes: workdir + `/private/tmp` + `/dev/null`
    (+ `/dev/dtracehelper` for cc); `deny network*`; mach-lookup limited to
    notification/logger services (deliberately NOT
    `com.apple.dt.CommandLineTools.installondemand` — no tool installs).
  - `harness/runner.py` — emit → `cargo check --offline
    --message-format=json` (cheap reject; build skipped on check failure) →
    `cargo build --offline --message-format=json-diagnostic-rendered-ansi` →
    run `target/debug/harness` (subprocess timeout = `TEMPER_TIMEOUT_MS` +
    15 s margin, whole process group SIGKILLed on expiry) → parse LAST
    protocol-v1 JSON line. Dataclasses `CompileResult`
    (ok|compile_error|timeout|error), `TestResult`
    (pass|fail|abort|error|timeout), `RunOutcome`. `find_cargo()` knows the
    homebrew rustup shim location. Child env gets `TMPDIR=<workdir>/.tmp` so
    rustc temp files stay inside the sandbox-writable workdir.
  - `feedback/diagnostics.py` — LENIENT cargo-JSON parser → `Diagnostic`
    (`code` from `code.code`, `level`, `message`, `rendered`), `Span`
    (`byte_start`/`byte_end`, `line_*`, `is_primary`, `label`),
    `Suggestion` (`suggested_replacement`, `suggestion_applicability`).
    Unknown fields/variants ignored; non-JSON lines skipped; ICE
    ("internal compiler error") surfaced distinctly via `Diagnostic.is_ice`.
- **`c8c2c12` — `scripts/demo_m0.py`**: 4-step acceptance demo, all seatbelt.
- **This commit — `scripts/measure_cycle.sh`, LICENSE, README.md,
  `.gitignore`** (target/, .work/ ignored; vendor/ committed), this report.
  `.gitignore` already covered target/.work from the checkpoint; vendor/
  untouched.

## Exact demo output (`python3 scripts/demo_m0.py`, exit 0)

```
temper M0 acceptance demo — sandbox backend: seatbelt

=== step 1: buggy candidate -> fail + reel seed + minimal_input (sandboxed) ===
raw protocol line: {'v': 1, 'status': 'fail', 'reason': 'index out of bounds: the len is 1 but the index is 1', 'minimal_input': '[0]', 'seed_algorithm': 'cc', 'seed': 'cc 6a8ee9b4c07a7f4d1eb44a62050e7a62d2e60df637595dea59836d4368615ea9', 'successes': 1, 'local_rejects': 0, 'global_rejects': 0, 'strategy_hash': '7f7407bab33757cfccf8454dee0876b7b5aa6dd6b0da7a32234bf6ee505f7beb'}
STEP 1 PASS: status=fail seed=cc 6a8ee9b4c07a7f4d1eb44a62050e7a62d2e60df637595dea59836d4368615ea9 minimal_input=[0]

=== step 2: re-run with TEMPER_SEED_HEX=<seed> -> same minimal_input ===
STEP 2 PASS: replay minimal_input=[0] (expected [0])

=== step 3: fixed candidate -> pass (sandboxed) ===
STEP 3 PASS: status=pass report={'v': 1, 'status': 'pass', 'successes': 256, 'strategy_hash': '7f7407bab33757cfccf8454dee0876b7b5aa6dd6b0da7a32234bf6ee505f7beb'}

=== step 4: broken candidate -> structured compile diagnostics (run skipped) ===
first error: code=E0308 message='mismatched types'
  spans: [('src/candidate.rs', 134, 141, True), ('src/candidate.rs', 125, 131, False)]
STEP 4 PASS: compile_status=compile_error errors=2 run_attempted=False

M0 DEMO: ALL STEPS PASS
```

Scenario details: buggy candidate = selection sort scanning `j in i+1..=len`
and indexing `v[j]` → panics for any non-empty input; property =
`candidate_sort(v) == std sorted v` over `Vec<i32>` len 0..64; fixed candidate
flips the bound to `..len`; broken candidate assigns `usize` to `String`
(E0308) — compile_error with code + primary byte spans, run NOT attempted.
(The spec's example bug shape `[1..=len]` was realized as the equivalent
off-by-one `..=len` scan bound; same bug class, deterministic shrink to `[0]`.)

## Cycle measurement (hyperfine 2.0.0, `scripts/measure_cycle.sh`)

Command per run: `touch src/candidate.rs && <mode> cargo build --offline` in
the emitted demo crate (`temper/.work/gen`), dep artifacts hardlink-primed
(143 links), 3 warmup runs each:

| mode     | mean            | σ        | min … max           | runs |
|----------|-----------------|----------|---------------------|------|
| seatbelt | **236.4 ms**    | ± 21.4 ms| 221.5 … 297.1 ms    | 11   |
| none     | **215.5 ms**    | ± 8.4 ms | 205.1 … 234.8 ms    | 13   |

Sandbox overhead ≈ 21 ms (~10%) for whole-build seatbelt wrapping — cheap
enough to keep on for every cycle. (The first hyperfine attempt measured the
wrong thing: hyperfine 2.0.0 parses `-S sh` as flag `sh` + a `sh` *command*;
fixed with `--shell=sh`.)

## Deviations from spec / caveats

1. **Seatbelt needed more than the specced allowlist.** Bisected via
   `log show --predicate …deny…`: (a) cargo walks workdir **ancestors** to
   `/` for workspace/`.cargo/config.toml` discovery → ancestor dirs need
   read; (b) `cc` is the xcode-select shim → needs metadata on `/var` +
   reads of `/private/var/select` and `/private/var/db/xcode_select_link`;
   (c) `cc` writes `/dev/dtracehelper`; (d) `mach-lookup` for
   notification_center/logger/logd; (e) `system-info` (statfs) for
   rustc/cargo. All resolved; sandbox never disabled for the demo.
   Cosmetic residue: xcrun cache-file warnings under sandbox (falls back
   gracefully; builds succeed).
2. **TMPDIR redirected into the workdir** for sandboxed runs (rustc temp
   files) — `/private/tmp` stays allowed per spec.
3. **hyperfine 2.0.0** requires `--shell=sh` (not `-S sh`) for compound
   commands and label handling differs from 1.x (see above).
4. cache.py falls back to copying if hardlinks are unsupported; primary path
   is `os.link` + `os.utime` per spec.
5. `abort` protocol path (generation gave up) is implemented and parsed but
   not exercised by the demo — no over-constrained strategy in scope.
6. This report is mirrored at the session artifact path
   `~/.pi/agent/sessions/--Users-richardf--/subagent-artifacts/outputs/
   d22f3804-ba9d-4507-a142-f101604b97e9/temper/docs/m0-report.md`
   (authoritative for this run).

## Real proptest 1.12.0 API notes (from `vendor/proptest/src/`)

- `TestRunner::new(Config)` + `runner.run(&strategy, prop_fn)` →
  `Result<(), TestError<T>>`; `TestError::Fail(reason, minimal_input)` /
  `TestError::Abort(reason)`.
- Panics inside test cases are caught (with `std`) and reported as
  `TestError::Fail` with the panic message; shrinking proceeds across panics
  (our OOB bug shrank to `[0]`).
- `MapFailurePersistence` is `BTreeMap<source_file, BTreeSet<PersistedSeed>>`;
  seeds are dropped unless `Config::source_file` is `Some`. Read-back via the
  trait's `as_any` downcast hook; `PersistedSeed` Display/parsing round-trips
  the `cc <64 hex>` persistence string — that string is what
  `TEMPER_SEED_HEX` replays.
- Success/reject counters are private; the public `Display` impl of
  `TestRunner` is the sanctioned read-out (line format `successes: N`,
  `local rejects: N`, `global rejects: N`) — parsed in `parse_counts`.
- `Config::rng_algorithm = RngAlgorithm::ChaCha` for reproducibility;
  `RngSeed::Fixed(u64)` gives determinism but does NOT reproduce a previously
  persisted failing input bit-for-bit (only the persisted-seed path does) —
  the demo deliberately uses the persisted-seed form.
- `Config::result_cache` set to `noop_result_cache` (per-run freshness);
  `max_shrink_iters` capped at 10 000.

## M1 risks

- **Untrusted candidates** will exercise paths M0 only hit with trusted ones
  (weird panics, `#[inline]` tricks, SIG signals, infinite loops → timeout
  path, enormous shrink trees). The seatbelt profile must be re-audited
  against a hostile candidate before model integration.
- `parse_counts` depends on proptest's Display format; pinned `=1.12.0`
  mitigates, but any dep bump must re-verify.
- Diagnostics parser tested on E0308 + linker failures only; ICE and
  multi-file spans untested against real rustc ICEs.
- `bwrap` backend is constructed but untested (no Linux host in M0).
- Sandbox profile is allowlist-tuned to this machine's layout (homebrew
  rustup, CLT SDK); paths are resolved at runtime (home, SDK) but new setups
  (Xcode.app SDKs, non-standard rustup homes) may need profile additions.
- Shrink determinism is only as stable as proptest's algorithm; seed-exact
  replay verified within 1.12.0 only.
- `emit` full-copies `vendor/` (11 MB) each cycle (~tens of ms) — fine now;
  consider hardlink/CoW if M1 loops thousands of candidates.
- hyperfine σ ±21 ms on seatbelt → gate M1 decisions on min/median, not mean.
