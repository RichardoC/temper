# temper

**temper** steers an open-weights diffusion language model to generate Rust
code, verified by sandboxed property-based testing. Generation is treated as
untrusted input: every candidate is compiled and executed inside a hardened
sandbox against a proptest property, and the structured outcome (pass /
fail + minimal counterexample seed / compile diagnostics) is the feedback
signal that steers the next generation round.

## Status: M0 (execution harness only)

M0 delivers the full verification loop around generated code — no model,
no GPU. The loop for one candidate:

```
emit ──► cache-prime ──► cargo check ──► cargo build ──► run harness ──► protocol-v1 JSON
 (candidate.rs +      (hardlink warm     (cheap reject)   (full build,     (single line:
  properties.rs)       dep artifacts)                      sandboxed)       pass/fail/abort)
```

1. **Emit** (`temper/harness/emit.py`) — copies the pinned `rust-template/`
   layout (Cargo.toml, Cargo.lock, .cargo/, vendor/, src/ — never `target/`)
   into a reused workdir `temper/.work/gen/` and overwrites `src/candidate.rs`
   (the generated code) and `src/properties.rs` (the property). The property's
   sha256 becomes `TEMPER_STRATEGY_HASH`.
2. **Cache** (`temper/harness/cache.py`) — hardlinks registry-dependency build
   artifacts (`deps/`, `.fingerprint/`, `build/`) from the warm template
   `target/` into the workdir, preserving mtimes, so each cycle rebuilds only
   the candidate bin.
3. **Check** — `cargo check --offline --message-format=json`: cheap reject of
   candidates that don't compile, before paying for a full build.
4. **Build** — `cargo build --offline` (vendored sources, no network).
5. **Run** (`temper/harness/runner.py`) — executes the harness binary, which
   runs the property under proptest 1.12.0's `TestRunner` API directly and
   prints exactly one protocol-v1 JSON line (see `rust-template/src/main.rs`).
6. **Diagnostics** (`temper/feedback/diagnostics.py`) — lenient parser for
   cargo JSON diagnostics (error codes, byte spans, machine-applicable
   suggestions); compiler ICEs are surfaced distinctly as toolchain failures.

Reproducibility: failing runs persist a proptest seed; re-running with
`TEMPER_SEED_HEX=<seed>` replays the exact failing input (seed-exact replay),
verified by `scripts/demo_m0.py` step 2.

## Security model

Generated code is **untrusted**. Layered mitigations in M0:

* **Zero-dependency generated crates.** The only third-party dependency is
  proptest (pinned `=1.12.0`, vendored). No `build.rs`, no proc-macros, no
  serde — the result protocol JSON is hand-serialized. This removes the
  `build.rs`/proc-macro arbitrary-code-execution surface entirely: the only
  code that ever executes is the harness + proptest + the candidate.
* **Whole-build sandboxing.** Every cargo invocation *and* the test binary run
  wrapped in one sandbox invocation (children inherit it):
  - macOS: **seatbelt** (`sandbox-exec`, SBPL, deny-default) — reads limited
    to system trees + the Rust toolchain; writes limited to the crate workdir,
    /private/tmp and /dev/null; `network*` denied; `file-map-executable`,
    `process-exec*`, `process-fork`, `sysctl-read` allowed (dyld requires the
    first or every binary dies).
  - Linux: **bwrap** equivalent (constructed, untested — review before use).
  - `TEMPER_SANDBOX=none` passthrough exists for debugging and prints a loud
    warning; never use it for real model output.
* **Offline builds.** `.cargo/config.toml` replaces crates.io with the
  committed `vendor/` tree; `[net] offline = true`. No downloads, no
  supply-chain mutation between template preparation and run.

## Usage

```sh
export PATH="/opt/homebrew/opt/rustup/bin:$PATH"   # rustup shims (this mac)
python3 scripts/demo_m0.py                          # 4-step acceptance demo
./scripts/measure_cycle.sh                          # hyperfine cycle timing
```

Backend selection: env `TEMPER_SANDBOX` = `seatbelt` | `bwrap` | `none`
(default: `seatbelt` on macOS).

## Layout

```
rust-template/     pinned candidate crate template (vendored proptest 1.12.0)
temper/harness/    emit / cache / sandbox / runner
temper/feedback/   cargo JSON diagnostics parser
scripts/           demo_m0.py, measure_cycle.sh
temper/docs/       milestone reports
```

License: Apache-2.0 (see `LICENSE`).
