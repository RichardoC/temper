# Adding a bench task (M1 procedure)

A bench task is a directory `bench/<name>/` with exactly four files:

```
bench/<name>/
├── task.toml              # spec + prompt template + approval flag
├── properties.rs          # HUMAN-AUTHORED, approved proptest property
├── candidate.example.rs   # human reference impl (must pass properties.rs)
└── runs/                  # created by `temper gen` (JSONL run records)
    └── corpus/<strategy_hash>/   # persisted failing seeds (M3 asset)
```

## 1. `task.toml`

```toml
[task]
name = "<name>"
signature = "pub fn thing(...) -> ..."       # exact Rust signature; may be
                                             # multi-line for several items
doc = """Precise behavioral spec. Every ambiguity here becomes a wrong
candidate the verifier has to catch — write the contract, not vibes."""
must_define = ["thing"]                      # symbols extract.py requires
prompt_template = """... {signature} ... {doc} ..."""   # keep the rules block
                                            # from bench/sort/task.toml

[properties]
approved = false                             # flipped to true ONLY by a human
```

Keep `prompt_template`'s rules block (std-only, one fenced block, no main,
no tests) — it is calibrated to Dream-Coder's output habits.

## 2. `properties.rs` (human-authored)

Must define, for value type `T` of your choice (M0's `main.rs` is generic
over `TestError<T>`):

- `pub const STRATEGY_DESC: &str`
- `fn strategy() -> impl Strategy<Value = T>` — include edge cases
  (empty, full, overflow, boundary sizes).
- `fn property(input: T) -> TestCaseResult` — behavioral oracle only.
- `pub fn run_property(runner: &mut TestRunner) -> Result<(), TestError<T>>`
  implemented as `runner.run(&strategy(), property)`.

Copy the skeleton from `bench/sort/properties.rs`. Reference only std +
vendored proptest `=1.12.0` (std feature).

## 3. Verify before first gen

```sh
./bin/temper gen --task bench/<name> -K 1 --steps 24 --canvas 320
```

Also verify the example candidate passes and that a deliberately buggy
candidate FAILS (the property must have teeth):

```python
from temper.harness.runner import run_candidate
from pathlib import Path
d = Path("bench/<name>")
print(run_candidate((d/"candidate.example.rs").read_text(),
                    (d/"properties.rs").read_text()).status)
```

## 4. Approval

Set `properties.approved = true` in `task.toml` by hand (or via
`temper draft --task bench/<name> --approve` after reviewing a GLM-5.3
draft interactively). `temper gen` refuses to run tasks whose properties
are not approved.

## 5. Optional: GLM-5.3 drafting

`./bin/temper draft --task bench/<name>` writes `properties.draft.rs`
(never overwrites `properties.rs`). A human reviews, then
`./bin/temper draft --task bench/<name> --approve` asks y/N and installs +
stamps approval. Needs `SYNTHETIC_API_KEY` (or models.json key).
