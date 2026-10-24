#!/usr/bin/env python3
"""M3.1 harness drafting: the prompt, the freeze-validation stub, the freeze tool.

This module is the part of M3.1 that must be frozen BEFORE any draft exists, so
it lives in one place with one sha (``draft_prompt_sha256``) recorded in every
P-gen row and in the freeze manifest:

  * ``HARNESS_DRAFT_INSTRUCTION`` — the pinned instruction text. It is written
    once, here, and never tuned against draft results; that is what keeps the
    G4 compile-rate measurement honest. Editing it after the first draft would
    be in-sample shaping of the property generator, which is precisely the
    failure the pre-registration discipline exists to prevent.
  * ``build_harness_prompt`` — instruction + the problem's docstring+signature
    ONLY. The MultiPL-E ``tests`` string is never in scope: the prompt is built
    from :class:`temper.steer.view.SteerProblemView`, whose field allowlist is
    ``("name", "prompt")`` (D5), and ``verify_prompt_sidecar`` proves the text
    the drafter sees is byte-identical to the banked
    ``<problem>__prompt.txt`` sidecar's docstring+signature region.
  * ``signature_stub`` — the ``todo!()`` module the harness must compile
    against (design §2.2: "compile success is the criterion"; runtime behaviour
    at freeze is recorded but not gated).
  * ``freeze`` — writes ``harness_freeze.json`` with its own canonical sha, the
    second of M3.1's two D7 freezes (the first is the verdict config v3).

Usage:

    # freeze validation over the P-gen drafts (CPU; cargo, no GPU)
    temper/.venv/bin/python scripts/m3_31_harness.py --validate \\
        --pgen bench/multipl-e/runs/<ts>-p-gen.jsonl --out /tmp/m3_31/freeze
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from temper.steer.property import (  # noqa: E402
    HARNESS_FREEZE_PATH,
    HARNESS_FREEZE_SCHEMA,
    Harness,
    compute_harness_freeze_sha256,
    load_harness_freeze,
    strategy_hash_of,
)
from temper.steer.view import SteerProblemView  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]

# --------------------------------------------------------------------------
# The instruction, TWO versions, both on record.
#
# v1 was written 2026-10-24 before any draft existed. It produced the Dream
# negative (0/12) and, on the AR twin, 2/25 compiles at canvas 256 — with the
# single biggest fixable class being that the model OMITTED the
# `use crate::candidate::{fn}` line (only 12/25 drafts had it) and got the
# `Strategy<Value = T>` / `run_property` shape wrong (E0271 x3, E0308 x1).
#
# v2 (operator-approved 2026-10-24, M3.1 §7 row C5) replaces the prose
# requirement list with a SKELETON TO FILL: the import is a literal line in the
# skeleton, and the exact `run_property` shape is given verbatim. Filling a
# skeleton sidesteps API-memory errors structurally instead of asking the model
# to recall an interface. The v1 text and its sha stay here as the labeled
# history of why v2 exists — a later reader must be able to see what changed and
# why, and every P-gen row records WHICH sha produced it.
# --------------------------------------------------------------------------

HARNESS_DRAFT_INSTRUCTION_V1 = """\
Write a Rust property-based test file for the `proptest` crate (pinned 1.12.0, \
std feature only, no other dependencies).

You are given a function's documentation and its exact signature; the body is \
missing. The function is provided by the module `crate::candidate` and is \
public.

```rust
{prompt}```

Write exactly ONE ```rust fenced code block containing the complete contents of \
`src/properties.rs`, which will be compiled alongside the candidate. It MUST:

1. import the function you are testing: `use crate::candidate::{fn_name};`
2. import proptest exactly as:
   `use proptest::prelude::*;`
   `use proptest::test_runner::{{TestCaseError, TestCaseResult, TestError, TestRunner}};`
3. define `pub const STRATEGY_DESC: &str = "...";` describing the input \
strategy in one line;
4. define a private generator `fn strategy() -> impl Strategy<Value = T>` where \
`T` is the tuple of the function's arguments (a bare type for one argument), \
covering the inputs that matter: the boundaries and edge cases the \
documentation mentions (empty, zero, one element, duplicates, negative and \
large values, non-ASCII or whitespace-only strings where strings are involved), \
plus the ordinary interior;
5. define a private checker `fn property(input: T) -> TestCaseResult` that \
calls the function and verifies the documented contract — the behaviour the \
documentation states, including its worked examples — returning \
`Err(TestCaseError::fail("readable explanation"))` when the contract is \
violated and `Ok(())` otherwise;
6. define exactly this entry point, with `T` substituted everywhere:
   `pub fn run_property(runner: &mut TestRunner) -> Result<(), TestError<T>> {{ \
runner.run(&strategy(), property) }}`

Rules:
- Only the Rust standard library and proptest 1.12.0 may be used. No serde, no \
other crates, no `proptest!` macro, no `#[test]` functions, no `fn main`, no \
`mod` declarations, no printing.
- Only the ONE function named above may be called from `crate::candidate`. Do \
not call anything else from that module.
- The property must be a genuine behavioural contract that a wrong \
implementation would violate. Do not merely call the function and return Ok: \
assert the documented postcondition, compare against an independent \
recomputation, or check a metamorphic relation, whichever the documentation \
makes checkable.
- Prefer inputs a wrong-but-plausible implementation gets wrong. Where the \
documentation states an example, make the strategy able to produce that input.
- The file must compile on its own against a stub whose body is `todo!()`: \
never assume the function's implementation, only its signature.
- Output ONLY the fenced Rust block, no prose.
"""

HARNESS_DRAFT_INSTRUCTION_V2 = """\
Complete the Rust file below. It is the property test for the function whose \
documentation and signature follow. Fill in the three marked parts; keep every \
other line EXACTLY as written.

The function's documentation and exact signature:

```rust
{prompt}```

Write exactly ONE ```rust fenced code block containing the completed file. It \
must have this shape, with your parts substituted:

```rust
use proptest::prelude::*;
use proptest::test_runner::{{TestCaseError, TestCaseResult, TestError, TestRunner}};
use crate::candidate::{fn_name};

pub const STRATEGY_DESC: &str = "<one line describing your input strategy>";

// PART 1: the generator. Return a boxed Strategy for the TUPLE of the
// function's arguments (a bare type if there is one argument). Give it the
// EXACT type below; only the body is yours.
fn strategy() -> BoxedStrategy<{value_type}> {{
    // your code
}}

// PART 2: the checker. Call {fn_name} with the input and verify the contract
// the documentation states. Return Ok(()) when it holds, and
// Err(TestCaseError::fail("...")) when it does not.
fn property(input: {value_type}) -> TestCaseResult {{
    // your code
}}

// PART 3: leave this entry point EXACTLY as it is.
pub fn run_property(runner: &mut TestRunner) -> Result<(), TestError<{value_type}>> {{
    runner.run(&strategy(), property)
}}
```

Rules:
- `{value_type}` is `{value_type}` — use that exact type in all three parts.
- Only the Rust standard library and proptest 1.12.0. No other crates, no \
serde, no `proptest!` macro, no `#[test]`, no `fn main`, no `mod`, no printing.
- Call ONLY `{fn_name}` from `crate::candidate`.
- Your strategy must be able to produce the inputs the documentation's examples \
use, plus the boundaries it mentions (empty, zero, one element, duplicates, \
negative or large values, multi-byte or whitespace strings where relevant).
- The property must be a real contract that a WRONG implementation would \
violate: assert the documented postcondition, compare against an independent \
recomputation, or check a metamorphic relation. Do not merely call the function \
and return Ok(()).
- The file is compiled against a stub whose body is `todo!()`: never assume the \
implementation, only the signature.
- Output ONLY the fenced Rust block, no prose.
"""

HARNESS_PROMPT_SHA256_V1 = hashlib.sha256(
    HARNESS_DRAFT_INSTRUCTION_V1.encode("utf-8")).hexdigest()
HARNESS_PROMPT_SHA256_V2 = hashlib.sha256(
    HARNESS_DRAFT_INSTRUCTION_V2.encode("utf-8")).hexdigest()

# The version every M3.1 cell runs. Changing it is a labeled amendment, and the
# sha recorded in each P-gen row must name the version that actually produced it.
HARNESS_INSTRUCTION_VERSION = "v2"
HARNESS_PROMPT_SHA256 = HARNESS_PROMPT_SHA256_V2


def value_type_of(signature_line: str) -> str:
    """The ``T`` of the skeleton: the tuple of the prompted argument types.

    The skeleton needs a CONCRETE type where v1 said "T", because "substitute T
    everywhere" is exactly the instruction the model kept failing. One argument
    gives that bare type; several give a tuple; none gives ``()``.

    Returns the type as source text. Refuses a signature it cannot parse rather
    than emitting a skeleton with a guessed type (a wrong ``T`` would fail every
    draft at once and look like a model failure).
    """
    m = re.search(r"\(([^)]*)\)\s*->", signature_line)
    if m is None:
        raise ValueError(
            f"cannot read the argument list from {signature_line!r}")
    args = m.group(1).strip()
    if not args:
        return "()"
    types: List[str] = []
    for part in _split_top_level(args):
        # `name: Type` -> Type (the name may be `_` or a pattern; the type is
        # everything after the first top-level colon)
        _, sep, ty = part.partition(":")
        if not sep:
            raise ValueError(
                f"argument {part!r} of {signature_line!r} has no `name: Type` "
                "shape")
        types.append(ty.strip())
    return types[0] if len(types) == 1 else "(" + ", ".join(types) + ")"


def _split_top_level(args: str) -> List[str]:
    """Split an argument list on top-level commas (brackets <> () [] aware)."""
    parts, depth, cur = [], 0, []
    for ch in args:
        if ch in "<([":
            depth += 1
        elif ch in ">)]":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    if "".join(cur).strip():
        parts.append("".join(cur).strip())
    return parts


def build_harness_prompt(view: SteerProblemView,
                         version: str = HARNESS_INSTRUCTION_VERSION) -> str:
    """The drafter's chat prompt: instruction + docstring+signature only.

    ``view.prompt`` is the allowlisted problem prompt (D5). Nothing else about
    the task is reachable here — in particular the vendored ``tests`` string
    never enters this scope, which is design §2.1's no-reference-leakage clause.
    """
    template = (HARNESS_DRAFT_INSTRUCTION_V2 if version == "v2"
                else HARNESS_DRAFT_INSTRUCTION_V1)
    return template.format(prompt=view.prompt, fn_name=view.fn_name,
                           value_type=value_type_of(view.signature_line))


def verify_prompt_sidecar(view: SteerProblemView, sidecar_path: Path) -> bool:
    """Is the banked ``<problem>__prompt.txt`` sidecar's problem region identical?

    The P-gen cell's input is documented as "the banked prompt sidecars from the
    C-null run". That sidecar is the *canvas* generator's chat prompt
    (``INSTRUCTION_TEMPLATE`` + the problem prompt); the harness drafter needs a
    different instruction around the SAME problem text. This check proves the
    bytes in between are the same, so the drafter sees exactly the information
    the canvas generator saw.
    """
    from temper.multipl_e.bench import INSTRUCTION_TEMPLATE

    if not sidecar_path.exists():
        return False
    banked = INSTRUCTION_TEMPLATE.format(prompt=view.prompt)
    return sidecar_path.read_text(encoding="utf-8") == banked


# --------------------------------------------------------------------------
# extraction + the validation stub
# --------------------------------------------------------------------------

_FENCE_RE = re.compile(r"```(?:rust|rs)?[ \t]*\r?\n(.*?)```", re.DOTALL)


def extract_rust_block(text: str) -> str:
    """The draft's Rust source: the last rust-ish fenced block, else the text.

    Takes the LAST block on purpose: a model that echoes the skeleton and then
    writes the file leaves the real answer last, and taking the first would
    silently validate an echo.
    """
    blocks = _FENCE_RE.findall(text)
    if not blocks:
        return text.strip() + ("\n" if text.strip() else "")
    return blocks[-1].strip() + "\n"


def signature_stub(view: SteerProblemView) -> str:
    """The ``todo!()`` module a harness must compile against (design §2.2).

    Exactly the prompted signature, made ``pub`` (``properties.rs`` imports
    ``crate::candidate::<fn>``, and a private item cannot be imported), with a
    ``todo!()`` body. Compiling against this proves the harness matches the
    PROMPTED signature and imports nothing from the benchmark's tests — the two
    things freeze validation exists to establish.
    """
    prompt = view.prompt.rstrip("\n")
    if not prompt.endswith("{"):
        raise ValueError(
            f"{view.name}: prompt does not end with '{{' — the MultiPL-E "
            "body-fragment prompt contract does not hold")
    lines = prompt.splitlines()
    for i in range(len(lines) - 1, -1, -1):
        m = re.match(r"^(\s*)(pub\s+)?fn\s+" + re.escape(view.fn_name) + r"\b",
                     lines[i])
        if m:
            if not m.group(2):
                lines[i] = m.group(1) + "pub " + lines[i][len(m.group(1)):]
            break
    return "\n".join(lines) + "\n    todo!()\n}\n"


def validate_harness(
    view: SteerProblemView,
    harness_source: str,
    *,
    cases: int = 8,
    timeout_ms: int = 500,
    workdir: Path | str | None = None,
) -> dict:
    """Compile-validate one harness against the ``todo!()`` stub (CPU, cargo).

    The gate is COMPILE success (design §2.2). Runtime behaviour on the stub is
    recorded anyway — the stub aborts or fails on the first case, so a harness
    that compiles but reports ``pass`` against ``todo!()`` is measuring nothing
    and the record shows it (that shape is reported in the G4 table, not
    silently accepted as a good harness).

    ``cases`` is deliberately tiny: validation asks whether the harness LINKS,
    not whether it can run 256 cases, and the stub aborts on case 1 either way.
    """
    from temper.harness.runner import run_candidate

    stub = signature_stub(view)
    try:
        out = run_candidate(stub, harness_source, cases=cases,
                            timeout_ms=timeout_ms, workdir=workdir)
    except Exception as exc:  # a cargo crash is a failed validation, not a crash
        return {"compiled": False, "status": "error",
                "error": f"{type(exc).__name__}: {exc}", "runtime_status": None,
                "diagnostics": []}
    codes: List[str] = []
    messages: List[str] = []
    for d in out.compile.diagnostics:
        if getattr(d, "is_error", False):
            codes.append(getattr(d, "code", "") or "")
            messages.append((getattr(d, "message", "") or "")[:200])
    return {
        "compiled": bool(out.compile.ok),
        "status": out.status,
        "compile_status": out.compile.status,
        "runtime_status": (out.test.status if out.test is not None else None),
        "runtime_reason": (out.test.reason if out.test is not None else ""),
        "error_codes": codes[:6],
        "diagnostics": messages[:6],
        "harness_sha": strategy_hash_of(harness_source),
    }


# --------------------------------------------------------------------------
# freeze manifest
# --------------------------------------------------------------------------


def build_freeze_payload(
    frozen: Dict[str, List[Harness]],
    *,
    draft_prompt_sha256: str,
    pgen_run: str,
    first_draft_results: List[dict],
    draft_dir: str,
) -> dict:
    """The freeze manifest: the sha-pinned harness set + the G4 pre-statistics."""
    n_first = len(first_draft_results)
    n_first_ok = sum(1 for r in first_draft_results if r["compiled"])
    n_second = sum(1 for r in first_draft_results if r.get("replacement_used"))
    n_second_ok = sum(1 for r in first_draft_results
                      if r.get("replacement_used") and r.get("replacement_compiled"))
    n_frozen = sum(len(v) for v in frozen.values())
    return {
        "schema": HARNESS_FREEZE_SCHEMA,
        "version": "m3.1-v1",
        "frozen_utc": "2026-10-24",
        "note": (
            "M3.1 D7's SECOND freeze (the first is steer_verdict_config.json "
            "v3). The harness set is a pre-commitment: P-gen drafts, the "
            "seed-2 replacement rule, and this manifest's sha are all fixed "
            "before any selection cell runs. P-prop-screen and P-prop-engine "
            "refuse to start unless the harness text re-hashes to the recorded "
            "strategy_hash."),
        "draft_prompt_sha256": draft_prompt_sha256,
        "pgen_run": pgen_run,
        "draft_dir": draft_dir,
        "config_sha_note": "verdict config v3 carries the same H/seeds/cases "
                           "facts; both shas are recorded in every M3.1 record",
        "compile_rate": {
            "first_drafts": n_first,
            "first_draft_compiled": n_first_ok,
            "first_draft_rate": round(n_first_ok / n_first, 4) if n_first else None,
            "gate": "G4 requires >= 90% of the FIRST drafts to compile; the "
                    "seed-2 replacements are reported separately and cannot "
                    "rescue the gate (§7 Q2 clarification)",
            "replacements_used": n_second,
            "replacements_compiled": n_second_ok,
            "frozen_harnesses": n_frozen,
        },
        "problems": [
            {"name": name,
             "harnesses": [h.record() for h in hs]}
            for name, hs in sorted(frozen.items())
        ],
        "first_draft_results": first_draft_results,
    }


def freeze(payload: dict, path: Path | str | None = None) -> dict:
    """Stamp and write the manifest (canonical sha over everything else)."""
    p = Path(path) if path is not None else HARNESS_FREEZE_PATH
    body = {k: v for k, v in payload.items() if k != "sha256"}
    stamped = dict(body, sha256=compute_harness_freeze_sha256(body))
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(stamped, indent=2) + "\n", encoding="utf-8")
    return stamped


# --------------------------------------------------------------------------
# CLI: --validate (build the freeze from a P-gen record)
# --------------------------------------------------------------------------


def load_pgen_rows(path: Path) -> Tuple[dict, List[dict]]:
    meta, rows = None, []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        o = json.loads(line)
        if o.get("type") == "run":
            meta = o
        elif o.get("type") == "candidate":
            rows.append(o)
    if meta is None:
        raise SystemExit(f"{path}: no run meta line")
    return meta, rows


def _read_raw(row: dict) -> str:
    rp = row.get("raw_path")
    if not rp:
        return ""
    p = Path(rp)
    if not p.is_absolute():
        p = REPO_ROOT / p
    return p.read_text(encoding="utf-8") if p.exists() else ""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--validate", action="store_true",
                    help="compile-validate every P-gen draft and freeze the "
                         "surviving set")
    ap.add_argument("--pgen", default=None, help="the P-gen run record")
    ap.add_argument("--out", default=None,
                    help="write harness source files + the manifest here")
    ap.add_argument("--manifest", default=None,
                    help="manifest path (default harness_freeze.json)")
    ap.add_argument("--check", default=None,
                    help="verify an existing manifest (tamper gate); exit 1 on "
                         "any post-freeze edit")
    ap.add_argument("--stub-preview", default=None,
                    help="print one problem's signature stub and exit")
    args = ap.parse_args(argv)

    if args.stub_preview:
        from temper.multipl_e.bench import load_problems
        for td in load_problems("humaneval-rs"):
            if td["name"] == args.stub_preview:
                print(signature_stub(SteerProblemView.from_task(td)))
                return 0
        raise SystemExit(f"no such problem: {args.stub_preview}")

    if args.check:
        try:
            frozen = load_harness_freeze(args.check)
        except Exception as exc:
            print(f"TAMPER/INVALID: {exc}", file=sys.stderr)
            return 1
        n = sum(len(v) for v in frozen.values())
        print(f"ok {args.check}: {len(frozen)} problems, {n} harnesses")
        return 0

    if not args.validate:
        print(__doc__)
        return 2
    if not args.pgen:
        raise SystemExit("--validate needs --pgen <run record>")
    out = Path(args.out) if args.out else Path("/tmp/m3_31/freeze")
    out.mkdir(parents=True, exist_ok=True)
    src_dir = out / "harnesses"
    src_dir.mkdir(parents=True, exist_ok=True)

    from temper.multipl_e.bench import load_problems

    views = {td["name"]: SteerProblemView.from_task(td)
             for td in load_problems("humaneval-rs")}
    meta, rows = load_pgen_rows(Path(args.pgen))
    print(f"P-gen record: {Path(args.pgen).name}  rows={len(rows)}")
    print(f"  draft prompt sha256: {meta.get('draft_prompt_sha256')}")

    by_problem: Dict[str, List[dict]] = {}
    for row in rows:
        by_problem.setdefault(row["problem"], []).append(row)

    frozen: Dict[str, List[Harness]] = {}
    first_results: List[dict] = []
    for name in meta.get("problem_names") or sorted(by_problem):
        view = views[name]
        slots = sorted(by_problem.get(name, []), key=lambda r: r["i"])
        kept: List[Harness] = []
        for slot_row in slots:
            slot = int(slot_row["i"])
            first_src = extract_rust_block(_read_raw(slot_row))
            first = validate_harness(view, first_src,
                                     workdir=out / "validate-wd")
            rec = {"problem": name, "slot": slot,
                   "seed": int(slot_row.get("seed") or 0),
                   "first_compiled": first["compiled"],
                   "first_runtime_status": first["runtime_status"],
                   "first_error_codes": first["error_codes"],
                   "first_sha": first["harness_sha"],
                   "replacement_used": False,
                   "replacement_compiled": None,
                   "frozen_sha": None,
                   "frozen_from": None}
            chosen = first_src if first["compiled"] else None
            if chosen is not None:
                rec["frozen_from"] = "first"
            else:
                # §2.2 / §7 Q2: ONE pre-registered replacement round (seed 2).
                rep_row = slot_row.get("replacement") or {}
                rep_text = _read_raw(rep_row) if rep_row else ""
                rec["replacement_used"] = True
                if rep_text:
                    rep_src = extract_rust_block(rep_text)
                    second = validate_harness(view, rep_src,
                                              workdir=out / "validate-wd")
                    rec["replacement_compiled"] = second["compiled"]
                    rec["replacement_runtime_status"] = second["runtime_status"]
                    rec["replacement_error_codes"] = second["error_codes"]
                    rec["replacement_sha"] = second["harness_sha"]
                    if second["compiled"]:
                        chosen = rep_src
                        rec["frozen_from"] = "replacement"
                else:
                    rec["replacement_compiled"] = False
                    rec["replacement_missing"] = True
            if chosen is None:
                print(f"  !! {name} slot {slot}: NO SURVIVOR "
                      f"(first codes {first['error_codes']})")
                rec["frozen_sha"] = None
                first_results.append(rec)
                continue
            sha = strategy_hash_of(chosen)
            src_path = src_dir / f"{name}__h{slot}.rs"
            src_path.write_text(chosen, encoding="utf-8")
            rec["frozen_sha"] = sha
            kept.append(Harness(
                problem=name, slot=slot, seed=int(slot_row.get("seed") or 0),
                source=chosen, strategy_hash=sha,
                source_path=str(src_path.relative_to(out)),
                replaced=rec["frozen_from"] == "replacement"))
            first_results.append(rec)
        if not kept:
            # Zero survivors is a STOP-and-escalate signal (§7 Q2), never a
            # silent degradation to confidence ranking.
            print(f"\nZERO SURVIVORS for {name}: the lane must stop and escalate "
                  "(§7 Q2(3)). Writing the manifest anyway so the failure is "
                  "inspectable; no cell may run against it.")
        frozen[name] = kept

    payload = build_freeze_payload(
        frozen, draft_prompt_sha256=str(meta.get("draft_prompt_sha256") or ""),
        pgen_run=str(args.pgen), first_draft_results=first_results,
        draft_dir=str(src_dir))
    manifest_path = Path(args.manifest) if args.manifest else (out / "harness_freeze.json")
    stamped = freeze(payload, manifest_path)

    rate = stamped["compile_rate"]
    print(f"\n=== freeze validation ===")
    print(f"  first drafts compiled : {rate['first_draft_compiled']}/"
          f"{rate['first_drafts']} = {rate['first_draft_rate']}")
    print(f"  replacements used     : {rate['replacements_used']} "
          f"(compiled {rate['replacements_compiled']})")
    print(f"  frozen harnesses      : {rate['frozen_harnesses']}")
    print(f"  G4 (>=90% first drafts): "
          f"{'PASS' if (rate['first_draft_rate'] or 0) >= 0.90 else 'FAIL'}")
    print(f"  manifest: {manifest_path}\n  sha256: {stamped['sha256']}")
    zero = [n for n, v in frozen.items() if not v]
    if zero:
        print(f"\nSTOP: zero-survivor problems {zero} — escalate before any cell")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
