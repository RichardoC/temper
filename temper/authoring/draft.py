"""Property drafting via GLM-5.3 on the Synthetic API (design §4a).

Stdlib only (urllib + json) — NO openai package. Key resolution order:
env SYNTHETIC_API_KEY, else ~/.pi/agent/models.json providers.synthetic.apiKey.

NEVER auto-approves: `temper draft` writes properties.draft.rs only;
`temper draft --approve` asks interactive y/N and only on 'y' copies the
draft to properties.rs and stamps task.toml properties.approved = true.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

__all__ = ["draft_properties", "resolve_api_key"]

ENDPOINT = "https://api.synthetic.new/openai/v1/chat/completions"
DEFAULT_MODEL = "hf:zai-org/GLM-5.3"
MODELS_JSON = Path.home() / ".pi" / "agent" / "models.json"

DRAFT_INSTRUCTIONS = """You are writing Rust property-based tests with the `proptest` crate (pinned 1.12.0, std feature only).

The candidate module `crate::candidate` exports exactly this API:
```rust
{signature}
```

Specification of the API:
{doc}

Write the BODY of a Rust file (it will be saved as src/properties.rs next to the candidate) that:
1. imports what it needs (`use crate::candidate::...; use proptest::prelude::*; use proptest::test_runner::{{TestCaseError, TestCaseResult, TestError, TestRunner}};`),
2. defines `pub const STRATEGY_DESC: &str = "...";` describing the strategy,
3. defines a `fn strategy() -> impl Strategy<Value = T>` producing inputs that stress the API including edge cases (empty, full, overflow, boundary sizes as appropriate),
4. defines a `fn property(input: T) -> TestCaseResult` that asserts the specification holds, returning Err(TestCaseError::fail("...")) with a readable message on violation,
5. defines the stable entry point:
   `pub fn run_property(runner: &mut TestRunner) -> Result<(), TestError<T>> {{ runner.run(&strategy(), property) }}`

Rules:
- Only the Rust standard library and proptest 1.12.0 (std feature) may be used. No serde, no other crates.
- The property must check BEHAVIOR, not implementation details.
- Output ONLY the Rust source (one ```rust fenced block), no prose.
"""


def resolve_api_key() -> str:
    key = os.environ.get("SYNTHETIC_API_KEY", "").strip()
    if key:
        return key
    if MODELS_JSON.exists():
        try:
            cfg = json.loads(MODELS_JSON.read_text(encoding="utf-8"))
            key = (cfg.get("providers", {}).get("synthetic", {})
                      .get("apiKey", "")).strip()
            if key:
                return key
        except Exception:
            pass
    raise SystemExit(
        "temper draft: no Synthetic API key found.\n"
        f"Set env SYNTHETIC_API_KEY, or put providers.synthetic.apiKey in "
        f"{MODELS_JSON}.\n"
        "Drafting must fail loudly, never silently skip (design §4a)."
    )


def _chat(messages, model: str, timeout_s: float = 300.0) -> str:
    """SSE-streaming chat completion (stdlib only).

    GLM-5.3 is a reasoning model: a non-streaming turn can run >7 min with
    zero bytes on the socket, which kills any fixed read timeout. Streaming
    keeps chunks flowing (per-read timeout only needs to cover the gap
    BETWEEN chunks), and gives live progress on stderr.
    """
    key = resolve_api_key()
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "temperature": 0.3,
    }
    req = urllib.request.Request(
        ENDPOINT,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
            "Accept": "text/event-stream",
        },
        method="POST",
    )
    chunks: list = []
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                continue
            delta = ((obj.get("choices") or [{}])[0].get("delta") or {})
            piece = delta.get("content")
            if piece:
                chunks.append(piece)
                sys.stderr.write(".")
                sys.stderr.flush()
    sys.stderr.write("\n")
    text = "".join(chunks)
    if not text.strip():
        raise RuntimeError("empty completion (stream produced no content)")
    return text


def draft_properties(
    task_dir: Path,
    *,
    approve: bool,
    model: str = DEFAULT_MODEL,
    write_draft_only: bool = True,
    _live_call=None,
) -> int:
    """Draft properties for a task; requires explicit --approve to install."""
    task_dir = Path(task_dir)
    if not task_dir.is_absolute():
        task_dir = Path.cwd() / task_dir
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib
    toml_path = task_dir / "task.toml"
    meta = tomllib.loads(toml_path.read_text(encoding="utf-8"))["task"]

    prompt = DRAFT_INSTRUCTIONS.format(
        signature=meta["signature"], doc=meta.get("doc", "")
    )
    call = _live_call or _chat
    print(f"draft: calling {model} at {ENDPOINT} ...", flush=True)
    try:
        text = call([{"role": "user", "content": prompt}], model=model)
    except Exception as exc:
        raise SystemExit(
            f"temper draft: API call failed ({type(exc).__name__}: {exc}). "
            "GLM-5.3 is a reasoning model — non-streaming turns can take "
            "several minutes; if this was a timeout, retry."
        )
    # pull the fenced rust block (drafts are reviewed by a human anyway)
    code = text
    if "```" in text:
        parts = text.split("```")
        for seg in parts[1::2]:
            seg = seg.strip()
            if seg.lower().startswith("rust"):
                seg = seg[4:].lstrip()
            if "fn strategy" in seg or "run_property" in seg:
                code = seg
                break
    draft_path = task_dir / "properties.draft.rs"
    draft_path.write_text(code.strip() + "\n", encoding="utf-8")
    print(f"draft: wrote {draft_path} ({len(code)} chars). "
          "REVIEW IT — temper never auto-approves.")

    if write_draft_only:
        print("draft: re-run with --approve to install after review.")
        return 0

    sys.stdout.write("Install as properties.rs and stamp "
                     "properties.approved=true? [y/N] ")
    sys.stdout.flush()
    ans = sys.stdin.readline().strip().lower()
    if ans != "y":
        print("draft: NOT approved; draft left in place, task.toml unchanged.")
        return 1
    (task_dir / "properties.rs").write_text(code.strip() + "\n", encoding="utf-8")
    stamp_task_approved(toml_path)
    print(f"draft: approved -> {task_dir/'properties.rs'} "
          f"(task.toml properties.approved = true)")
    return 0


def stamp_task_approved(toml_path: Path) -> None:
    text = toml_path.read_text(encoding="utf-8")
    if "properties_approved" in text:
        return
    if not text.endswith("\n"):
        text += "\n"
    text += "\n# Human-approved property drafting (temper draft --approve)\nproperties_approved = true\n"
    toml_path.write_text(text, encoding="utf-8")
