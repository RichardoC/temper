"""Emit a generated candidate crate from the pinned rust-template layout.

The emitted crate lives in a REUSED workdir (``temper/.work/gen`` by default):
every emission wipes and re-copies the template skeleton, then overwrites
``src/candidate.rs`` and ``src/properties.rs``. The layout copied is exactly
what the template commits: Cargo.toml, .cargo/, vendor/, src/, Cargo.lock —
never ``target/`` (that is hydrated by :mod:`temper.harness.cache`).

The strategy hash is ``sha256(properties.rs)`` and is passed to the harness
binary via ``TEMPER_STRATEGY_HASH`` so run reports are attributable to the
exact property text they ran against.
"""

from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path

# Template source, relative to this file: <repo>/rust-template
REPO_ROOT = Path(__file__).resolve().parents[2]
TEMPLATE_DIR = REPO_ROOT / "rust-template"
DEFAULT_WORKDIR = REPO_ROOT / "temper" / ".work" / "gen"

# Template entries copied into the workdir. target/ is deliberately absent.
_TEMPLATE_LAYOUT = ("Cargo.toml", "Cargo.lock", ".cargo", "vendor", "src")


@dataclass(frozen=True)
class EmittedCrate:
    """An emitted candidate crate ready for cache-priming and build."""

    workdir: Path
    strategy_hash: str
    candidate_path: Path
    properties_path: Path


def strategy_hash_of(properties_source: str) -> str:
    """sha256 hex digest of the properties.rs source text."""
    return hashlib.sha256(properties_source.encode("utf-8")).hexdigest()


def emit(
    candidate_source: str,
    properties_source: str,
    workdir: Path | str | None = None,
) -> EmittedCrate:
    """Materialize a generated crate in ``workdir`` (reused across runs).

    Raises FileNotFoundError if the template tree is missing an expected
    layout entry — a broken template should fail loudly, not silently.
    """
    workdir = Path(workdir).resolve() if workdir is not None else DEFAULT_WORKDIR

    for entry in _TEMPLATE_LAYOUT:
        if not (TEMPLATE_DIR / entry).exists():
            raise FileNotFoundError(
                f"rust-template is missing expected layout entry: {entry}"
            )

    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)

    for entry in _TEMPLATE_LAYOUT:
        src = TEMPLATE_DIR / entry
        dst = workdir / entry
        if src.is_dir():
            shutil.copytree(src, dst, symlinks=True)
        else:
            shutil.copy2(src, dst)

    candidate_path = workdir / "src" / "candidate.rs"
    properties_path = workdir / "src" / "properties.rs"
    candidate_path.write_text(candidate_source, encoding="utf-8")
    properties_path.write_text(properties_source, encoding="utf-8")

    return EmittedCrate(
        workdir=workdir,
        strategy_hash=strategy_hash_of(properties_source),
        candidate_path=candidate_path,
        properties_path=properties_path,
    )
