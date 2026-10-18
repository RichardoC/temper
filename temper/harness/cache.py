"""Hardlink-cache warm registry-dependency artifacts into the workdir.

The vendored third-party deps (proptest etc.) never change between emissions,
so their build artifacts under the warm ``rust-template/target`` can be
hardlinked into ``.work/gen/target`` instead of rebuilt. This makes the
incremental per-candidate cycle measure only the candidate bin itself.

Hardlinks (not copies) keep the cache effectively free; mtimes are preserved
with ``os.utime`` because cargo's fingerprinting compares timestamps.

Stdlib only.
"""

from __future__ import annotations

import os
from pathlib import Path

from temper.harness.emit import TEMPLATE_DIR

# Warm target we hardlink FROM.
WARM_TARGET = TEMPLATE_DIR / "target"

# Subdirectories of target/debug holding registry-dependency artifacts.
# (.fingerprint covers dep fingerprints; build/ holds build-script outputs.)
_CACHE_SUBDIRS = ("deps", ".fingerprint", "build")


def _link_preserving_mtime(src: Path, dst: Path) -> bool:
    """Hardlink src->dst, preserving mtime. Returns False if dst existed."""
    try:
        os.link(src, dst)
    except FileExistsError:
        return False
    except OSError:
        # Cross-device or unsupported: fall back to a real copy.
        if dst.exists():
            return False
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(src.read_bytes())
    st = os.stat(src, follow_symlinks=False)
    os.utime(dst, ns=(st.st_atime_ns, st.st_mtime_ns), follow_symlinks=False)
    return True


def prime_target_cache(
    workdir: Path,
    warm_target: Path | None = None,
) -> int:
    """Hardlink warm registry-dep artifacts into ``workdir/target``.

    Returns the number of links created (already-linked files are skipped).
    Never fails on a missing warm target — returns 0 and lets cargo build
    from scratch.
    """
    warm_target = Path(warm_target) if warm_target is not None else WARM_TARGET
    workdir = Path(workdir)
    debug_warm = warm_target / "debug"
    if not debug_warm.is_dir():
        return 0

    linked = 0
    for sub in _CACHE_SUBDIRS:
        src_root = debug_warm / sub
        if not src_root.is_dir():
            continue
        for dirpath, _dirnames, filenames in os.walk(src_root):
            rel = os.path.relpath(dirpath, debug_warm)
            dst_dir = workdir / "target" / "debug" / rel
            dst_dir.mkdir(parents=True, exist_ok=True)
            for name in filenames:
                src_file = Path(dirpath) / name
                dst_file = dst_dir / name
                if dst_file.exists():
                    continue
                if _link_preserving_mtime(src_file, dst_file):
                    linked += 1
    return linked
