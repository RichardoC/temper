"""Sandbox backends for whole-crate-build isolation.

Backends
--------
``seatbelt``  (DEFAULT on macOS) — macOS Seatbelt (SBPL) via ``sandbox-exec``.
    One invocation wraps the WHOLE cargo call (cargo -> rustc -> build scripts
    -> test binary all inherit the sandbox). The profile denies by default,
    explicitly allows ``file-map-executable`` (dyld breaks without it),
    ``process-exec*``, ``process-fork``, ``sysctl-read``; reads are limited to
    system trees + the Rust toolchain; writes are limited to the crate
    workdir, /private/tmp and /dev/null; ``network*`` is denied outright.
``bwrap``     — Linux bubblewrap equivalent. CONSTRUCTED BUT UNTESTED (M0 ran
    on macOS only). Reviewed before first real use.
``none``      — passthrough, with a loud warning. For debugging only.

Selection: env ``TEMPER_SANDBOX`` (``seatbelt`` | ``bwrap`` | ``none``);
defaults to ``seatbelt`` on macOS, ``none`` elsewhere (with warning).

Stdlib only.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

__all__ = ["select_backend", "wrap_command", "write_seatbelt_profile"]

_BACKENDS = ("seatbelt", "bwrap", "none")
_warned_none = False


def _sdk_path() -> str | None:
    """Best-effort Apple SDK path (cargo may need it for linker inputs)."""
    try:
        out = subprocess.run(
            ["xcrun", "--show-sdk-path"],
            capture_output=True, text=True, timeout=10, check=True,
        )
        p = out.stdout.strip()
        return p or None
    except Exception:
        return None


def write_seatbelt_profile(workdir: Path) -> Path:
    """Write the SBPL profile next to (not inside) the wiped workdir.

    Layout notes:
      * ``file-map-executable`` is allowed globally: dyld maps every library
        it loads, and a scoped-only allow breaks even fully-readable binaries.
        Reads remain scoped, so the effective map surface stays scoped too.
      * Toolchains under ~/.rustup AND cargo's shim tree under /opt/homebrew
        need read+execute: cargo (/opt/homebrew/opt/rustup/bin/cargo) invokes
        rustc shims that resolve into ~/.rustup/toolchains/...
      * Writes: crate workdir + /private/tmp + /dev/null only. The runner
        points TMPDIR at the workdir so rustc temp files stay inside it.
      * ``network*`` denied explicitly (also implied by deny-default).
    """
    workdir = Path(workdir).resolve()
    home = Path.home()
    sdk = _sdk_path()

    # Ancestor directories must be readable (stat/open): cargo walks from the
    # crate up to / looking for workspace roots and .cargo/config.toml.
    ancestors = [str(p) for p in workdir.parents]

    read_paths = [
        "/usr", "/System", "/Library", "/private/etc", "/dev",
        "/opt/homebrew",           # cargo/rustc shims + homebrew runtime
        str(home / ".rustup"),     # toolchains, settings
        str(home / ".cargo"),      # cargo global config (read-only)
        "/private/tmp",            # allowed read+write (scratch)
        str(workdir),              # the crate itself
        # /usr/bin/cc (xcode-select shim) resolves the developer dir through
        # these links; without them: "unable to read data link
        # at '/var/select/developer_dir'".
        "/private/var/select",
        "/private/var/db/xcode_select_link",
    ]
    if sdk and sdk not in read_paths:
        read_paths.append(sdk)

    read_lines = "\n".join(f'  (subpath "{p}")' for p in read_paths)
    read_lines += "\n" + "\n".join(f'  (literal "{p}")' for p in ancestors)
    # /var is a symlink to /private/var; path resolution needs its metadata.
    read_lines += '\n  (literal "/var")'

    profile = f"""; temper generated-crate sandbox (seatbelt/SBPL)
; wraps the whole cargo build+run; generated code never escapes this box.
(version 1)
(deny default)

; ---- process lifecycle ----------------------------------------------------
(allow process-exec*)
(allow process-fork)
(allow process-info*)
(allow sysctl-read)

; ---- dynamic linker: MUST be allowed or every binary dies in dyld ---------
(allow file-map-executable)

; ---- filesystem reads (system + toolchain + crate) ------------------------
(allow file-read*
{read_lines}
)

; ---- filesystem writes (crate workdir, tmp, null only) --------------------
(allow file-write*
  (subpath "{workdir}")
  (subpath "/private/tmp")
  (literal "/dev/null")
  (literal "/dev/dtracehelper")  ; cc attaches dtrace helpers
)
(allow file-ioctl
  (literal "/dev/null")
  (subpath "{workdir}")
)

; ---- mach: standard logging/notification services ------------------------
; (deliberately NOT com.apple.dt.CommandLineTools.installondemand: no
;  on-demand developer-tool installs from inside the sandbox)
(allow mach-lookup
  (global-name "com.apple.system.notification_center")
  (global-name "com.apple.system.logger")
  (global-name "com.apple.logd")
)

; ---- misc info (statfs/sysctl): needed by rustc/cargo bookkeeping --------
(allow system-info)

; ---- network: generation is fully offline --------------------------------
(deny network*)
"""
    profile_path = workdir.parent / "seatbelt.sb"
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile_path.write_text(profile, encoding="utf-8")
    return profile_path


def _seatbelt_prefix(workdir: Path) -> list[str]:
    profile = write_seatbelt_profile(workdir)
    return ["sandbox-exec", "-f", str(profile), "-D", f"WORKDIR={workdir}"]


def _bwrap_prefix(workdir: Path) -> list[str]:
    """bubblewrap equivalent. UNTESTED — constructed for Linux parity.

    Unshares everything (net, pid, ipc...), binds the world read-only and
    only the workdir (+/tmp) read-write. Review before first real use.
    """
    workdir = Path(workdir).resolve()
    ro = []
    for p in ("/usr", "/bin", "/lib", "/lib64", "/etc", "/opt"):
        if Path(p).exists():
            ro += ["--ro-bind", p, p]
    return [
        "bwrap",
        "--unshare-all",          # network + pid + ipc + uts: full isolation
        "--die-with-parent",
        "--new-session",
        "--dev", "/dev",
        "--proc", "/proc",
        "--tmpfs", "/tmp",
        *ro,
        "--bind", str(workdir), str(workdir),
    ]


def select_backend(backend: str | None = None) -> str:
    """Resolve the sandbox backend from arg or TEMPER_SANDBOX env."""
    name = (backend or os.environ.get("TEMPER_SANDBOX") or "").strip().lower()
    if not name:
        if sys.platform == "darwin":
            name = "seatbelt"
        else:
            name = "none"  # conservative default off-macOS; documented in README
    if name not in _BACKENDS:
        raise ValueError(
            f"unknown sandbox backend {name!r}; expected one of {_BACKENDS}"
        )
    return name


def wrap_command(
    cmd: list[str],
    workdir: Path,
    backend: str | None = None,
) -> list[str]:
    """Return ``cmd`` wrapped in the selected sandbox backend's invocation.

    ONE wrapping invocation covers the whole cargo call — inheritance covers
    rustc, build scripts and the test binary. ``none`` passthrough warns loudly.
    """
    global _warned_none
    backend = select_backend(backend)
    if backend == "seatbelt":
        return _seatbelt_prefix(workdir) + cmd
    if backend == "bwrap":
        if shutil.which("bwrap") is None:
            raise RuntimeError("sandbox backend 'bwrap' selected but bwrap not found")
        return _bwrap_prefix(workdir) + cmd
    # passthrough
    if not _warned_none:
        _warned_none = True
        print(
            "*** WARNING: TEMPER_SANDBOX=none — generated code is running "
            "UNSandboxed. Never point this at untrusted model output. ***",
            file=sys.stderr,
            flush=True,
        )
    return list(cmd)
