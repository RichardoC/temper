"""Emit -> cache-prime -> cargo check -> cargo build -> run -> parse protocol.

Every stage has a hard subprocess timeout. Compile failures are rejected
cheaply via ``cargo check`` before a full build is attempted; the harness
binary is then executed under the selected sandbox backend and its single
protocol-v1 JSON line is parsed into a :class:`TestResult`.

Stdlib only.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from temper.feedback.diagnostics import Diagnostic, parse_cargo_messages
from temper.harness.cache import prime_target_cache
from temper.harness.emit import DEFAULT_WORKDIR, EmittedCrate, emit
from temper.harness.sandbox import select_backend, wrap_command

__all__ = ["CompileResult", "TestResult", "RunOutcome", "run_candidate"]

DEFAULT_CASES = 256
DEFAULT_TIMEOUT_MS = 2000
# Cargo stages get generous ceilings; the *candidate* wall clock is
# TEMPER_TIMEOUT_MS + margin below.
CHECK_TIMEOUT_S = 120.0
BUILD_TIMEOUT_S = 300.0
RUN_MARGIN_S = 15.0


def find_cargo() -> str:
    """Locate a cargo binary; on this mac, rustup shims live under
    /opt/homebrew/opt/rustup/bin (NOT ~/.cargo/bin)."""
    found = os.environ.get("TEMPER_CARGO") or ""
    for candidate in (
        found,
        "cargo",
        "/opt/homebrew/opt/rustup/bin/cargo",
        str(Path.home() / ".cargo" / "bin" / "cargo"),
    ):
        if not candidate:
            continue
        resolved = candidate if os.path.sep in candidate else None
        if resolved:
            if Path(resolved).exists():
                return resolved
        else:
            for d in os.environ.get("PATH", "").split(os.pathsep):
                if (Path(d) / "cargo").exists():
                    return str(Path(d) / "cargo")
    raise FileNotFoundError("cargo not found; set TEMPER_CARGO or PATH")


@dataclass(frozen=True)
class CompileResult:
    """Outcome of the check+build stages (both run even if check fails? no —
    build is skipped when check rejects, to keep the cheap-reject cheap)."""

    status: str                      # ok | compile_error | timeout | error
    diagnostics: list[Diagnostic] = field(default_factory=list)
    stderr: str = ""
    duration_s: float = 0.0
    stage: str = "check"             # which stage produced the failure

    @property
    def ok(self) -> bool:
        return self.status == "ok"


@dataclass(frozen=True)
class TestResult:
    """Parsed protocol-v1 result of running the harness binary."""

    status: str                       # pass | fail | abort | error | timeout
    report: dict | None = None        # the full parsed protocol line
    exit_code: int | None = None
    duration_s: float = 0.0
    stderr: str = ""

    @property
    def seed(self) -> str:
        return (self.report or {}).get("seed", "")

    @property
    def minimal_input(self) -> str:
        return (self.report or {}).get("minimal_input", "")

    @property
    def reason(self) -> str:
        return (self.report or {}).get("reason", "")


@dataclass(frozen=True)
class RunOutcome:
    compile: CompileResult
    test: TestResult | None
    emitted: EmittedCrate | None
    strategy_hash: str
    cache_links: int = 0

    @property
    def status(self) -> str:
        return self.test.status if self.test else self.compile.status


def _kill_tree(proc: subprocess.Popen) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        try:
            proc.kill()
        except ProcessLookupError:
            pass


def _run(
    cmd: list[str],
    cwd: Path,
    env: dict,
    timeout_s: float,
) -> tuple[subprocess.Popen, bytes, bytes, float, bool]:
    """Run cmd with a hard timeout; kill the whole process group on expiry."""
    start = time.monotonic()
    proc = subprocess.Popen(
        cmd, cwd=str(cwd), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        out, err = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        _kill_tree(proc)
        out, err = proc.communicate()
        return proc, out, err, time.monotonic() - start, True
    return proc, out, err, time.monotonic() - start, False


def _child_env(workdir: Path, strategy_hash: str, cases: int,
               timeout_ms: int, seed_hex: str | None) -> dict:
    env = dict(os.environ)
    # Keep the shim dir first so cargo -> rustup -> rustc all resolve.
    cargo = find_cargo()
    env["PATH"] = os.pathsep.join(
        [str(Path(cargo).parent)] + env.get("PATH", "").split(os.pathsep)
    )
    # rustc temp files must land inside the sandbox-writable workdir.
    tmpdir = workdir / ".tmp"
    tmpdir.mkdir(exist_ok=True)
    env["TMPDIR"] = str(tmpdir)
    env["TEMPER_CASES"] = str(cases)
    env["TEMPER_TIMEOUT_MS"] = str(timeout_ms)
    env["TEMPER_STRATEGY_HASH"] = strategy_hash
    env.pop("TEMPER_SANDBOX", None)  # backend is chosen by the caller, not leaked
    if seed_hex:
        env["TEMPER_SEED_HEX"] = seed_hex
    else:
        env.pop("TEMPER_SEED_HEX", None)
    return env


def _compile_failure(status: str, stage: str, stdout: bytes, stderr: bytes,
                     duration: float) -> CompileResult:
    diags = parse_cargo_messages(stdout.decode("utf-8", "replace"))
    if status == "compile_error" and not any(d.is_error for d in diags):
        status = "error"  # cargo died without structured diagnostics
    return CompileResult(
        status=status,
        diagnostics=diags,
        stderr=stderr.decode("utf-8", "replace"),
        duration_s=duration,
        stage=stage,
    )


def _parse_protocol_line(stdout: bytes) -> dict | None:
    """Last JSON object on stdout carrying the protocol-v1 marker."""
    report = None
    for line in stdout.decode("utf-8", "replace").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and obj.get("v") == 1 and "status" in obj:
            report = obj
    return report


def run_candidate(
    candidate_source: str,
    properties_source: str,
    *,
    cases: int = DEFAULT_CASES,
    timeout_ms: int = DEFAULT_TIMEOUT_MS,
    seed_hex: str | None = None,
    sandbox: str | None = None,
    workdir: Path | str | None = None,
) -> RunOutcome:
    """Full M0 cycle for one candidate: emit, cache, check, build, run.

    ``sandbox=None`` selects via TEMPER_SANDBOX env (seatbelt default on
    macOS). The returned outcome always has a ``compile`` result; ``test``
    is None unless compilation succeeded.
    """
    backend = select_backend(sandbox)
    emitted = emit(candidate_source, properties_source, workdir)
    links = prime_target_cache(emitted.workdir)
    env = _child_env(emitted.workdir, emitted.strategy_hash, cases, timeout_ms, seed_hex)
    cargo = find_cargo()

    # ---- stage 1: cheap reject (cargo check) --------------------------------
    check_cmd = wrap_command(
        [cargo, "check", "--offline", "--message-format=json"], emitted.workdir, backend
    )
    proc, out, err, dur, timed_out = _run(check_cmd, emitted.workdir, env, CHECK_TIMEOUT_S)
    if timed_out:
        return RunOutcome(_compile_failure("timeout", "check", out, err, dur),
                          None, emitted, emitted.strategy_hash, links)
    if proc.returncode != 0:
        return RunOutcome(_compile_failure("compile_error", "check", out, err, dur),
                          None, emitted, emitted.strategy_hash, links)
    check_diags = parse_cargo_messages(out.decode("utf-8", "replace"))

    # ---- stage 2: full build -------------------------------------------------
    build_cmd = wrap_command(
        [cargo, "build", "--offline", "--message-format=json-diagnostic-rendered-ansi"],
        emitted.workdir, backend,
    )
    proc, out, err, dur, timed_out = _run(build_cmd, emitted.workdir, env, BUILD_TIMEOUT_S)
    if timed_out:
        return RunOutcome(_compile_failure("timeout", "build", out, err, dur),
                          None, emitted, emitted.strategy_hash, links)
    if proc.returncode != 0:
        return RunOutcome(_compile_failure("compile_error", "build", out, err, dur),
                          None, emitted, emitted.strategy_hash, links)

    compile_ok = CompileResult(status="ok", diagnostics=check_diags, duration_s=dur)

    # ---- stage 3: run the harness binary under the same sandbox --------------
    binary = emitted.workdir / "target" / "debug" / "harness"
    if not binary.exists():
        compile_err = CompileResult(
            status="error", diagnostics=check_diags,
            stderr="build reported success but harness binary is missing",
            stage="build",
        )
        return RunOutcome(compile_err, None, emitted, emitted.strategy_hash, links)

    run_cmd = wrap_command([str(binary)], emitted.workdir, backend)
    run_timeout = timeout_ms / 1000.0 + RUN_MARGIN_S
    proc, out, err, dur, timed_out = _run(run_cmd, emitted.workdir, env, run_timeout)
    if timed_out:
        return RunOutcome(
            compile_ok,
            TestResult(status="timeout", exit_code=None, duration_s=dur,
                       stderr=err.decode("utf-8", "replace")),
            emitted, emitted.strategy_hash, links,
        )
    report = _parse_protocol_line(out)
    if report is None:
        return RunOutcome(
            compile_ok,
            TestResult(status="error", exit_code=proc.returncode, duration_s=dur,
                       stderr=err.decode("utf-8", "replace")),
            emitted, emitted.strategy_hash, links,
        )
    return RunOutcome(
        compile_ok,
        TestResult(status=str(report["status"]), report=report,
                   exit_code=proc.returncode, duration_s=dur,
                   stderr=err.decode("utf-8", "replace")),
        emitted, emitted.strategy_hash, links,
    )
