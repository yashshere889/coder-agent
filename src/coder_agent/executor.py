"""Run generated code under bounds, and capture everything needed to diagnose it.

There is no Docker daemon on Barkla, so this is a subprocess sandbox, not a
security boundary: resource rlimits, a scratch working directory, an explicit
environment allowlist, and a wall-clock timeout. The trust model is that the
code came from our own model responding to our own plan — enough to contain a
script that goes wrong (infinite loop, memory blowup, crash), not enough to
contain something adversarial. `static_safety_check` runs before execution as a
coarse tripwire for the obviously destructive cases.

Output is captured in full to disk and returned as bounded head/tail slices:
a fix prompt needs the traceback at the end and the setup at the start, and a
200MB progress bar in between helps nobody and would blow the context window.
"""

from __future__ import annotations

import os
import re
import resource
import shutil
import signal
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

TAIL_CHARS = 8000
HEAD_CHARS = 2000

# Coarse, regex-based on purpose: this list runs against code the fix loop
# rewrites repeatedly, and it has to be cheap to extend. Treat additions as a
# security change.
DANGEROUS_PATTERNS: list[tuple[str, str]] = [
    (r"\brm\s+-rf\s+[/~]", "recursive delete of an absolute or home path"),
    (r"\bshutil\.rmtree\s*\(\s*['\"]?[/~]", "rmtree of an absolute or home path"),
    (r"\bos\.system\s*\(", "os.system shell-out"),
    (r"\bsubprocess\.[A-Za-z_]+\([^)]*shell\s*=\s*True", "shell=True subprocess"),
    (r"\bsocket\.socket\s*\(", "raw socket"),
    (r"\beval\s*\(\s*(?:input|open|requests)", "eval of external input"),
    (r"\b__import__\s*\(\s*['\"]os['\"]\s*\)\s*\.\s*system", "obfuscated os.system"),
    (r"\bpip\s+install\b|\bsubprocess[^\n]*['\"]pip['\"]", "self-installing packages (envman's job)"),
    (r"\bos\.environ\s*\[\s*['\"]CUDA_VISIBLE_DEVICES", "overriding the GPU allocation"),
    (r"/mnt/(?:fastscratch|scratch)/users/(?!\$)[A-Za-z0-9_]+/(?!.*experiments)", "writing outside the experiment tree"),
]

# Environment variables a generated experiment legitimately needs. Everything
# else (API keys, the agent's own LLM credentials, the user's shell config) is
# withheld — generated code has no business reading them, and a leaked key in a
# results file is not recoverable.
ENV_ALLOWLIST = (
    "PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "TZ", "USER",
    "HF_HOME", "HF_HUB_OFFLINE", "MPLCONFIGDIR", "MPLBACKEND",
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "CUDA_HOME", "LD_LIBRARY_PATH",
    "CMDSTAN", "STAN_NUM_THREADS", "XDG_CACHE_HOME",
)


@dataclass
class ExecutionResult:
    """What one run of a script produced. Everything the classifier reads is here."""

    exit_code: int
    stdout: str
    stderr: str
    duration_seconds: float
    timed_out: bool = False
    killed_signal: int | None = None
    stdout_path: str | None = None
    stderr_path: str | None = None
    artifacts: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out

    @property
    def output_tail(self) -> str:
        """stderr tail first — a traceback is what a fix prompt needs most."""
        parts = []
        if self.stderr.strip():
            parts.append("--- stderr (tail) ---\n" + _tail(self.stderr))
        if self.stdout.strip():
            parts.append("--- stdout (tail) ---\n" + _tail(self.stdout))
        return "\n\n".join(parts) or "(no output)"


def _tail(text: str, limit: int = TAIL_CHARS) -> str:
    if len(text) <= limit + HEAD_CHARS:
        return text
    return f"{text[:HEAD_CHARS]}\n...[{len(text) - limit - HEAD_CHARS} chars elided]...\n{text[-limit:]}"


def static_safety_check(code: str) -> list[str]:
    """Return a list of findings. Empty means nothing matched."""
    findings = []
    for pattern, description in DANGEROUS_PATTERNS:
        if re.search(pattern, code):
            findings.append(description)
    return findings


def compile_check(code: str, filename: str = "run.py") -> str | None:
    """Return a formatted SyntaxError, or None if the source compiles.

    Cheaper than launching a subprocess to discover the same thing, and it runs
    before the safety scan's cost on every regeneration.
    """
    try:
        compile(code, filename, "exec")
    except SyntaxError as exc:
        location = f"{exc.filename}:{exc.lineno}:{exc.offset}"
        return f"SyntaxError at {location}: {exc.msg}\n    {(exc.text or '').rstrip()}"
    except ValueError as exc:  # e.g. a null byte in the source
        return f"Source rejected by the compiler: {exc}"
    return None


def build_env(python_bin: str, extra: dict[str, str] | None = None, gpus: str = "") -> dict[str, str]:
    """The allowlisted environment a generated experiment runs under."""
    env = {key: os.environ[key] for key in ENV_ALLOWLIST if key in os.environ}
    env.setdefault("HOME", str(Path.home()))
    env.setdefault("PATH", "/usr/local/bin:/usr/bin:/bin")
    # Put the overlay venv's bin first so `python` inside the script's own
    # subprocesses (cmdstanpy compiling a model, say) is the same interpreter.
    bin_dir = str(Path(python_bin).parent)
    env["PATH"] = bin_dir + os.pathsep + env["PATH"]
    env["VIRTUAL_ENV"] = str(Path(python_bin).parent.parent)
    # Unbuffered, so a run killed by the timeout still has its output on disk.
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    # Headless: a plot call must not block forever waiting on a display.
    env.setdefault("MPLBACKEND", "Agg")
    # The agent holds the rest of the node's GPUs for the model server. An empty
    # string is meaningful and correct: it means "no GPU", not "all GPUs".
    env["CUDA_VISIBLE_DEVICES"] = gpus
    if extra:
        env.update(extra)
    return env


def _limits(memory_limit_gb: int, cpu_seconds: int):
    """preexec_fn: apply rlimits and start a new process group.

    The process group matters more than it looks — a timed-out script that
    spawned a compiler leaves orphans holding the GPU otherwise, and the next
    attempt then fails with an out-of-memory error that has nothing to do with
    its own code.
    """

    def apply() -> None:
        os.setsid()
        limits = [
            (resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 30)),
            (resource.RLIMIT_NOFILE, (4096, 4096)),
            (resource.RLIMIT_CORE, (0, 0)),
        ]
        if memory_limit_gb > 0 and sys.platform.startswith("linux"):
            # RLIMIT_AS is honoured on Linux; on macOS it mostly is not, and
            # setting it there produces spurious MemoryErrors in numpy, so the
            # laptop path deliberately skips it.
            nbytes = memory_limit_gb * 1024**3
            limits.append((resource.RLIMIT_AS, (nbytes, nbytes)))
        for what, values in limits:
            try:
                resource.setrlimit(what, values)
            except (ValueError, OSError):
                # A limit the platform won't accept is not worth failing the run
                # over; the wall-clock timeout is the backstop that always works.
                pass

    return apply


def run_script(
    script: Path,
    *,
    python_bin: str,
    workdir: Path,
    timeout_seconds: int,
    memory_limit_gb: int = 32,
    gpus: str = "",
    env_extra: dict[str, str] | None = None,
    log_dir: Path | None = None,
) -> ExecutionResult:
    """Execute one script and return everything the classifier needs."""
    import time

    workdir.mkdir(parents=True, exist_ok=True)
    log_dir = log_dir or workdir
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = log_dir / "stdout.log"
    stderr_path = log_dir / "stderr.log"

    before = _snapshot(workdir)
    started = time.monotonic()
    timed_out = False
    killed_signal: int | None = None

    with open(stdout_path, "wb") as out, open(stderr_path, "wb") as err:
        process = subprocess.Popen(
            [python_bin, str(script)],
            cwd=str(workdir),
            stdout=out,
            stderr=err,
            stdin=subprocess.DEVNULL,
            env=build_env(python_bin, env_extra, gpus),
            preexec_fn=_limits(memory_limit_gb, timeout_seconds),
        )
        try:
            exit_code = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            _kill_group(process)
            exit_code = process.wait()

    duration = time.monotonic() - started
    if exit_code < 0:
        killed_signal = -exit_code

    return ExecutionResult(
        exit_code=exit_code,
        stdout=_read(stdout_path),
        stderr=_read(stderr_path),
        duration_seconds=duration,
        timed_out=timed_out,
        killed_signal=killed_signal,
        stdout_path=str(stdout_path),
        stderr_path=str(stderr_path),
        artifacts=sorted(_snapshot(workdir) - before),
    )


def _kill_group(process: subprocess.Popen) -> None:
    """SIGTERM the whole group, then SIGKILL what ignored it."""
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(os.getpgid(process.pid), sig)
        except (ProcessLookupError, PermissionError):
            return
        try:
            process.wait(timeout=10)
            return
        except subprocess.TimeoutExpired:
            continue


def _read(path: Path, limit: int = 2_000_000) -> str:
    """Read a log back, capped — a runaway script can write gigabytes."""
    try:
        size = path.stat().st_size
        with open(path, "rb") as handle:
            if size > limit:
                handle.seek(size - limit)
            return handle.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def _snapshot(directory: Path) -> set[str]:
    """Relative paths under `directory`, so a run's new files can be listed after."""
    try:
        return {
            str(p.relative_to(directory))
            for p in directory.rglob("*")
            if p.is_file() and "__pycache__" not in p.parts
        }
    except OSError:
        return set()


def which_python(preferred: str | None = None) -> str:
    """Resolve an interpreter path, falling back to the one running the agent."""
    if preferred and Path(preferred).exists():
        return preferred
    return shutil.which("python3") or sys.executable
