"""Classify a failed run, so the repair goes to whoever can actually fix it.

This is the module that exists because of one observed failure: three fix
attempts spent regenerating an experiment's source code in response to
`ModuleNotFoundError: No module named 'pandas'`. The code was never wrong. The
environment was. Regenerating could not possibly help, and the identical error
came back verbatim every time.

So nothing here decides *how* to rewrite anything. It answers one question —
what kind of failure is this? — and hands back a route. It is a pure function
over text: no LLM, no filesystem, no network, which is what makes it testable
against a corpus of real tracebacks.

The `signature` it computes is the other half of the fix. It is a normalized
identity for a failure (exception type + the symbol involved + where it
happened, with paths and line numbers flattened), so the loop can notice it is
seeing the same wall for the second time and change strategy instead of
spending its remaining budget re-running into it.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any

# Routes. A class maps to exactly one of these; `repair.py` dispatches on it.
ROUTE_ENV = "env_repair"
ROUTE_DATA = "data_repair"
ROUTE_CODE = "code_regeneration"
ROUTE_DOWNSCALE = "downscale"
ROUTE_TERMINAL = "terminal"

# Failure classes, in the order the checks below try them. Order matters: a
# ModuleNotFoundError also produces a traceback, and must not be read as
# generic runtime breakage.
CLASSES = (
    "missing_dependency",
    "missing_system_library",
    "api_mismatch",
    "missing_data",
    "network_unavailable",
    "resource",
    "syntax",
    "runtime_logic",
    "contract",
    "format",
    "safety",
)

ROUTE_BY_CLASS = {
    "missing_dependency": ROUTE_ENV,
    "missing_system_library": ROUTE_TERMINAL,
    "api_mismatch": ROUTE_CODE,
    "missing_data": ROUTE_DATA,
    "network_unavailable": ROUTE_DATA,
    "resource": ROUTE_DOWNSCALE,
    "syntax": ROUTE_CODE,
    "runtime_logic": ROUTE_CODE,
    "contract": ROUTE_CODE,
    "format": ROUTE_CODE,
    "safety": ROUTE_CODE,
}

_MODULE_RE = re.compile(r"ModuleNotFoundError:\s*No module named ['\"]([\w.]+)['\"]")
_IMPORT_NO_MODULE_RE = re.compile(r"ImportError:\s*No module named ['\"]?([\w.]+)")
_CANNOT_IMPORT_RE = re.compile(
    r"ImportError:\s*cannot import name ['\"]([\w.]+)['\"](?:\s*from\s*['\"]?([\w.]+))?"
)
_SO_RE = re.compile(r"(lib[\w.+-]*\.so[\w.]*|[\w.+-]+\.dylib):\s*cannot open shared object file|"
                    r"OSError:.*?(lib[\w.+-]*\.so[\w.]*)")
_FILE_RE = re.compile(r"FileNotFoundError:\s*\[Errno 2\][^:]*:\s*['\"](.+?)['\"]")
_HTTP_RE = re.compile(r"\b(?:HTTP Error |status[_ ]code[=: ]+|Response \[)(\d{3})\b")
_CUDA_OOM_RE = re.compile(r"CUDA out of memory|CUDA error: out of memory|cuDNN.*?OUT_OF_MEMORY")
_EXC_LINE_RE = re.compile(r"^(?P<type>[A-Za-z_][\w.]*(?:Error|Exception|Interrupt|Warning)):?(?: (?P<msg>.*))?$")
_FRAME_RE = re.compile(r'^\s*File "(?P<file>[^"]+)", line (?P<line>\d+), in (?P<func>\S+)')
_DIGITS_RE = re.compile(r"\b\d+\b")
_HEXADDR_RE = re.compile(r"0x[0-9a-fA-F]+")
_PATH_RE = re.compile(r"(/[\w.\-/]+)+")

# A FileNotFoundError on one of these is a *data* problem; on anything else it
# is likely the code writing to a path it never created, which is a code bug.
DATA_SUFFIXES = (
    ".csv", ".tsv", ".parquet", ".json", ".jsonl", ".nc", ".h5", ".hdf5", ".zip",
    ".shp", ".geojson", ".gpkg", ".tif", ".tiff", ".xlsx", ".dta", ".sav", ".txt",
)
DATA_DIR_HINTS = ("data/", "/data", "datasets", "raw", "inputs")


@dataclass(frozen=True)
class Diagnosis:
    """What went wrong, in the terms the repair router needs."""

    failure_class: str
    route: str
    signature: str
    summary: str
    evidence: str
    module: str | None = None
    path: str | None = None
    details: dict[str, Any] | None = None

    @property
    def is_environment(self) -> bool:
        return self.route == ROUTE_ENV

    def to_dict(self) -> dict[str, Any]:
        return {
            "failure_class": self.failure_class,
            "route": self.route,
            "signature": self.signature,
            "summary": self.summary,
            "module": self.module,
            "path": self.path,
            "details": self.details or {},
        }


def last_traceback(text: str) -> str:
    """The final traceback in a log — the one that actually ended the process.

    Long runs print warnings with tracebacks in them; taking the first match
    would diagnose an exception the script already handled.
    """
    if not text:
        return ""
    marker = "Traceback (most recent call last):"
    index = text.rfind(marker)
    if index != -1:
        return text[index:].strip()
    # No marker: only call it a traceback if the tail actually names an
    # exception. Returning the tail unconditionally would make ordinary stdout
    # ("done", a progress line) look like a crash, and a clean run that simply
    # produced no results would be misdiagnosed as a runtime error.
    tail = text[-4000:].strip()
    return tail if exception_type(tail) else ""


def exception_type(traceback_text: str) -> str:
    """The exception class named on the last non-empty line of a traceback."""
    for line in reversed([ln for ln in traceback_text.splitlines() if ln.strip()]):
        match = _EXC_LINE_RE.match(line.strip())
        if match:
            return match.group("type")
    return ""


def _last_user_frame(traceback_text: str, code_filename: str = "run.py") -> str:
    """The deepest frame inside our own generated file, not inside a library.

    A pandas internal frame is the same for a hundred different bugs; the line
    in `run.py` is what distinguishes them, and what a fix prompt has to name.
    """
    frames = _FRAME_RE.findall(traceback_text)
    ours = [f for f in frames if code_filename in f[0]]
    chosen = (ours or frames)[-1] if (ours or frames) else None
    return f"{chosen[0].rsplit('/', 1)[-1]}:{chosen[2]}" if chosen else ""


def _normalize(text: str) -> str:
    """Flatten the parts of a message that vary between otherwise identical failures."""
    text = _HEXADDR_RE.sub("0xADDR", text)
    text = _PATH_RE.sub(lambda m: m.group(0).rsplit("/", 1)[-1], text)
    return _DIGITS_RE.sub("N", text).strip()


def make_signature(failure_class: str, *parts: str) -> str:
    """A stable identity for 'this same failure again'."""
    payload = "|".join([failure_class, *(_normalize(p) for p in parts if p)])
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def _looks_like_data_path(path: str) -> bool:
    lowered = path.lower()
    return lowered.endswith(DATA_SUFFIXES) or any(hint in lowered for hint in DATA_DIR_HINTS)


def classify(
    *,
    exit_code: int,
    stdout: str,
    stderr: str,
    timed_out: bool = False,
    killed_signal: int | None = None,
    results: dict[str, Any] | None = None,
    results_present: bool = False,
    expected_metrics: list[str] | None = None,
) -> Diagnosis | None:
    """Diagnose one execution. Returns None when the run is genuinely fine.

    `results` / `results_present` / `expected_metrics` carry the contract check:
    a script can exit 0 and still have produced nothing, which is a failure the
    exit code alone will never reveal.
    """
    combined = f"{stderr}\n{stdout}"
    traceback_text = last_traceback(stderr) or last_traceback(stdout)
    # Fall back to the raw tail for *reading* the exception type: a native crash
    # can print `OSError: ...` with no Python traceback above it at all.
    exc = exception_type(traceback_text) or exception_type(combined[-4000:])
    frame = _last_user_frame(traceback_text)
    evidence = traceback_text[-6000:] if traceback_text else combined[-4000:]

    # --- environment: a package that isn't installed -------------------------
    # First, and unconditionally, because this is the failure that must never
    # reach the code generator.
    match = _MODULE_RE.search(combined) or _IMPORT_NO_MODULE_RE.search(combined)
    if match:
        module = match.group(1)
        return Diagnosis(
            failure_class="missing_dependency",
            route=ROUTE_ENV,
            signature=make_signature("missing_dependency", module),
            summary=f"The experiment imports {module!r}, which is not installed in its environment.",
            evidence=evidence,
            module=module,
            details={"import_name": module},
        )

    # A missing shared library is also environmental, but pip cannot fix it —
    # it needs a module load or a rebuilt base env, so it stops the run rather
    # than looping.
    so_match = _SO_RE.search(combined)
    if so_match and ("cannot open shared object" in combined or "image not found" in combined):
        library = so_match.group(1) or so_match.group(2) or "a shared library"
        return Diagnosis(
            failure_class="missing_system_library",
            route=ROUTE_TERMINAL,
            signature=make_signature("missing_system_library", library),
            summary=(
                f"A native library ({library}) is missing. pip cannot supply this — "
                "load the matching Barkla module or rebuild the base env with it."
            ),
            evidence=evidence,
            module=library,
        )

    # --- a package that is installed, but not the version the code assumes ---
    cannot_import = _CANNOT_IMPORT_RE.search(combined)
    if cannot_import:
        symbol, module = cannot_import.group(1), cannot_import.group(2) or ""
        return Diagnosis(
            failure_class="api_mismatch",
            route=ROUTE_CODE,
            signature=make_signature("api_mismatch", module, symbol),
            summary=(
                f"{module or 'a dependency'} is installed but has no {symbol!r} — "
                "the code targets a different version's API."
            ),
            evidence=evidence,
            module=module or None,
            details={"symbol": symbol},
        )

    # --- resources: OOM, timeout, or a kill with no traceback ----------------
    if timed_out:
        return Diagnosis(
            failure_class="resource",
            route=ROUTE_DOWNSCALE,
            signature=make_signature("resource", "timeout"),
            summary="The experiment exceeded its wall-clock budget and was killed.",
            evidence=(combined[-3000:] or "(no output before the timeout)"),
            details={"kind": "timeout"},
        )
    if _CUDA_OOM_RE.search(combined):
        return Diagnosis(
            failure_class="resource",
            route=ROUTE_DOWNSCALE,
            signature=make_signature("resource", "cuda_oom", frame),
            summary="The experiment ran out of GPU memory.",
            evidence=evidence,
            details={"kind": "cuda_oom"},
        )
    if exc == "MemoryError" or killed_signal == 9:
        return Diagnosis(
            failure_class="resource",
            route=ROUTE_DOWNSCALE,
            signature=make_signature("resource", "host_oom", frame),
            summary=(
                "The experiment ran out of host memory"
                + (" and was killed by the OOM killer." if killed_signal == 9 else ".")
            ),
            evidence=evidence,
            details={"kind": "host_oom", "signal": killed_signal},
        )

    # --- data: a file that isn't there, or a source that won't serve it ------
    file_match = _FILE_RE.search(combined)
    if file_match and _looks_like_data_path(file_match.group(1)):
        path = file_match.group(1)
        return Diagnosis(
            failure_class="missing_data",
            route=ROUTE_DATA,
            signature=make_signature("missing_data", path.rsplit("/", 1)[-1]),
            summary=f"The experiment expected a data file that does not exist: {path}",
            evidence=evidence,
            path=path,
        )
    http_match = _HTTP_RE.search(combined)
    if http_match and exc in {
        "HTTPError", "URLError", "ConnectionError", "ConnectTimeout", "ReadTimeout",
        "SSLError", "MaxRetryError", "NewConnectionError", "RequestException",
    }:
        status = http_match.group(1)
        return Diagnosis(
            failure_class="network_unavailable" if status.startswith("5") else "missing_data",
            route=ROUTE_DATA,
            signature=make_signature("data_fetch", exc, status),
            summary=f"Fetching the input data failed with HTTP {status} ({exc}).",
            evidence=evidence,
            details={"status": status, "exception": exc},
        )
    if exc in {"ConnectionError", "URLError", "ConnectTimeout", "NewConnectionError", "MaxRetryError"}:
        return Diagnosis(
            failure_class="network_unavailable",
            route=ROUTE_DATA,
            signature=make_signature("network_unavailable", exc),
            summary=(
                "The experiment could not reach the network to fetch its data. "
                "Barkla compute nodes are frequently isolated — stage data ahead of the run."
            ),
            evidence=evidence,
            details={"exception": exc},
        )

    # --- code ---------------------------------------------------------------
    if exc in {"SyntaxError", "IndentationError", "TabError"}:
        return Diagnosis(
            failure_class="syntax",
            route=ROUTE_CODE,
            signature=make_signature("syntax", exc, frame),
            summary=f"The generated code does not parse: {exc}.",
            evidence=evidence,
        )

    if exit_code != 0 or traceback_text:
        message = ""
        for line in reversed([ln for ln in traceback_text.splitlines() if ln.strip()]):
            if _EXC_LINE_RE.match(line.strip()):
                message = line.strip()
                break
        return Diagnosis(
            failure_class="runtime_logic",
            route=ROUTE_CODE,
            signature=make_signature("runtime_logic", exc or str(exit_code), frame, message),
            summary=message or f"The experiment exited with code {exit_code}.",
            evidence=evidence,
            details={"exception": exc, "frame": frame, "exit_code": exit_code},
        )

    # --- contract: exited clean, produced nothing usable --------------------
    return check_contract(results=results, results_present=results_present,
                          expected_metrics=expected_metrics, stdout=stdout)


def check_contract(
    *,
    results: dict[str, Any] | None,
    results_present: bool,
    expected_metrics: list[str] | None,
    stdout: str = "",
) -> Diagnosis | None:
    """Did a clean exit actually produce results? Exit code 0 does not imply it did.

    A hollow success is worse than a crash: it looks finished, gets written up,
    and nobody notices the metrics were an empty dict.
    """
    if not results_present:
        return Diagnosis(
            failure_class="contract",
            route=ROUTE_CODE,
            signature=make_signature("contract", "no_results_file"),
            summary="The run exited cleanly but never wrote results.json.",
            evidence=stdout[-2000:] or "(no output)",
            details={"problem": "missing_results_file"},
        )

    metrics = (results or {}).get("metrics")
    if not isinstance(metrics, dict) or not metrics:
        return Diagnosis(
            failure_class="contract",
            route=ROUTE_CODE,
            signature=make_signature("contract", "empty_metrics"),
            summary="results.json was written, but its 'metrics' object is empty.",
            evidence=str(results)[:2000],
            details={"problem": "empty_metrics"},
        )

    missing = [m for m in (expected_metrics or []) if not _metric_satisfied(metrics, m)]
    if missing and len(missing) == len(expected_metrics or []):
        # Every planned metric absent means the script computed something else
        # entirely. A partial match is left alone — naming is fuzzy and a fix
        # loop that argues about metric names burns budget for nothing.
        return Diagnosis(
            failure_class="contract",
            route=ROUTE_CODE,
            signature=make_signature("contract", "wrong_metrics"),
            summary=(
                "results.json reports none of the metrics the plan asked for. "
                f"Planned: {expected_metrics}. Reported: {sorted(metrics)}."
            ),
            evidence=str(results)[:2000],
            details={"problem": "wrong_metrics", "reported": sorted(metrics)},
        )

    degenerate = _degenerate_metrics(metrics)
    if degenerate:
        return Diagnosis(
            failure_class="contract",
            route=ROUTE_CODE,
            signature=make_signature("contract", "degenerate", ",".join(sorted(degenerate))),
            summary=(
                "Every reported metric is NaN, None or exactly zero — the computation "
                f"did not produce real numbers ({sorted(degenerate)})."
            ),
            evidence=str(results)[:2000],
            details={"problem": "degenerate_metrics", "keys": sorted(degenerate)},
        )
    return None


def _metric_satisfied(metrics: dict[str, Any], planned: str) -> bool:
    """Fuzzy match of a planned metric name against what was reported.

    Planned names are prose ("Posterior mean of the interaction coefficient
    between PM2.5 nickel and deprivation index"); reported keys are identifiers.
    Requiring equality would fail every correct run, so this asks whether the
    reported key's words appear in the planned phrase, or vice versa.
    """
    planned_words = {w for w in re.split(r"[^a-z0-9]+", planned.lower()) if len(w) > 3}
    if not planned_words:
        return True
    for key in metrics:
        key_words = {w for w in re.split(r"[^a-z0-9]+", key.lower()) if len(w) > 3}
        if key_words and (key_words <= planned_words or len(key_words & planned_words) >= 2):
            return True
    return False


def _degenerate_metrics(metrics: dict[str, Any]) -> list[str]:
    """Keys whose values carry no information. Empty list means at least one real number."""
    import math

    def is_dead(value: Any) -> bool:
        if value is None:
            return True
        if isinstance(value, bool):
            return False
        if isinstance(value, (int, float)):
            return value == 0 or math.isnan(value) or math.isinf(value)
        if isinstance(value, (list, tuple)):
            return len(value) == 0 or all(is_dead(v) for v in value)
        if isinstance(value, dict):
            return len(value) == 0 or all(is_dead(v) for v in value.values())
        if isinstance(value, str):
            return value.strip().lower() in {"", "nan", "none", "null", "n/a", "todo"}
        return False

    dead = [key for key, value in metrics.items() if is_dead(value)]
    return dead if len(dead) == len(metrics) else []


def format_failure(error: Exception, attempt: int) -> Diagnosis:
    """The model's response could not be parsed — a failure of the transport, not the code."""
    return Diagnosis(
        failure_class="format",
        route=ROUTE_CODE,
        signature=make_signature("format", type(error).__name__, str(error)),
        summary=f"The model's response was not in the required section format: {error}",
        evidence=str(error)[:3000],
        details={"attempt": attempt},
    )


def safety_failure(findings: list[str]) -> Diagnosis:
    """The generated code tripped the pre-execution safety scan; nothing has run."""
    return Diagnosis(
        failure_class="safety",
        route=ROUTE_CODE,
        signature=make_signature("safety", ",".join(sorted(findings))),
        summary="The generated code was refused before execution: " + "; ".join(findings),
        evidence="; ".join(findings),
        details={"findings": findings},
    )


def syntax_failure(message: str) -> Diagnosis:
    """`compile()` rejected the source — caught before a subprocess is even started."""
    return Diagnosis(
        failure_class="syntax",
        route=ROUTE_CODE,
        signature=make_signature("syntax", message.split(":")[0], message[:120]),
        summary=f"The generated code does not compile: {message.splitlines()[0]}",
        evidence=message,
    )
