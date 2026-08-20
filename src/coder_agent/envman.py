"""Provision and repair the environment a generated experiment runs in.

The design point of this module is the one thing that broke the previous run:
a missing package is an *environment* fact, and no amount of rewriting the
experiment's source will change it. So installing is a first-class repair
action here, sitting beside code regeneration rather than underneath it.

Layout is a pre-baked base env plus a per-experiment overlay:

    base env (built once, on a viz node, with network)
        numpy pandas scipy scikit-learn statsmodels matplotlib pyarrow ...
        └── overlay venv (per experiment, --system-site-packages)
                only what this particular plan additionally needs

The base env carries the packages nearly every experiment reaches for, so the
common case installs nothing at all; the overlay keeps one experiment's exotic
pins from breaking the next one's. Compute nodes may have no outbound network,
which is why the base env is built ahead of time and why `has_network()` is
probed rather than assumed — offline, a missing package is a terminal,
clearly-worded failure instead of six retries against an unreachable index.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Import name → PyPI distribution name, for the cases where they differ, and
# ONLY where installing that distribution actually provides that import. An
# unknown import falls through to its own name, which is right far more often
# than it is wrong.
#
# Successor packages do NOT belong here — see DEAD_IMPORTS below for why.
IMPORT_TO_PACKAGE: dict[str, str] = {
    "sklearn": "scikit-learn",
    "skimage": "scikit-image",
    "cv2": "opencv-python-headless",
    "PIL": "pillow",
    "yaml": "pyyaml",
    "bs4": "beautifulsoup4",
    "dateutil": "python-dateutil",
    "sqlalchemy": "SQLAlchemy",
    "serial": "pyserial",
    "Crypto": "pycryptodome",
    "OpenSSL": "pyOpenSSL",
    "pkg_resources": "setuptools",
    "torch": "torch",
    "arviz": "arviz",
    "geopandas": "geopandas",
    "rasterio": "rasterio",
    "shapely": "shapely",
    "libpysal": "libpysal",
    "esda": "esda",
    "spreg": "spreg",
    "statsmodels": "statsmodels",
    "netCDF4": "netCDF4",
    "xarray": "xarray",
    "nx": "networkx",
    "mpl_toolkits": "matplotlib",
    "google.protobuf": "protobuf",
    "jax": "jax",
    "flax": "flax",
    "transformers": "transformers",
    "datasets": "datasets",
}

# Imports that no installable distribution provides, mapped to what to use
# instead. These are NOT aliases, and the difference is the whole point:
# installing `scikit-learn` really does provide `import sklearn`, but installing
# `pymc` does not provide `import pymc3` — PyMC 3 is end-of-life and its module
# name is simply gone. Treating the second case like the first produces an
# install that reports success while the identical ImportError comes back on
# every retry, which is exactly the loop this project exists to prevent.
#
# So these route to code regeneration, carrying the replacement API with them.
DEAD_IMPORTS: dict[str, tuple[str, str]] = {
    "pymc3": ("pymc", "PyMC 3 is end-of-life. Use PyMC 5: `import pymc as pm`. The API is close but not identical — `pm.Model()`, `pm.sample()` and the distributions are the same, while `pm.sample(return_inferencedata=True)` is now the default and theano/aesara references should become pytensor."),
    "pystan": ("cmdstanpy", "PyStan 2.x is unmaintained and will not build on a cluster. Use CmdStanPy: `from cmdstanpy import CmdStanModel`, write the Stan program to a .stan file, then `CmdStanModel(stan_file=...)` and `.sample()`."),
    "theano": ("pytensor", "Theano is dead. PyMC 5 uses PyTensor: `import pytensor.tensor as pt`."),
    "aesara": ("pytensor", "Aesara was renamed. Use `import pytensor.tensor as pt`."),
    "sklearn.cross_validation": ("scikit-learn", "That module was removed in scikit-learn 0.20. Use `sklearn.model_selection`."),
    "scipy.misc": ("scipy", "scipy.misc was removed. Use `imageio` for image IO or the specific scipy submodule."),
}

# Packages a generated experiment must never pull in: they either fight the
# agent for the GPU, try to manage the environment from inside it, or are
# unmaintained forks that will not build on a cluster.
INSTALL_DENYLIST = {"pystan", "vllm", "pip", "uv", "conda", "setuptools", "wheel"}

BASE_ENV_PACKAGES = [
    "numpy", "pandas", "scipy", "scikit-learn", "statsmodels", "matplotlib",
    "seaborn", "pyarrow", "tqdm", "requests", "geopandas", "shapely", "rasterio",
    "libpysal", "esda", "spreg", "cmdstanpy", "pymc", "arviz", "xarray", "networkx",
]


class EnvError(RuntimeError):
    """The environment could not be provisioned or repaired."""


@dataclass
class ExperimentEnv:
    """One experiment's overlay venv, plus the record of what had to be installed."""

    venv_path: Path
    python_bin: str
    base_env: str = ""
    installs: list[dict[str, object]] = field(default_factory=list)

    def record(self, packages: list[str], *, ok: bool, detail: str, reason: str) -> None:
        self.installs.append(
            {"packages": packages, "ok": ok, "reason": reason, "detail": detail[-2000:]}
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "venv_path": str(self.venv_path),
            "python": self.python_bin,
            "base_env": self.base_env,
            "installs": self.installs,
        }


def has_uv() -> bool:
    return shutil.which("uv") is not None


def has_network(host: str = "pypi.org", port: int = 443, timeout: float = 4.0) -> bool:
    """Probe, never assume: the same code runs on a laptop and on an isolated compute node."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def is_dead_import(module: str) -> tuple[str, str] | None:
    """If this import can never be satisfied by installing anything, say what to use instead."""
    if module in DEAD_IMPORTS:
        return DEAD_IMPORTS[module]
    root = module.split(".")[0]
    return DEAD_IMPORTS.get(root)


def resolve_package(module: str) -> str:
    """Map an import name to the distribution that provides it."""
    if module in IMPORT_TO_PACKAGE:
        return IMPORT_TO_PACKAGE[module]
    root = module.split(".")[0]
    return IMPORT_TO_PACKAGE.get(root, root)


def _base_python(base_env: str) -> str | None:
    if not base_env:
        return None
    candidate = Path(base_env)
    if candidate.is_dir():
        for relative in ("bin/python3", "bin/python"):
            if (candidate / relative).exists():
                return str(candidate / relative)
        return None
    return str(candidate) if candidate.exists() else None


def provision(
    experiment_dir: Path,
    *,
    base_env: str = "",
    requirements: list[str] | None = None,
    reuse: bool = True,
    venv_root: str = "",
) -> ExperimentEnv:
    """Create (or reuse) the overlay venv for one experiment.

    Inherits the base env's site-packages when one is configured, so the heavy
    scientific stack is shared rather than reinstalled per experiment — which on
    a cluster is the difference between seconds and a quarter of an hour, and
    which Barkla's own Python guidance recommends for exactly the reason it
    matters here: duplicated packages "often lead to exceeding the file number
    quota".

    `venv_root` puts the venv somewhere other than beside the results. On Barkla
    that is localscratch, which has no inode quota and is node-local — the venv
    is disposable, the results are not, and they should not share a filesystem.
    """
    if venv_root:
        venv_path = Path(venv_root) / experiment_dir.name / ".venv"
    else:
        venv_path = experiment_dir / ".venv"
    python_bin = str(venv_path / "bin" / "python")

    if reuse and Path(python_bin).exists():
        env = ExperimentEnv(venv_path=venv_path, python_bin=python_bin, base_env=base_env)
    else:
        parent = _base_python(base_env)
        if base_env and parent is None:
            raise EnvError(
                f"CODER_BASE_ENV={base_env!r} does not contain a Python interpreter. "
                "Build it with scripts/build_base_env.sh, or unset it to run bare."
            )
        _create_venv(venv_path, parent)
        if not Path(python_bin).exists():
            raise EnvError(f"venv creation produced no interpreter at {python_bin}")
        env = ExperimentEnv(venv_path=venv_path, python_bin=python_bin, base_env=base_env)

    if requirements:
        missing = [r for r in requirements if not is_available(env, _import_name_of(r))]
        if missing:
            install(env, missing, reason="declared requirements")
    return env


def _create_venv(venv_path: Path, parent_python: str | None) -> None:
    venv_path.parent.mkdir(parents=True, exist_ok=True)
    if venv_path.exists():
        shutil.rmtree(venv_path, ignore_errors=True)

    if has_uv():
        command = ["uv", "venv", str(venv_path)]
        if parent_python:
            command += ["--python", parent_python, "--system-site-packages"]
    else:
        # No uv (a bare login shell, a container without it): stdlib venv is
        # slower but always present, and the rest of this module only ever
        # talks to the resulting interpreter.
        interpreter = parent_python or sys.executable
        command = [interpreter, "-m", "venv", str(venv_path)]
        if parent_python:
            command.insert(3, "--system-site-packages")

    result = subprocess.run(command, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise EnvError(f"could not create venv at {venv_path}: {result.stderr.strip()[-1500:]}")


def is_available(env: ExperimentEnv, module: str) -> bool:
    """Ask the overlay interpreter whether it can import a module.

    Asked of the target interpreter, not of the agent's own — the agent process
    having pandas says nothing about the venv the experiment will run in, and
    conflating the two is exactly how a run reaches execution and dies on import.
    """
    if not module:
        return False
    probe = (
        "import importlib.util,sys;"
        f"sys.exit(0 if importlib.util.find_spec({module!r}) else 1)"
    )
    try:
        return subprocess.run(
            [env.python_bin, "-c", probe], capture_output=True, timeout=120
        ).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _import_name_of(requirement: str) -> str:
    """Strip a version spec/extras off a requirement to get something importable."""
    name = requirement.split(";")[0].strip()
    for separator in ("==", ">=", "<=", "~=", "!=", ">", "<", "["):
        name = name.split(separator)[0]
    name = name.strip()
    reverse = {v.lower(): k for k, v in IMPORT_TO_PACKAGE.items()}
    return reverse.get(name.lower(), name.replace("-", "_"))


def install(env: ExperimentEnv, packages: list[str], *, reason: str) -> tuple[bool, str]:
    """Install into the overlay. Returns (ok, detail) and records the attempt.

    Never raises on a failed install: the caller needs the detail to decide
    whether to try a different distribution name or to stop, and an exception
    here would lose the difference between "no network" and "no such package".
    """
    wanted = [p for p in dict.fromkeys(packages) if p and p.lower() not in INSTALL_DENYLIST]
    refused = [p for p in packages if p and p.lower() in INSTALL_DENYLIST]
    if refused:
        detail = (
            f"refused to install {refused}: on the denylist "
            "(see envman.INSTALL_DENYLIST for why each one is there)"
        )
        env.record(refused, ok=False, detail=detail, reason=reason)
        if not wanted:
            return False, detail
    if not wanted:
        return False, "nothing to install"

    if not has_network():
        detail = (
            f"cannot install {wanted}: no outbound network from this node. "
            "Add the package to scripts/build_base_env.sh and rebuild the base "
            "env on a viz node — compute nodes cannot reach the package index."
        )
        env.record(wanted, ok=False, detail=detail, reason=reason)
        return False, detail

    if has_uv():
        command = ["uv", "pip", "install", "--python", env.python_bin, *wanted]
    else:
        command = [env.python_bin, "-m", "pip", "install", *wanted]

    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=1800)
    except subprocess.TimeoutExpired:
        detail = f"install of {wanted} timed out after 30 minutes"
        env.record(wanted, ok=False, detail=detail, reason=reason)
        return False, detail

    ok = result.returncode == 0
    detail = (result.stdout + "\n" + result.stderr).strip()
    env.record(wanted, ok=ok, detail=detail, reason=reason)
    return ok, detail


def freeze(env: ExperimentEnv) -> str:
    """The overlay's resolved package set, recorded alongside results for reproducibility."""
    command = (
        ["uv", "pip", "freeze", "--python", env.python_bin]
        if has_uv()
        else [env.python_bin, "-m", "pip", "freeze"]
    )
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=300)
        return result.stdout if result.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def probe(env: ExperimentEnv, modules: list[str]) -> dict[str, bool]:
    """Batch availability check, for a preflight report before anything executes."""
    return {module: is_available(env, module) for module in modules}


def describe() -> dict[str, object]:
    """What the agent can see about its own runtime — logged at startup."""
    return {
        "uv": shutil.which("uv") or None,
        "network": has_network(),
        "python": sys.version.split()[0],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "(unset)"),
        "slurm_job": os.environ.get("SLURM_JOB_ID", "(none)"),
    }


def write_manifest(env: ExperimentEnv, path: Path) -> None:
    path.write_text(json.dumps({**env.to_dict(), "freeze": freeze(env)}, indent=2))
