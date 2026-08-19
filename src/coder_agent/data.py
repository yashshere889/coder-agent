"""Resolve a plan's data requirements, and refuse to let synthetic pass as real.

The failure this prevents is quiet and much worse than a crash. The previous
run's generated experiment assumed "placeholder CSV files in `data/`" that
nothing had ever created; had they existed as random numbers, it would have
produced posterior means, credible intervals and a supported/refuted verdict
off invented data, and every one of those numbers would have looked exactly
like a result.

So every input resolves to exactly one declared kind:

    real_local          a file you staged (e.g. CMS claims obtained under a DUA)
    real_download       fetched from a named open source, with a checksum
    synthetic_surrogate generated, with the reason the real source was unavailable
    unresolved          nothing worked, and the experiment cannot honestly run

and the verdict is computed in Python, not asked of the model: if any input is
a surrogate, the plan's `success_criteria` are NOT reported as met or refuted.
The experiment still runs — a working, reviewable pipeline on surrogate data is
a legitimate deliverable — it just does not get to claim evidence it does not
have.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

KIND_REAL_LOCAL = "real_local"
KIND_REAL_DOWNLOAD = "real_download"
KIND_SURROGATE = "synthetic_surrogate"
KIND_UNRESOLVED = "unresolved"

REAL_KINDS = {KIND_REAL_LOCAL, KIND_REAL_DOWNLOAD}

VERDICT_EVIDENCE = "real data — findings are interpretable as evidence for the hypothesis"
VERDICT_SURROGATE = (
    "synthetic surrogate data — the pipeline is validated but the findings are NOT "
    "interpretable as evidence for or against the hypothesis"
)
VERDICT_MIXED = (
    "mixed real and synthetic inputs — findings are NOT interpretable as evidence; "
    "the synthetic inputs are listed in data_provenance.json"
)

# Open sources whose access terms are public and whose endpoints are stable.
# Purely a hint offered to the code generator — nothing here is fetched by this
# module, and a source being listed does not mean the compute node can reach it.
OPEN_SOURCE_HINTS: list[tuple[str, str, str]] = [
    (r"\bEPA\b|air quality system|\bAQS\b", "EPA AQS", "https://aqs.epa.gov/data/api"),
    (r"american community survey|\bACS\b|census", "US Census ACS", "https://api.census.gov/data"),
    (r"\bNDVI\b|landsat|\bUSGS\b", "USGS EarthExplorer", "https://earthexplorer.usgs.gov"),
    (r"\bNOAA\b|climate data", "NOAA CDO", "https://www.ncei.noaa.gov/cdo-web/api/v2"),
    (r"\bWHO\b|global health observatory", "WHO GHO", "https://ghoapi.azureedge.net/api"),
    (r"world bank", "World Bank", "https://api.worldbank.org/v2"),
    (r"open ?street ?map|\bOSM\b", "OpenStreetMap", "https://overpass-api.de/api"),
    (r"hugging ?face", "Hugging Face datasets", "https://datasets-server.huggingface.co/rows"),
]

# Sources that are real but categorically NOT openly downloadable. Naming them
# explicitly is the difference between an agent that reports "I could not obtain
# CMS claims, here is why" and one that quietly makes some up.
RESTRICTED_SOURCES: list[tuple[str, str]] = [
    (r"\bCMS\b|medicare|medicaid|hospital claims",
     "CMS claims require a Data Use Agreement and are not openly downloadable"),
    (r"\bUK ?Biobank\b", "UK Biobank requires an approved application"),
    (r"\bMIMIC\b", "MIMIC requires PhysioNet credentialing and a signed DUA"),
    (r"\bNHS\b digital|hospital episode statistics|\bHES\b",
     "NHS HES data requires a Data Access Request"),
    (r"electronic health record|\bEHR\b|patient[- ]level",
     "patient-level records require ethics approval and a data agreement"),
    (r"\bSEER\b", "SEER research data requires a signed data-use agreement"),
]


@dataclass
class DataSource:
    """One resolved input, and the honest story of where it came from."""

    name: str
    kind: str
    description: str = ""
    uri: str = ""
    local_path: str = ""
    checksum: str = ""
    rows: int | None = None
    reason: str = ""
    hints: list[str] = field(default_factory=list)

    @property
    def is_real(self) -> bool:
        return self.kind in REAL_KINDS

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "description": self.description,
            "uri": self.uri,
            "local_path": self.local_path,
            "checksum": self.checksum,
            "rows": self.rows,
            "reason": self.reason,
            "hints": self.hints,
        }


def split_requirements(description: str, source: str) -> list[str]:
    """Break the planner's `source` field into the distinct inputs it names.

    The plan gives one `source` string covering several datasets ("EPA AQS for
    PM2.5; CMS claims for admissions; Census ACS for deprivation"). Each has its
    own availability story, so each needs its own provenance record.

    `description` is deliberately NOT split alongside it. It describes the
    *derived* dataset the inputs are merged into ("integrated neighbourhood-level
    panel linking monitors to tracts"), and splitting it produces a phantom
    fourth input that nothing can resolve. It is used only when `source` is
    empty.
    """
    text = (source or description).strip()
    parts = [p.strip() for p in re.split(r"[;\n]|(?<=[a-z])\s+and\s+(?=[A-Z])", text) if p.strip()]
    return parts or ["unspecified input"]


def _match(patterns: list[tuple[str, str]] | list[tuple[str, str, str]], text: str):
    for entry in patterns:
        if re.search(entry[0], text, re.IGNORECASE):
            return entry
    return None


def _staged_file(staging_dir: Path | None, requirement: str) -> Path | None:
    """Look for a file the user staged for this requirement.

    Matching is on shared keywords rather than an exact name: a human staging
    data names the file after the data, not after the plan's phrasing.
    """
    if not staging_dir or not staging_dir.is_dir():
        return None
    words = {w for w in re.split(r"[^a-z0-9]+", requirement.lower()) if len(w) > 3}
    if not words:
        return None
    best: tuple[int, Path] | None = None
    for candidate in staging_dir.rglob("*"):
        if not candidate.is_file() or candidate.name.startswith("."):
            continue
        name_words = {w for w in re.split(r"[^a-z0-9]+", candidate.stem.lower()) if len(w) > 3}
        overlap = len(words & name_words)
        if overlap and (best is None or overlap > best[0]):
            best = (overlap, candidate)
    return best[1] if best else None


def checksum_of(path: Path, limit: int = 64 * 1024 * 1024) -> str:
    """SHA-256 of the file's first 64MB — enough to detect a swap, cheap on big rasters."""
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as handle:
            digest.update(handle.read(limit))
    except OSError:
        return ""
    return digest.hexdigest()[:32]


def resolve(
    requirements: list[str],
    *,
    staging_dir: Path | None = None,
    network: bool = False,
) -> list[DataSource]:
    """Decide, per requirement, what data the experiment will actually use.

    Order is deliberate: a file the user staged beats anything this module could
    infer, a restricted source is never guessed at, and a surrogate is the last
    resort and is always labelled as one.
    """
    resolved: list[DataSource] = []
    for requirement in requirements:
        staged = _staged_file(staging_dir, requirement)
        if staged:
            resolved.append(
                DataSource(
                    name=requirement,
                    kind=KIND_REAL_LOCAL,
                    description=requirement,
                    local_path=str(staged),
                    checksum=checksum_of(staged),
                    reason=f"staged locally at {staged}",
                )
            )
            continue

        restricted = _match(RESTRICTED_SOURCES, requirement)
        if restricted:
            resolved.append(
                DataSource(
                    name=requirement,
                    kind=KIND_SURROGATE,
                    description=requirement,
                    reason=(
                        f"{restricted[1]}. No staged file was found, so a documented "
                        "surrogate is generated instead."
                    ),
                )
            )
            continue

        hint = _match(OPEN_SOURCE_HINTS, requirement)
        if hint and network:
            resolved.append(
                DataSource(
                    name=requirement,
                    kind=KIND_REAL_DOWNLOAD,
                    description=requirement,
                    uri=hint[2],
                    reason=f"open source {hint[1]}, reachable from this node",
                    hints=[f"{hint[1]}: {hint[2]}"],
                )
            )
            continue

        reason = (
            f"open source {hint[1]} identified, but this node has no outbound network"
            if hint
            else "no open source identified for this input and nothing staged locally"
        )
        resolved.append(
            DataSource(
                name=requirement,
                kind=KIND_SURROGATE,
                description=requirement,
                reason=reason + "; a documented surrogate is generated instead",
                hints=[f"{hint[1]}: {hint[2]}"] if hint else [],
            )
        )
    return resolved


def verdict(sources: list[DataSource]) -> str:
    """The methodological validity stamp — computed, never asked of the model."""
    if not sources:
        return VERDICT_SURROGATE
    real = [s for s in sources if s.is_real]
    if len(real) == len(sources):
        return VERDICT_EVIDENCE
    return VERDICT_MIXED if real else VERDICT_SURROGATE


def all_real(sources: list[DataSource]) -> bool:
    return bool(sources) and all(s.is_real for s in sources)


def write_provenance(sources: list[DataSource], path: Path) -> dict[str, Any]:
    """Record every input's origin next to the results it produced."""
    document = {
        "inputs": [s.to_dict() for s in sources],
        "methodological_validity": verdict(sources),
        "all_inputs_real": all_real(sources),
        "surrogate_count": sum(1 for s in sources if s.kind == KIND_SURROGATE),
    }
    path.write_text(json.dumps(document, indent=2))
    return document


def stamp_evaluation(
    results: dict[str, Any],
    sources: list[DataSource],
    *,
    success_criteria: str,
    refute_criteria: str,
) -> dict[str, Any]:
    """Attach the evaluation block, withholding a verdict when the data is not real.

    This is the hard rule, enforced here in Python rather than requested in a
    prompt: on surrogate data, `hypothesis_outcome` is `not_assessable`, full
    stop. There is no phrasing of a prompt that makes that guarantee.
    """
    evaluation: dict[str, Any] = {
        "methodological_validity": verdict(sources),
        "success_criteria": success_criteria,
        "refute_criteria": refute_criteria,
        "metrics": results.get("metrics", {}),
    }

    if all_real(sources):
        evaluation["hypothesis_outcome"] = results.get("hypothesis_outcome", "inconclusive")
        evaluation["meets_success_criteria"] = bool(results.get("meets_success_criteria", False))
    else:
        evaluation["hypothesis_outcome"] = "not_assessable"
        evaluation["meets_success_criteria"] = False
        evaluation["withheld_because"] = (
            "One or more inputs are synthetic surrogates. The metrics below describe "
            "the pipeline's behaviour on generated data and say nothing about the "
            "real-world hypothesis. See data_provenance.json."
        )
        # Whatever the generated code claimed about the hypothesis is discarded
        # rather than reported alongside a caveat — a number next to a warning
        # still gets quoted without the warning.
        evaluation["model_reported_outcome_discarded"] = results.get("hypothesis_outcome")
    return evaluation


def prompt_block(sources: list[DataSource]) -> str:
    """What the code generator is told about its inputs.

    Surrogates are named as surrogates in the prompt too, with the instruction to
    write and label a generator — not to pretend a file will be there.
    """
    if not sources:
        return "No data requirements were resolved; synthesize all inputs and label them clearly."

    lines = ["The following inputs have already been resolved. Use exactly these, and nothing else:"]
    for index, source in enumerate(sources, start=1):
        lines.append(f"\n{index}. {source.name}")
        if source.kind == KIND_REAL_LOCAL:
            lines.append(f"   REAL, already on disk at: {source.local_path}")
            lines.append("   Read it directly. Do not download anything for this input.")
        elif source.kind == KIND_REAL_DOWNLOAD:
            lines.append(f"   REAL, fetch from: {source.uri}")
            lines.append(
                "   Wrap the fetch in try/except. On failure, raise a clear error — do NOT "
                "silently fall back to made-up numbers."
            )
        else:
            lines.append(f"   SURROGATE — {source.reason}")
            lines.append(
                "   Write an explicit, seeded generator function named `synthesize_<input>` with a "
                "docstring stating it is synthetic and why. Give it realistic ranges and a "
                "realistic correlation structure so the pipeline is exercised properly, and record "
                "`\"synthetic\": True` for this input in the results."
            )
    lines.append(
        "\nNever create a file under data/ and then read it back as if it were real, and never "
        "assume an unstated file already exists."
    )
    return "\n".join(lines)
