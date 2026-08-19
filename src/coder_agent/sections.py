"""The transport for model responses that carry source code.

Delimited blocks, nothing escaped:

    ===BEGIN imports===
    import pandas as pd
    ===END imports===

A real newline is a real newline and a real backslash is a real backslash. This
exists instead of JSON because putting generated Python inside a JSON string
value forces the model to hand-escape multi-line code on every response, and
that is the single most reliable way to break a code-generating model — one
stray `\\d` in a regex and the whole response is unparseable, including the
nine sections that were fine.

Reasoning traces are stripped defensively: a server started without the right
reasoning parser leaves `<think>...</think>` in `content`, where it would sit
outside the delimiters and confuse the parser about where a section starts.
"""

from __future__ import annotations

import re

BEGIN = "===BEGIN {name}==="
END = "===END {name}==="

_SECTION_RE = re.compile(
    r"^===BEGIN[ \t]+(?P<name>[A-Za-z0-9_]+)[ \t]*===[ \t]*\r?\n"
    r"(?P<body>.*?)"
    r"^===END[ \t]+(?P=name)[ \t]*===[ \t]*$",
    re.DOTALL | re.MULTILINE,
)

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_UNCLOSED_THINK_RE = re.compile(r"^\s*<think>.*?(?=^===BEGIN)", re.DOTALL | re.MULTILINE)
_FENCE_RE = re.compile(r"^[ \t]*```[A-Za-z0-9_+-]*[ \t]*$", re.MULTILINE)


class SectionParseError(ValueError):
    """The response did not carry the sections the caller asked for."""

    def __init__(self, message: str, *, missing: list[str], found: list[str]) -> None:
        super().__init__(message)
        self.missing = missing
        self.found = found


def strip_reasoning(text: str) -> str:
    """Remove `<think>` traces, closed or (when truncated) left hanging."""
    text = _THINK_RE.sub("", text)
    # A trace cut off by max_tokens never gets its closing tag; if real content
    # follows, drop everything up to the first delimiter rather than give up.
    if "<think>" in text and "===BEGIN" in text:
        text = _UNCLOSED_THINK_RE.sub("", text)
    return text.strip()


def parse_sections(text: str, required: list[str] | None = None) -> dict[str, str]:
    """Extract every delimited section. Raise if any `required` name is absent.

    Markdown fences are stripped *inside* section bodies only — models wrap code
    in ``` out of habit, and a fence line left in the body becomes a SyntaxError
    that costs a whole fix attempt to discover.
    """
    cleaned = strip_reasoning(text or "")
    found: dict[str, str] = {}
    for match in _SECTION_RE.finditer(cleaned):
        body = match.group("body")
        if _FENCE_RE.search(body):
            body = _FENCE_RE.sub("", body)
        # Keep interior blank lines and indentation; only trim the edges the
        # delimiters introduced.
        found[match.group("name")] = body.strip("\r\n")

    if required:
        missing = [name for name in required if name not in found or not found[name].strip()]
        if missing:
            raise SectionParseError(
                "Response is missing or empty for section(s): "
                + ", ".join(missing)
                + ". Sections present: "
                + (", ".join(sorted(found)) or "(none)"),
                missing=missing,
                found=sorted(found),
            )
    return found


def render_section(name: str, body: str) -> str:
    return f"{BEGIN.format(name=name)}\n{body}\n{END.format(name=name)}"


def render_sections(sections: dict[str, str]) -> str:
    """Build a response in the wire format — used for prompt examples and test fixtures.

    The format the prompt demonstrates is generated from the same function that
    the parser accepts, so the two cannot drift apart.
    """
    return "\n\n".join(render_section(name, body) for name, body in sections.items())


def format_template(field_names: list[str]) -> str:
    """The exact shape to show a model, built from the field list itself."""
    return render_sections({name: f"<{name} here>" for name in field_names})
