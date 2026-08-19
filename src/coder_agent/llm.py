"""The one place a model client is constructed.

Points at any OpenAI-compatible endpoint — in practice the vLLM server started
by `scripts/serve_vllm.sh` on the same node, reached over localhost. Swapping
models is an env var; swapping providers is this file and nothing else.

Deliberately no tool-calling: the agent loop is orchestrated in Python and the
model only ever emits delimited text sections (see `sections.py`). That removes
a dependency on a server-side tool-call parser matching the model's chat
template, which is a version-coupling that breaks silently and costs a run.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from .config import Settings, settings

logger = logging.getLogger(__name__)

# Rough chars-per-token for budgeting. Deliberately conservative: overestimating
# the prompt costs a few hundred completion tokens, while underestimating it
# gets a 400 back from the server instead of a completion.
CHARS_PER_TOKEN = 3.2
RESERVE_TOKENS = 512


class LLMError(RuntimeError):
    """The model server could not be reached, or refused the request."""


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0

    def add(self, prompt: int, completion: int) -> None:
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.calls += 1


class ChatModel:
    """A thin, retrying wrapper. One method: give it a prompt, get text back."""

    def __init__(self, config: Settings | None = None) -> None:
        self.settings = config or settings
        self.usage = Usage()
        self._client = None

    @property
    def client(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover - dependency is declared
                raise LLMError("the `openai` package is required to reach the model server") from exc
            self._client = OpenAI(
                base_url=self.settings.llm_base_url,
                api_key=self.settings.llm_api_key or "not-needed",
                timeout=1800.0,
                max_retries=0,  # retries are handled here, with logging
            )
        return self._client

    def bounded_max_tokens(self, prompt: str) -> int:
        """Completion budget that fits under the server's real context window.

        The fix prompts are the long ones — they carry the previous source, the
        traceback, the plan and the data provenance — so a fixed max_tokens plus
        a long prompt is exactly the combination that overflows.
        """
        estimated_prompt = int(len(prompt) / CHARS_PER_TOKEN) + RESERVE_TOKENS
        available = self.settings.llm_context_window - estimated_prompt
        if available < 1024:
            raise LLMError(
                f"prompt is too long for the context window: ~{estimated_prompt} tokens of "
                f"{self.settings.llm_context_window}. Shorten the evidence passed into it."
            )
        return max(1024, min(self.settings.llm_max_tokens, available))

    def complete(self, prompt: str, *, system: str = "", attempts: int = 3) -> str:
        """Send one prompt, return the response text. Retries transport failures only."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.settings.llm_model,
                    messages=messages,
                    temperature=self.settings.llm_temperature,
                    top_p=self.settings.llm_top_p,
                    max_tokens=self.bounded_max_tokens(prompt),
                )
            except LLMError:
                raise
            except Exception as exc:
                last_error = exc
                if attempt == attempts:
                    break
                delay = 2**attempt
                logger.warning("model call failed (%s); retrying in %ss", exc, delay)
                time.sleep(delay)
                continue

            usage = getattr(response, "usage", None)
            if usage:
                self.usage.add(usage.prompt_tokens or 0, usage.completion_tokens or 0)

            choice = response.choices[0]
            text = choice.message.content or ""
            if getattr(choice, "finish_reason", "") == "length":
                # Worth saying out loud: a truncated response is the most common
                # cause of a section that parses but ends mid-function.
                logger.warning(
                    "response hit the completion cap (%s tokens) and is probably truncated",
                    self.settings.llm_max_tokens,
                )
            return text

        raise LLMError(
            f"model server at {self.settings.llm_base_url} did not respond after "
            f"{attempts} attempts: {last_error}"
        )

    def health(self) -> tuple[bool, str]:
        """Is the server up and serving the model we think it is?

        Called once at startup: discovering the endpoint is wrong takes seconds
        here and forty minutes into a job otherwise.
        """
        try:
            models = self.client.models.list()
            served = [m.id for m in models.data]
        except Exception as exc:
            return False, f"cannot reach {self.settings.llm_base_url}: {exc}"
        if self.settings.llm_model not in served:
            return False, (
                f"server is up but serving {served}, not {self.settings.llm_model!r}. "
                "Set LLM_MODEL to the served id."
            )
        return True, f"serving {self.settings.llm_model}"


def get_chat_model(config: Settings | None = None) -> ChatModel:
    return ChatModel(config)
