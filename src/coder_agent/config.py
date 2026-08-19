"""Central settings, loaded from environment variables / .env.

Frozen dataclass, read once at import. Every module that needs configuration
imports `settings` from here; nothing else reads `os.environ` directly, so a
test can swap the whole object with `dataclasses.replace`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    llm_base_url: str
    llm_api_key: str
    llm_model: str
    llm_temperature: float
    llm_top_p: float
    llm_max_tokens: int
    llm_context_window: int
    experiments_dir: str
    venv_root: str
    base_env: str
    experiment_gpus: str
    max_code_attempts: int
    max_env_repairs: int
    max_format_retries: int
    timeout_low: int
    timeout_medium: int
    timeout_high: int
    memory_limit_gb: int

    def timeout_for(self, complexity: str) -> int:
        """Wall-clock budget for one execution of a plan of this complexity."""
        return {
            "low": self.timeout_low,
            "medium": self.timeout_medium,
            "high": self.timeout_high,
        }.get((complexity or "medium").strip().lower(), self.timeout_medium)


def load_settings() -> Settings:
    return Settings(
        llm_base_url=os.environ.get("LLM_BASE_URL", "http://127.0.0.1:8000/v1"),
        llm_api_key=os.environ.get("LLM_API_KEY", "not-needed"),
        llm_model=os.environ.get("LLM_MODEL", "Qwen/Qwen3-Coder-30B-A3B-Instruct"),
        llm_temperature=float(os.environ.get("LLM_TEMPERATURE", "0.2")),
        llm_top_p=float(os.environ.get("LLM_TOP_P", "0.95")),
        llm_max_tokens=int(os.environ.get("LLM_MAX_TOKENS", "16384")),
        # Read client-side only, to bound how many completion tokens a request may
        # ask for given how long the prompt already is. MUST match the
        # --max-model-len the server was started with, or a long fix prompt gets a
        # 400 back instead of a completion.
        llm_context_window=int(os.environ.get("LLM_CONTEXT_WINDOW", "131072")),
        experiments_dir=os.environ.get("CODER_EXPERIMENTS_DIR", "experiments"),
        # Where the per-experiment overlay venvs live. Empty means "inside the
        # experiment directory", which is right on a laptop. On Barkla it should
        # point at localscratch (/tmp/users/$USER): scratch and fastscratch have
        # inode quotas (300k / 500k files) that a venv per experiment eats into,
        # localscratch has no quota and is the fastest filesystem on the node,
        # and the venv is rebuilt per job anyway so losing it costs nothing.
        venv_root=os.environ.get("CODER_VENV_ROOT", ""),
        # Empty = no pre-baked base env; every overlay venv starts bare. Set by
        # scripts/build_base_env.sh on the cluster.
        base_env=os.environ.get("CODER_BASE_ENV", ""),
        # Which GPU ordinals the generated experiment may see. vLLM sits on the
        # others, so this is never the full set on a 2-GPU node.
        experiment_gpus=os.environ.get("CODER_EXPERIMENT_GPUS", ""),
        max_code_attempts=int(os.environ.get("CODER_MAX_CODE_ATTEMPTS", "4")),
        max_env_repairs=int(os.environ.get("CODER_MAX_ENV_REPAIRS", "6")),
        max_format_retries=int(os.environ.get("CODER_MAX_FORMAT_RETRIES", "2")),
        timeout_low=int(os.environ.get("CODER_TIMEOUT_LOW", "300")),
        timeout_medium=int(os.environ.get("CODER_TIMEOUT_MEDIUM", "1800")),
        timeout_high=int(os.environ.get("CODER_TIMEOUT_HIGH", "7200")),
        memory_limit_gb=int(os.environ.get("CODER_MEMORY_LIMIT_GB", "32")),
    )


settings = load_settings()
