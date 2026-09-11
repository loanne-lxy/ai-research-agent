"""Project config — loads secrets from .env (same convention as weekly-ai-report).

No API keys in code. .env lives in the project root (chmod 600, gitignored)
and is copied from the weekly-ai-report project's .env so both projects
share the same QWEN/EXA/GITHUB credentials.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class LLMConfig:
    base_url: str
    api_key: str
    model: str
    temperature: float
    max_tokens: int
    timeout: int


def load_llm_config() -> LLMConfig:
    base_url = os.environ.get("QWEN_BASE_URL", "http://10.70.222.26:4100/v1")
    api_key = os.environ.get("QWEN_API_KEY", "")
    if not api_key:
        raise RuntimeError("QWEN_API_KEY missing — check .env in project root")
    return LLMConfig(
        base_url=base_url,
        api_key=api_key,
        model=os.environ.get("QWEN_MODEL", "Qwen3.8-27B"),
        temperature=float(os.environ.get("QWEN_TEMPERATURE", "0.3")),
        max_tokens=int(os.environ.get("QWEN_MAX_TOKENS", "8192")),
        timeout=int(os.environ.get("QWEN_TIMEOUT", "600")),
    )


@dataclass(frozen=True)
class Credentials:
    exa_api_key: str
    github_token: str

    @classmethod
    def load(cls) -> "Credentials":
        return cls(
            exa_api_key=os.environ.get("EXA_API_KEY", ""),
            github_token=os.environ.get("GITHUB_TOKEN", ""),
        )