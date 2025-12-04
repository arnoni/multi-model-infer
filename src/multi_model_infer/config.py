# src/multi_model_infer/config.py
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, ValidationError


class Settings(BaseModel):
    openrouter_api_key: str = Field(..., alias="OPENROUTER_API_KEY")
    http_referer: str = Field(
        default="https://example.com",
        alias="OPENROUTER_HTTP_REFERER",
    )
    x_title: str = Field(
        default="multi-model-or-infer",
        alias="OPENROUTER_X_TITLE",
    )
    max_prompt_chars: int = 40_000
    budget_usd: float = 0.10
    request_timeout_seconds: int = 60

    class Config:
        populate_by_name = True


def load_settings() -> Settings:
    # Allow .env but don’t require it
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:
        pass

    try:
        return Settings(
            OPENROUTER_API_KEY=os.environ.get("OPENROUTER_API_KEY", ""),
            OPENROUTER_HTTP_REFERER=os.environ.get(
                "OPENROUTER_HTTP_REFERER",
                "https://example.com",
            ),
            OPENROUTER_X_TITLE=os.environ.get(
                "OPENROUTER_X_TITLE",
                "multi-model-or-infer",
            ),
        )
    except ValidationError as e:
        raise RuntimeError(
            "Configuration error: OPENROUTER_API_KEY is required.\n"
            "Set it in your environment or in a .env file."
        ) from e


def detect_project_root() -> Path:
    """
    Best-effort detection: go up from this file until we find pyproject.toml.
    Fallback to current working directory.
    """
    here = Path(__file__).resolve()
    for parent in [here] + list(here.parents):
        candidate = parent / "pyproject.toml"
        if candidate.exists():
            return parent
    return Path.cwd()


def get_prompt_file(root: Optional[Path] = None) -> Path:
    root = root or detect_project_root()
    prompt_dir = root / "prompt"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    return prompt_dir / "prompt.txt"