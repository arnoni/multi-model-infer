# src/multi_model_infer/client.py
from __future__ import annotations

from typing import Any, Dict

from openai import OpenAI
from openai._base_client import SyncAPIError

from .config import Settings


def build_openrouter_client(settings: Settings) -> OpenAI:
    """
    OpenRouter is OpenAI-compatible; we just override base_url and headers. :contentReference[oaicite:11]{index=11}
    """
    client = OpenAI(
        api_key=settings.openrouter_api_key,
        base_url="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": settings.http_referer,
            "X-Title": settings.x_title,
        },
        max_retries=2,
        timeout=settings.request_timeout_seconds,
    )
    return client


class InferenceError(RuntimeError):
    pass


def safe_chat_completion(
    client: OpenAI,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    temperature: float = 0.7,
) -> str:
    """
    Thin wrapper with defensive error handling.
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    except SyncAPIError as e:
        raise InferenceError(f"OpenRouter API error for model={model}: {e}") from e
    except Exception as e:
        raise InferenceError(f"Unexpected error for model={model}: {e}") from e

    try:
        choice = response.choices[0]
        content = choice.message.content
        if isinstance(content, list):
            # Some models return a list of content blocks
            joined = "".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in content
            )
            text = joined.strip()
        else:
            text = (content or "").strip()
    except Exception as e:
        raise InferenceError(
            f"Malformed response structure for model={model}: {e}"
        ) from e

    if not text:
        raise InferenceError(f"Empty response content for model={model}")

    return text