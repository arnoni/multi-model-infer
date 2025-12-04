# src/multi_model_infer/models.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class ModelSpec:
    key: str
    display_name: str
    model_id: str
    input_cost_per_1k: float  # USD per 1,000 input tokens
    output_cost_per_1k: float  # USD per 1,000 output tokens
    max_output_tokens: int = 512
    enabled: bool = True
    notes: str = ""


# --- Model definitions -------------------------------------------------------
# Pricing numbers are approximate public estimates as of late 2025 and may drift.
# All prices below are PER 1,000 TOKENS (converted from “per 1M tokens” articles).
# Sources (approx):
# - GPT-5.1, Gemini 3 Pro, Claude 4.5, Grok 4.1: :contentReference[oaicite:1]{index=1}
# - Qwen3 Max: :contentReference[oaicite:2]{index=2}
# - DeepSeek V3.2: :contentReference[oaicite:3]{index=3}
# - Kimi K2 Thinking: :contentReference[oaicite:4]{index=4}

MODELS: List[ModelSpec] = [
    ModelSpec(
        key="gpt-5.1",
        display_name="GPT 5.1",
        model_id="openai/gpt-5.1",
        input_cost_per_1k=1.25 / 1000.0,
        output_cost_per_1k=10.0 / 1000.0,
        max_output_tokens=512,
        notes="OpenAI frontier general model.",
    ),
    ModelSpec(
        key="gemini-3-pro",
        display_name="Gemini 3 Pro",
        model_id="google/gemini-3-pro-preview",
        input_cost_per_1k=2.0 / 1000.0,
        output_cost_per_1k=12.0 / 1000.0,
        max_output_tokens=512,
        notes="Google frontier multimodal model. :contentReference[oaicite:5]{index=5}",
    ),
    ModelSpec(
        key="claude-4.5-opus",
        display_name="Claude Opus 4.5",
        model_id="anthropic/claude-4.5-opus-20251124",
        input_cost_per_1k=3.0 / 1000.0,
        output_cost_per_1k=15.0 / 1000.0,
        max_output_tokens=512,
        notes="Anthropic high-end reasoning model. :contentReference[oaicite:6]{index=6}",
    ),
    ModelSpec(
        key="grok-4.1",
        display_name="Grok 4.1 Fast",
        model_id="x-ai/grok-4.1-fast",
        input_cost_per_1k=3.0 / 1000.0,
        output_cost_per_1k=15.0 / 1000.0,
        max_output_tokens=512,
        notes="xAI Grok 4.1 Fast on OpenRouter. :contentReference[oaicite:7]{index=7}",
    ),
    ModelSpec(
        key="qwen3-max",
        display_name="Qwen 3 Max",
        model_id="qwen/qwen3-max",
        input_cost_per_1k=0.861 / 1000.0,
        output_cost_per_1k=3.441 / 1000.0,
        max_output_tokens=512,
        notes="Alibaba Qwen3 Max on OpenRouter. :contentReference[oaicite:8]{index=8}",
    ),
    ModelSpec(
        key="deepseek-v3.2",
        display_name="DeepSeek V3.2",
        model_id="deepseek/deepseek-v3.2-exp",
        input_cost_per_1k=0.21 / 1000.0,
        output_cost_per_1k=0.32 / 1000.0,
        max_output_tokens=512,
        notes=(
            "DeepSeek V3.2 Exp on OpenRouter; very low cost, "
            "used as default merge model as well. :contentReference[oaicite:9]{index=9}"
        ),
    ),
    ModelSpec(
        key="kimi-k2-thinking",
        display_name="Kimi K2 Thinking",
        model_id="moonshotai/kimi-k2-thinking",
        input_cost_per_1k=0.60 / 1000.0,
        output_cost_per_1k=2.50 / 1000.0,
        max_output_tokens=512,
        notes="MoonshotAI Kimi K2 Thinking on OpenRouter. :contentReference[oaicite:10]{index=10}",
    ),
]

# Merge / “meta” model – cheap, used to fuse all answers into one text
MERGE_MODEL = ModelSpec(
    key="merge-deepseek-v3.2",
    display_name="DeepSeek V3.2 (merge step)",
    model_id="deepseek/deepseek-v3.2-exp",
    input_cost_per_1k=0.21 / 1000.0,
    output_cost_per_1k=0.32 / 1000.0,
    max_output_tokens=768,
    notes="Cheap summarizer / merger.",
)

DEFAULT_BUDGET_USD = 0.10


def approximate_token_count(char_count: int) -> int:
    """
    Very rough heuristic: ~4 characters per token.
    Works well enough for budgeting and guardrails.
    """
    return max(1, char_count // 4)


def estimate_total_cost_usd(
    models: List[ModelSpec],
    prompt_chars: int,
    merge_model: ModelSpec = MERGE_MODEL,
) -> float:
    """
    Estimate upper-bound cost for:
    - N models answering independently
    - One merge model consuming the prompt + all model outputs
    Assumes each model uses its max_output_tokens.
    """
    input_tokens = approximate_token_count(prompt_chars)

    total = 0.0
    for m in models:
        total += (input_tokens / 1000.0) * m.input_cost_per_1k
        total += (m.max_output_tokens / 1000.0) * m.output_cost_per_1k

    # Merge step: worst case, sees prompt + every model's full response.
    if models:
        merge_input_tokens = input_tokens + sum(m.max_output_tokens for m in models)
    else:
        merge_input_tokens = input_tokens

    total += (merge_input_tokens / 1000.0) * merge_model.input_cost_per_1k
    total += (merge_model.max_output_tokens / 1000.0) * merge_model.output_cost_per_1k

    return total


def enforce_budget(
    prompt_chars: int,
    budget_usd: float = DEFAULT_BUDGET_USD,
) -> Tuple[List[ModelSpec], float]:
    """
    Iteratively drop the most expensive model until estimated cost is <= budget.
    Returns:
        (selected_models, estimated_cost)
    Raises:
        RuntimeError if no model can fit the budget.
    """
    active = [m for m in MODELS if m.enabled]
    if not active:
        raise RuntimeError("No models enabled.")

    while True:
        total = estimate_total_cost_usd(active, prompt_chars)
        if total <= budget_usd:
            return active, total

        if len(active) == 1:
            # If even one model blows the budget, bail out.
            raise RuntimeError(
                f"Even single model '{active[0].display_name}' exceeds budget "
                f"of ${budget_usd:.2f} with current prompt size."
            )

        # Remove the single most expensive model and try again.
        def per_model_cost(m: ModelSpec) -> float:
            input_tokens = approximate_token_count(prompt_chars)
            return (
                (input_tokens / 1000.0) * m.input_cost_per_1k
                + (m.max_output_tokens / 1000.0) * m.output_cost_per_1k
            )

        active.sort(key=per_model_cost, reverse=True)
        dropped = active.pop(0)
        # Loop again with the remaining models.