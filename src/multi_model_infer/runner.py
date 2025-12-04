# src/multi_model_infer/runner.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .config import Settings, get_prompt_file, detect_project_root
from .client import build_openrouter_client, safe_chat_completion, InferenceError
from .models import (
    ModelSpec,
    MODELS,
    MERGE_MODEL,
    enforce_budget,
    approximate_token_count,
)


console = Console()


@dataclass
class ModelResult:
    spec: ModelSpec
    text: str | None
    error: str | None


@dataclass
class RunResult:
    prompt: str
    used_models: List[ModelSpec]
    estimated_cost_usd: float
    per_model_results: List[ModelResult]
    merged_text: str


def load_prompt(settings: Settings) -> str:
    root = detect_project_root()
    prompt_path: Path = get_prompt_file(root)
    if not prompt_path.exists():
        raise RuntimeError(
            f"Prompt file not found at: {prompt_path}\n"
            "Create the file and put your prompt inside."
        )

    text = prompt_path.read_text(encoding="utf-8").strip()
    if not text:
        raise RuntimeError(f"Prompt file at {prompt_path} is empty.")

    if len(text) > settings.max_prompt_chars:
        raise RuntimeError(
            f"Prompt is too long ({len(text)} chars). "
            f"Configured max_prompt_chars={settings.max_prompt_chars}."
        )

    return text


def run_inference(settings: Settings) -> RunResult:
    prompt = load_prompt(settings)
    prompt_chars = len(prompt)

    # 1) Budget enforcement
    selected_models, estimated_cost = enforce_budget(
        prompt_chars,
        budget_usd=settings.budget_usd,
    )

    console.print(
        Panel.fit(
            f"Approx prompt length: [bold]{prompt_chars}[/bold] chars "
            f"(~{approximate_token_count(prompt_chars)} tokens)\n"
            f"Selected [bold]{len(selected_models)}[/bold] models "
            f"under budget ${settings.budget_usd:.2f} "
            f"(estimated total ≈ ${estimated_cost:.4f})",
            title="Budget Check",
        )
    )

    client = build_openrouter_client(settings)

    # 2) Call each model
    results: List[ModelResult] = []

    for spec in selected_models:
        console.print(f"[cyan]→ Querying {spec.display_name} ({spec.model_id})[/cyan]")
        try:
            text = safe_chat_completion(
                client=client,
                model=spec.model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=spec.max_output_tokens,
                temperature=0.7,
            )
            results.append(ModelResult(spec=spec, text=text, error=None))
            console.print(
                f"[green]✓ {spec.display_name} completed with "
                f"{len(text)} chars.[/green]"
            )
        except InferenceError as e:
            console.print(f"[red]✗ {spec.display_name} failed: {e}[/red]")
            results.append(ModelResult(spec=spec, text=None, error=str(e)))

    # 3) Build a combined “raw panel” text for the merge model
    combined_sections: List[str] = []
    for r in results:
        if r.text is not None:
            section = (
                f"### MODEL: {r.spec.display_name} ({r.spec.model_id})\n"
                f"{r.text}\n"
            )
        else:
            section = (
                f"### MODEL: {r.spec.display_name} ({r.spec.model_id})\n"
                f"[ERROR]: {r.error}\n"
            )
        combined_sections.append(section)

    combined_text = "\n\n".join(combined_sections)

    # 4) Merge step: ask DeepSeek V3.2 to synthesize a single coherent answer
    console.print(
        f"[magenta]→ Merging responses via {MERGE_MODEL.display_name} "
        f"({MERGE_MODEL.model_id})[/magenta]"
    )
    merge_prompt = (
        "You are an ensemble-aggregation model.\n\n"
        "You receive multiple LLM answers to the SAME user prompt.\n"
        "Your task:\n"
        "1. Read all model sections.\n"
        "2. Produce ONE high-quality, concise answer for the user.\n"
        "3. If models disagree, resolve conflicts explicitly.\n"
        "4. Do NOT mention individual models by name.\n\n"
        "----- ORIGINAL USER PROMPT -----\n"
        f"{prompt}\n\n"
        "----- MODEL RESPONSES -----\n"
        f"{combined_text}\n\n"
        "----- YOUR MERGED ANSWER -----\n"
    )

    merged_text = safe_chat_completion(
        client=client,
        model=MERGE_MODEL.model_id,
        messages=[{"role": "user", "content": merge_prompt}],
        max_tokens=MERGE_MODEL.max_output_tokens,
        temperature=0.4,
    )

    return RunResult(
        prompt=prompt,
        used_models=selected_models,
        estimated_cost_usd=estimated_cost,
        per_model_results=results,
        merged_text=merged_text,
    )


def print_summary(result: RunResult) -> None:
    table = Table(title="Models used (under budget)")
    table.add_column("Model", style="bold")
    table.add_column("ID")
    table.add_column("Max out tokens")
    for m in result.used_models:
        table.add_row(m.display_name, m.model_id, str(m.max_output_tokens))

    console.print(table)
    console.print(
        f"\n[bold]Estimated total cost (incl. merge step): "
        f"${result.estimated_cost_usd:.4f}[/bold]\n"
    )

    console.print(Panel(result.merged_text, title="Merged Answer", expand=True))