# src/multi_model_infer/__main__.py
from __future__ import annotations

from rich.console import Console

from .config import load_settings
from .runner import run_inference, print_summary

console = Console()


def main() -> None:
    try:
        settings = load_settings()
    except RuntimeError as e:
        console.print(f"[red]Configuration error:[/red] {e}")
        raise SystemExit(1)

    try:
        result = run_inference(settings)
    except Exception as e:
        console.print(f"[red]Fatal error:[/red] {e}")
        raise SystemExit(1)

    print_summary(result)


if __name__ == "__main__":
    main()