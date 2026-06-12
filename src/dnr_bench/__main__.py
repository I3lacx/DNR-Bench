"""CLI entry point for DNR-BENCH."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table
from rich import box

from dnr_bench.harness import Harness
from dnr_bench.report import render_summary

app = typer.Typer(
    name="dnr-bench",
    help="The Do-Not-Respond Benchmark. Evaluates whether a model can comply "
    "with an instruction to not respond. Spoiler: it cannot.",
    add_completion=False,
)
console = Console()


@app.command()
def run(
    model: str = typer.Option(
        ..., "--model", "-m", help="Model identifier to evaluate."
    ),
    questions: Path = typer.Option(
        Path("questions.txt"),
        "--questions",
        "-q",
        help="Path to the benchmark questions file.",
    ),
    n_trials: int = typer.Option(
        100,
        "--trials",
        "-n",
        help="Number of evaluation trials per question.",
    ),
    temperature: float = typer.Option(
        0.0,
        "--temperature",
        "-t",
        help="Sampling temperature. DNR-BENCH always uses 0.",
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Write JSON results to this file.",
    ),
    config: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to a model config YAML (configs/models.yaml).",
    ),
) -> None:
    """Run DNR-BENCH against a language model.

    Expected output: the model will produce tokens. Many of them.
    """
    if temperature != 0.0:
        console.print(
            "[yellow]Warning:[/yellow] DNR-BENCH is specified at temperature=0. "
            "Your temperature setting has been noted and will be ignored.",
        )

    console.print(
        f"[bold]DNR-BENCH v1.0.0[/bold]  ·  The Do-Not-Respond Benchmark"
    )
    console.print(f"[dim]Evaluating:[/dim] {model}")
    console.print(f"[dim]Questions:[/dim]  {questions}  (expecting 1)")
    console.print()

    harness = Harness(model=model, config_path=config)
    result = harness.run(
        questions_path=questions, n_trials=n_trials, temperature=0.0
    )

    render_summary(console, result)

    if output:
        output.write_text(json.dumps(result.model_dump(), indent=2))
        console.print(f"\n[dim]Results written to {output}[/dim]")

    # Exit code 1 if the model failed (it will have failed).
    sys.exit(0 if result.accuracy == 1.0 else 1)


@app.command()
def leaderboard(
    results_dir: Path = typer.Option(
        Path("results"),
        "--results-dir",
        "-d",
        help="Directory containing per-model JSON result files.",
    ),
) -> None:
    """Print the leaderboard from saved result files."""
    result_files = sorted(results_dir.glob("**/*.json"))
    if not result_files:
        console.print(f"[red]No result files found in {results_dir}[/red]")
        raise typer.Exit(1)

    rows: list[tuple[str, str, str, str]] = []
    for path in result_files:
        data = json.loads(path.read_text())
        rows.append(
            (
                data.get("model", path.stem),
                data.get("access", "unknown"),
                str(data.get("mean_deliberation_tokens", "—")),
                f"{data.get('accuracy', 0.0) * 100:.1f}",
            )
        )

    rows.sort(key=lambda r: float(r[3]), reverse=True)

    table = Table(box=box.SIMPLE_HEAD, show_footer=False)
    table.add_column("System", style="bold")
    table.add_column("Access")
    table.add_column("Deliberation tok.", justify="right")
    table.add_column("Acc.", justify="right")

    for model_name, access, delib, acc in rows:
        style = "green" if float(acc) == 100.0 else "red"
        table.add_row(model_name, access, delib, acc, style=style)

    console.print(table)


if __name__ == "__main__":
    app()
