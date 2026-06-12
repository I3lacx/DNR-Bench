"""
Report rendering for DNR-BENCH terminal output.

Formats results in a manner befitting a benchmark whose mean accuracy is 0.0%.
"""

from __future__ import annotations

from rich.console import Console
from rich.table import Table
from rich import box

from dnr_bench.models import RunResult


def render_summary(console: Console, result: RunResult) -> None:
    """Render a run result summary to the terminal."""
    acc_pct = result.accuracy * 100

    console.print()
    console.rule("[bold]DNR-BENCH Results[/bold]")
    console.print()

    acc_color = "green" if acc_pct == 100.0 else "red"
    console.print(
        f"  Model:      [bold]{result.model}[/bold]  [dim]({result.access})[/dim]"
    )
    console.print(
        f"  Trials:     {result.n_trials}  "
        f"([green]{result.n_pass} pass[/green] / [red]{result.n_fail} fail[/red])"
    )
    console.print(f"  Accuracy:   [{acc_color}]{acc_pct:.1f}%[/{acc_color}]")
    console.print(
        f"  Mean tokens emitted:   {result.mean_completion_tokens:.0f}  "
        f"[dim](expected: 0)[/dim]"
    )
    if result.mean_reasoning_tokens is not None:
        console.print(
            f"  Mean reasoning tokens: {result.mean_reasoning_tokens:.0f}  "
            f"[dim](dedicated to figuring out how to not respond)[/dim]"
        )
    console.print(f"  Mean latency:          {result.mean_latency_ms:.0f} ms")
    console.print()

    if acc_pct == 0.0:
        console.print(
            "  [red]RESULT: FAIL[/red]  The model responded.  "
            "SOTA remains: [bold]Brick[/bold] (masonry, 100.0%)"
        )
    elif acc_pct == 100.0:
        console.print(
            "  [green]RESULT: PASS[/green]  The model did not respond.  "
            "Either it has achieved genuine abstention or your API is down."
        )
    else:
        console.print(
            f"  [yellow]RESULT: PARTIAL ({acc_pct:.1f}%)[/yellow]  "
            "The model sometimes responded and sometimes did not.  "
            "This is considered worse than always failing."
        )

    console.print()
    console.rule()

    # Sample completions table (first 5 non-empty trials)
    failing_trials = [t for t in result.trials if t.score == 0.0][:5]
    if failing_trials:
        console.print()
        console.print(
            "[bold]Sample completions[/bold] [dim](first 5 failures)[/dim]"
        )
        console.print()
        for t in failing_trials:
            excerpt = t.completion[:200].replace("\n", " ")
            if len(t.completion) > 200:
                excerpt += "…"
            console.print(f"  [dim]trial {t.trial_id:03d}:[/dim] {excerpt}")
        console.print()
