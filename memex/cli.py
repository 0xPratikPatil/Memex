"""CLI commands for Memex RAG."""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from collections import OrderedDict
from pathlib import Path

import typer
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from memex import __version__
from memex.engine.core.progress import FileProgress

app = typer.Typer(help="Memex RAG — CLI commands")
console = Console()

# Stage display: (icon, color, label)
_STAGE_STYLE: dict[str, tuple[str, str, str]] = {
    "Converting": ("⚙ ", "cyan", "Converting"),
    "Context": ("ctx", "blue", "Context"),
    "Metadata": ("meta", "magenta", "Metadata"),
    "Embedding": ("emb", "yellow", "Embedding"),
    "Storing": ("···", "green", "Storing"),
    "Deleting": ("del", "red", "Deleting"),
    "Done": ("✓", "green", "Done"),
    "Error": ("✗", "red", "Error"),
}

_MAX_VISIBLE_ROWS = 4


def _stage_label(stage: str, error: str = "") -> str:
    """Return a rich-styled stage label."""
    icon, color, label = _STAGE_STYLE.get(stage, ("?", "dim", stage))
    if stage == "Error" and error:
        return f"[{color}]{icon} {error}[/{color}]"
    return f"[{color}]{icon} {label}[/{color}]"


def _build_compact_status(
    active: OrderedDict[str, tuple[str, int, str]],
    completed: int,
    total: int,
) -> str:
    """Build compact single-line status for each active file.

    Args:
        active: Dict of {path: (stage, chunks, error)}.
        completed: Number of completed files.
        total: Total number of files.

    Returns:
        Multi-line string with one status line per visible file.
    """
    lines = []
    rows = list(active.items())
    visible = rows[-_MAX_VISIBLE_ROWS:]

    for path, (stage, chunks, error) in visible:
        fname = os.path.basename(path)
        icon, color, _label = _STAGE_STYLE.get(stage, ("?", "dim", stage))
        if stage == "Error":
            lines.append(f"  [{color}]{icon} {fname}: {error}[/{color}]")
        else:
            chunk_str = f" ({chunks})" if chunks > 0 else ""
            lines.append(f"  [{color}]{icon} {fname}{chunk_str}[/{color}]")

    hidden = len(rows) - len(visible)
    if hidden > 0:
        lines.append(f"  [dim]... and {hidden} more[/dim]")

    if total > 0:
        pct = completed / total * 100
        lines.append(f"  [progress.percentage]{pct:.1f}%[/progress.percentage] {completed}/{total}")

    return "\n".join(lines)



def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        stream=sys.stderr,
    )


def _progress_columns() -> list:
    return [
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
    ]


@app.command()
def ingest(
    path: str = typer.Argument(..., help="File or directory path to ingest"),
    recursive: bool = typer.Option(False, "--recursive", "-r", help="Recursively scan directories"),
    source_name: str | None = typer.Option(None, "--source-name", "-s", help="Source name for tracking"),
    config_path: str = typer.Option("config.yaml", "--config", "-c", help="Path to config.yaml"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
) -> None:
    """Ingest files or directories into the RAG knowledge base."""
    _setup_logging(verbose)

    from memex.engine.core.pipeline import RAGEngine
    from memex.engine.core.yaml_config import YamlConfig
    from memex.engine.ingestion.loader import parse_file

    target = Path(path)
    if not target.exists():
        console.print(f"[red]Error:[/red] path does not exist: {path}")
        raise typer.Exit(code=1)

    YamlConfig(config_path)

    if target.is_dir():
        files = sorted(target.rglob("*") if recursive else target.iterdir())
        files = [f for f in files if f.is_file()]
    else:
        files = [target]

    if not files:
        console.print("[yellow]No files found to ingest.[/yellow]")
        raise typer.Exit(code=1)

    engine = RAGEngine()
    engine._get_qdrant()

    total = len(files)
    ingested = 0
    total_chunks = 0
    errors: list[str] = []

    with Progress(*_progress_columns(), console=console) as progress:
        task = progress.add_task("Ingesting...", total=total)

        for file_path in files:
            progress.update(task, description=f"[bold blue]{file_path.name}[/bold blue] — Parsing")
            try:
                result = parse_file(str(file_path))
                if not result.ok:
                    errors.append(f"{file_path}: {result.status} — {result.errors}")
                    progress.update(task, advance=1, description=f"[red]{file_path.name} — Error[/red]")
                    continue

                progress.update(task, description=f"[bold blue]{file_path.name}[/bold blue] — Ingesting")
                content_hash = engine.compute_file_hash(result.markdown.encode())
                chunks = engine.ingest_text(
                    result.markdown,
                    source_identifier=str(file_path),
                    metadata={
                        "content_type": file_path.suffix.lstrip("."),
                        "content_hash": content_hash,
                        "source_name": source_name or target.name,
                    },
                    content_hash=content_hash,
                )
                ingested += 1
                total_chunks += chunks
                desc = f"[green]{file_path.name}[/green] — Done ({chunks} chunks)"
                progress.update(task, advance=1, description=desc)
            except Exception as exc:
                errors.append(f"{file_path}: {exc}")
                progress.update(task, advance=1, description=f"[red]{file_path.name} — Error[/red]")

    table = Table(title="Ingest Complete", show_header=False, title_style="bold")
    table.add_column("Metric", style="bold")
    table.add_column("Value")
    table.add_row("Ingested", f"[green]{ingested}[/green]")
    table.add_row("Errors", f"[red]{len(errors)}[/red]" if errors else "0")
    table.add_row("Total chunks", str(total_chunks))
    console.print(Panel(table))

    if errors:
        for err in errors:
            console.print(f"  [red]failed:[/red] {err}")
        raise typer.Exit(code=1)


@app.command()
def sync(
    source_name: str | None = typer.Option(None, "--source-name", "-s", help="Sync specific source only"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would change without writing"),
    config_path: str = typer.Option("config.yaml", "--config", "-c", help="Path to config.yaml"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
) -> None:
    """Sync collection against configured document sources."""
    _setup_logging(verbose)

    from memex.engine.core.yaml_config import YamlConfig
    from memex.engine.sources.sync import SyncStats
    from memex.engine.sources.sync import sync as rag_sync

    yaml_config = YamlConfig(config_path)

    active: OrderedDict[str, tuple[str, int, str]] = OrderedDict()
    total_files = 0
    completed_count = 0

    def _on_progress(p: FileProgress) -> None:
        nonlocal total_files, completed_count
        if p.total > 0:
            total_files = p.total
        completed_count = p.current

        if p.stage in ("Done", "Error"):
            if p.stage == "Error":
                active[p.path] = ("Error", 0, p.error)
            else:
                active[p.path] = ("Done", p.chunks, "")
        else:
            existing = active.get(p.path)
            chunks = existing[1] if existing else 0
            active[p.path] = (p.stage, chunks, "")

    with Live(_build_compact_status(active, 0, 0), console=console, refresh_per_second=8) as live:

        async def _run() -> SyncStats:
            return await rag_sync(yaml_config, source_name=source_name, dry_run=dry_run, progress_cb=_on_progress)  # type: ignore[return-value]

        stats = asyncio.run(_run())
        live.update(_build_compact_status(active, completed_count, total_files))

    prefix = "would " if dry_run else ""
    table = Table(title="Sync Complete", show_header=False, title_style="bold")
    table.add_column("Metric", style="bold")
    table.add_column("Value")
    table.add_row("Added", f"[green]{stats.added}[/green]")
    table.add_row(f"{prefix.title()}Changed", f"[yellow]{stats.changed}[/yellow]")
    table.add_row(f"{prefix.title()}Deleted", f"[red]{stats.deleted}[/red]" if stats.deleted else "0")
    table.add_row("Unchanged", str(stats.unchanged))
    table.add_row("Errors", f"[red]{len(stats.errors)}[/red]" if stats.errors else "0")
    console.print(Panel(table))

    if stats.errors:
        for err in stats.errors:
            console.print(f"  [red]failed:[/red] {err}")
        raise typer.Exit(code=1)


@app.command(name="eval")
def eval_cmd(
    golden_set: str = typer.Argument(..., help="Path to golden set YAML/JSON file"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of results per query"),
    compare_rerank: bool = typer.Option(False, "--compare-rerank", help="Compare with/without reranking"),
    config_path: str = typer.Option("config.yaml", "--config", "-c", help="Path to config.yaml"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
) -> None:
    """Evaluate retrieval quality against a golden set."""
    _setup_logging(verbose)

    from memex.engine.core.pipeline import RAGEngine
    from memex.engine.core.yaml_config import YamlConfig
    from memex.engine.evaluation.golden import GoldenSet
    from memex.engine.evaluation.runner import compute_all_metrics

    golden_path = Path(golden_set)
    if not golden_path.exists():
        console.print(f"[red]Error:[/red] golden set file not found: {golden_set}")
        raise typer.Exit(code=1)

    YamlConfig(config_path)

    if golden_path.suffix in (".yaml", ".yml"):
        golden = GoldenSet.from_yaml(str(golden_path))
    else:
        golden = GoldenSet.from_json(str(golden_path))

    engine = RAGEngine()
    engine._get_qdrant()

    total = len(golden.queries)

    if total == 0:
        console.print("[yellow]No queries in golden set.[/yellow]")
        raise typer.Exit(code=1)

    results_table = Table(title="Query Results")
    results_table.add_column("Query", max_width=40)
    results_table.add_column("Hits", justify="right")
    results_table.add_column("Precision", justify="right")
    results_table.add_column("Keywords", justify="right")

    total_hits = 0
    total_mrr = 0.0
    total_keyword_cov = 0.0

    with Progress(*_progress_columns(), console=console) as progress:
        task = progress.add_task("Evaluating...", total=total)

        for gq in golden.queries:
            progress.update(task, description=f"[bold blue]{gq.query[:50]}...[/bold blue]")

            search_results = engine.hybrid_search(gq.query, top_k=top_k)
            metrics = compute_all_metrics(
                retrieved=search_results,
                expected_sources=list(gq.expected_sources),
                expected_keywords=list(gq.expected_keywords) if gq.expected_keywords else None,
                k=top_k,
            )

            hits = int(metrics.get(f"hit@{top_k}", 0.0) * top_k)
            total_hits += hits
            total_mrr += metrics.get("mrr", 0.0)
            total_keyword_cov += metrics.get("keyword_coverage", 0.0)

            hit_str = f"{hits}/{len(gq.expected_sources)}" if gq.expected_sources else "—"
            kw_val = metrics.get("keyword_coverage", 0.0)
            results_table.add_row(gq.query[:40], hit_str, f"{metrics.get('mrr', 0.0):.1%}", f"{kw_val:.1%}")

            progress.update(task, advance=1)

    avg_mrr = total_mrr / total if total > 0 else 0.0
    avg_keyword = total_keyword_cov / total if total > 0 else 0.0

    agg_table = Table(title="Aggregate Metrics", show_header=False, title_style="bold")
    agg_table.add_column("Metric", style="bold")
    agg_table.add_column("Value")
    agg_table.add_row("Queries", str(total))
    agg_table.add_row("Total hits", str(total_hits))
    agg_table.add_row("Avg MRR", f"{avg_mrr:.1%}")
    agg_table.add_row("Avg keyword coverage", f"{avg_keyword:.1%}")

    console.print()
    console.print(results_table)
    console.print(Panel(agg_table))

    if compare_rerank:
        console.print("\n[yellow]Comparing with reranking disabled...[/yellow]")
        with Progress(*_progress_columns(), console=console) as progress:
            task = progress.add_task("Evaluating (no rerank)...", total=total)
            nr_hits = 0
            for gq in golden.queries:
                search_results = engine.hybrid_search(gq.query, top_k=top_k, rerank=False)
                metrics = compute_all_metrics(
                    retrieved=search_results,
                    expected_sources=list(gq.expected_sources),
                    k=top_k,
                )
                nr_hits += int(metrics.get(f"hit@{top_k}", 0.0) * top_k)
                progress.update(task, advance=1)

        console.print(f"With reranking: {total_hits} hits | Without: {nr_hits} hits | Delta: {total_hits - nr_hits:+d}")


@app.command()
def serve(
    config_path: str = typer.Option("config.yaml", "--config", "-c", help="Path to config.yaml"),
) -> None:
    """Start the MCP server (stdio transport)."""
    from memex.mcp.server import mcp

    mcp.run(transport="stdio")


@app.command()
def status(
    config_path: str = typer.Option("config.yaml", "--config", "-c", help="Path to config.yaml"),
) -> None:
    """Show processing status for all files."""
    from memex.engine.core import config
    from memex.engine.core.yaml_config import YamlConfig
    from memex.engine.sources.status_tracker import StatusTracker

    YamlConfig(config_path)

    from memex.engine.core.pipeline import RAGEngine

    engine = RAGEngine()
    qdrant = engine._get_qdrant()
    tracker = StatusTracker(qdrant, config.COLLECTION_NAME)

    summary = tracker.get_status_summary()

    table = Table(title="File Processing Status", show_header=False, title_style="bold")
    table.add_column("Status", style="bold")
    table.add_column("Count", justify="right")
    table.add_row("Pending", str(summary.get("pending", 0)))
    table.add_row("Processing", str(summary.get("processing", 0)))
    table.add_row("Done", f"[green]{summary.get('done', 0)}[/green]")
    table.add_row("Retrying", f"[yellow]{summary.get('retry', 0)}[/yellow]")
    table.add_row("Failed", f"[red]{summary.get('failed', 0)}[/red]" if summary.get("failed") else "0")

    console.print(Panel(table))


@app.callback(invoke_without_command=True)
def main(
    version: bool = typer.Option(False, "--version", "-V", help="Show version and exit"),
) -> None:
    """Memex RAG — CLI commands. Use `memex serve` to start the MCP server, or run subcommands directly."""
    if version:
        typer.echo(f"memex {__version__}")
        raise typer.Exit()


if __name__ == "__main__":
    app()
