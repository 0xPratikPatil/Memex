"""CLI commands for Memex RAG."""

from __future__ import annotations

import asyncio
import os
from collections import OrderedDict
from pathlib import Path

import typer
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text

from memex import __version__
from memex.engine.core.progress import FileProgress

app = typer.Typer(help="Memex RAG — CLI commands")
console = Console()

# Stage display: (icon, color, label)
_STAGE_STYLE: dict[str, tuple[str, str, str]] = {
    "Scanning": ("·", "dim", "Scanning"),
    "Reconciling": ("·", "dim", "Reconciling"),
    "Hashing": ("#", "blue", "Hashing"),
    "Parsing": ("p", "cyan", "Parsing"),
    "Converting": ("⚙ ", "cyan", "Converting"),
    "Chunking": ("⚙ ", "cyan", "Chunking"),
    "Context": ("ctx", "blue", "Context"),
    "Metadata": ("meta", "magenta", "Metadata"),
    "Embedding": ("emb", "yellow", "Embedding"),
    "Storing": ("···", "green", "Storing"),
    "Deleting": ("del", "red", "Deleting"),
    "Done": ("✓", "green", "Done"),
    "Skipped": ("↷", "cyan", "Skipped"),
    "Error": ("✗", "red", "Error"),
}


def _stage_label(stage: str, error: str = "") -> str:
    """Return a rich-styled stage label."""
    icon, color, label = _STAGE_STYLE.get(stage, ("?", "dim", stage))
    if stage == "Error" and error:
        return f"[{color}]{icon} {error}[/{color}]"
    return f"[{color}]{icon} {label}[/{color}]"


def _build_live_display(
    active: OrderedDict[str, tuple[str, int, str]],
    completed: int,
    total: int,
) -> Text:
    """Build a single Text object with all file lines + progress bar.

    Using Text (not Group) ensures Rich.Live.update() replaces cleanly.
    """
    display = Text()

    for path, (stage, chunks, error) in active.items():
        fname = os.path.basename(path)
        icon, color, label = _STAGE_STYLE.get(stage, ("?", "dim", stage))

        display.append(f"  {icon} ", style=color)
        display.append(f"{fname:<36s}", style="bold" if stage not in ("Done", "Skipped") else "")
        display.append(f" {label}", style=color)

        if stage == "Error" and error:
            display.append(f"  {error[:50]}", style="red")
        elif chunks > 0:
            display.append(f"  {chunks} chunks", style="dim")

        display.append("\n")

    # Overall progress bar
    if total > 0:
        pct = completed / total * 100
        filled = int(pct / 5)  # 20 chars wide
        bar = "━" * filled + "╸" + "─" * (20 - filled - 1)
        display.append(f"\n  {bar} {completed}/{total} ", style="bold")
        display.append(f"{pct:.0f}%", style="green" if pct >= 100 else "yellow")

    return display


def _setup_logging(verbose: bool) -> None:
    from memex.engine.core.logging_setup import setup_logging

    setup_logging(verbose=verbose)


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

    # Normalize to absolute real paths so dedup against stored chunks works
    # regardless of how the CLI was invoked (./docs vs /abs/path/docs).
    files = [Path(os.path.realpath(f)) for f in files]
    files = sorted(files)

    if not files:
        console.print("[yellow]No files found to ingest.[/yellow]")
        raise typer.Exit(code=1)

    engine = RAGEngine()

    total = len(files)
    ingested = 0
    total_chunks = 0
    errors: list[str] = []

    from memex.engine.ingestion.status import FileStatusStore

    status_store = FileStatusStore(engine._get_qdrant())

    active: OrderedDict[str, tuple[str, int, str]] = OrderedDict()
    completed_count = 0

    def _on_progress(p: FileProgress) -> None:
        nonlocal completed_count
        if p.total > 0:
            completed_count = p.current

        if p.stage in ("Done", "Error", "Skipped"):
            if p.stage == "Error":
                active[p.path] = ("Error", 0, p.error)
            else:
                active[p.path] = (p.stage, p.chunks, "")
        else:
            existing = active.get(p.path)
            chunks = existing[1] if existing else 0
            active[p.path] = (p.stage, chunks, "")

        live.update(_build_live_display(active, completed_count, total))

    with Live(_build_live_display(active, 0, total), console=console, refresh_per_second=8) as live:
        for file_path in files:
            src = str(file_path)
            status_store.mark_pending(src, source_name=source_name or target.name)
            active[src] = ("Checking", 0, "")
            live.update(_build_live_display(active, completed_count, total))

            try:
                # Pre-check 1: local file unchanged since last ingest (mtime+size)
                can_skip, chunk_count = engine.check_unmodified_local(src)
                if can_skip:
                    status_store.mark_skipped(src, reason="unchanged")
                    active[src] = ("Skipped", chunk_count, "")
                    completed_count += 1
                    live.update(_build_live_display(active, completed_count, total))
                    continue

                active[src] = ("Parsing", 0, "")
                live.update(_build_live_display(active, completed_count, total))
                result = parse_file(src)
                if not result.ok:
                    err = f"{file_path}: {result.status} — {result.errors}"
                    errors.append(err)
                    status_store.mark_failed(src, str(result.errors))
                    active[src] = ("Error", 0, str(result.errors))
                    completed_count += 1
                    live.update(_build_live_display(active, completed_count, total))
                    continue

                # Pre-check 2: same content hash already ingested
                active[src] = ("Hashing", 0, "")
                live.update(_build_live_display(active, completed_count, total))
                content_hash = engine.compute_file_hash(result.markdown.encode())
                already, existing_chunks = engine.is_already_ingested(src, content_hash)
                if already:
                    status_store.mark_skipped(src, reason="dedup")
                    active[src] = ("Skipped", existing_chunks, "")
                    completed_count += 1
                    live.update(_build_live_display(active, completed_count, total))
                    continue

                active[src] = ("Converting", 0, "")
                live.update(_build_live_display(active, completed_count, total))
                chunks = engine.ingest_text(
                    result.markdown,
                    source_identifier=src,
                    metadata={
                        "content_type": file_path.suffix.lstrip("."),
                        "content_hash": content_hash,
                        "source_name": source_name or target.name,
                    },
                    content_hash=content_hash,
                    progress_cb=_on_progress,
                )
                ingested += 1
                total_chunks += chunks
                status_store.mark_done(src, chunks=chunks)
                active[src] = ("Done", chunks, "")
                completed_count += 1
                live.update(_build_live_display(active, completed_count, total))
            except Exception as exc:
                errors.append(f"{file_path}: {exc}")
                status_store.mark_failed(src, str(exc), exc=exc)
                active[src] = ("Error", 0, str(exc))
                completed_count += 1
                live.update(_build_live_display(active, completed_count, total))

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

        if p.stage in ("Done", "Error", "Skipped"):
            if p.stage == "Error":
                active[p.path] = ("Error", 0, p.error)
            else:
                active[p.path] = (p.stage, p.chunks, "")
        else:
            existing = active.get(p.path)
            chunks = existing[1] if existing else 0
            active[p.path] = (p.stage, chunks, "")

        # Refresh the live display on every progress event. Without this the
        # Live view renders a static string and shows nothing during sync.
        live.update(_build_live_display(active, completed_count, total_files))

    with Live(_build_live_display(active, 0, 0), console=console, refresh_per_second=8) as live:

        async def _run() -> SyncStats:
            return await rag_sync(yaml_config, source_name=source_name, dry_run=dry_run, progress_cb=_on_progress)  # type: ignore[return-value]

        stats = asyncio.run(_run())
        live.update(_build_live_display(active, completed_count, total_files))

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
    status_filter: str | None = typer.Option(
        None,
        "--status",
        help="Filter by status (pending/processing/done/skipped/failed/retry)",
    ),
    limit: int = typer.Option(50, "--limit", "-n", help="Max records to show"),
) -> None:
    """Show processing status for all files."""
    from memex.engine.core.yaml_config import YamlConfig
    from memex.engine.ingestion.status import FileStatusStore

    YamlConfig(config_path)

    from memex.engine.core.pipeline import RAGEngine

    engine = RAGEngine()
    qdrant = engine._get_qdrant()
    store = FileStatusStore(qdrant)

    summary = store.get_summary()

    table = Table(title="File Processing Status", show_header=False, title_style="bold")
    table.add_column("Status", style="bold")
    table.add_column("Count", justify="right")
    table.add_row("Pending", str(summary.get("pending", 0)))
    table.add_row("Processing", str(summary.get("processing", 0)))
    table.add_row("Done", f"[green]{summary.get('done', 0)}[/green]")
    table.add_row("Skipped", f"[cyan]{summary.get('skipped', 0)}[/cyan]")
    table.add_row("Retrying", f"[yellow]{summary.get('retry', 0)}[/yellow]")
    table.add_row("Failed", f"[red]{summary.get('failed', 0)}[/red]" if summary.get("failed") else "0")

    console.print(Panel(table))

    records = store.list_records(status_filter=status_filter, limit=limit)
    if not records:
        console.print("[dim]No per-file records to show.[/dim]")
        return

    detail = Table(title="Per-File Status", title_style="bold")
    detail.add_column("File")
    detail.add_column("Status")
    detail.add_column("Stage")
    detail.add_column("Chunks", justify="right")
    detail.add_column("Error")
    for r in records:
        fname = os.path.basename(r.get("source", "")) or r.get("source", "")
        st = r.get("status", "")
        stage = r.get("stage", "")
        err = r.get("error", "") or ""
        if st == "done":
            st_disp = f"[green]{st}[/green]"
        elif st == "failed":
            st_disp = f"[red]{st}[/red]"
        elif st == "retry":
            st_disp = f"[yellow]{st}[/yellow]"
        else:
            st_disp = st
        detail.add_row(fname, st_disp, stage, str(r.get("chunks", "")), err[:80])
    console.print(detail)


@app.command()
def retry(
    config_path: str = typer.Option("config.yaml", "--config", "-c", help="Path to config.yaml"),
    source_filter: str | None = typer.Option(
        None,
        "--source-filter",
        "-s",
        help="Only retry files whose source contains this substring",
    ),
) -> None:
    """Retry failed files immediately (bypasses backoff)."""
    _setup_logging(False)

    from memex.engine.core.yaml_config import YamlConfig
    from memex.engine.ingestion.status import FileStatusStore

    YamlConfig(config_path)

    from memex.engine.core.pipeline import RAGEngine
    from memex.engine.sources.retry_queue import RetryQueue

    engine = RAGEngine()
    store = FileStatusStore(engine._get_qdrant())
    queue = RetryQueue(status_store=store)

    count = queue.reset_failed(status_filter=source_filter)
    if count:
        console.print(f"[green]Reset {count} failed file(s) to processing.[/green]")
    else:
        console.print("[dim]No failed files to retry.[/dim]")


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
