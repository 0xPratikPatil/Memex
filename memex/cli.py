"""CLI commands for Memex RAG."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
from collections import OrderedDict
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Column, Table

from memex import __version__
from memex.engine.core.progress import FileProgress

app = typer.Typer(help="Memex RAG — CLI commands")
console = Console()

_TERMINAL_STAGES = ("Done", "Skipped", "Error")

# Completed file rows are kept in a small rolling window and older rows are
# removed. Rich Progress cannot clear a display taller than the terminal
# (vertical_overflow="visible") — unbounded rows make every refresh scroll
# instead of redraw in place.
_MAX_TERMINAL_ROWS = 8

_STAGE_ICONS: dict[str, str] = {
    "Checking": "·",
    "Scanning": "·",
    "Reconciling": "·",
    "Hashing": "#",
    "Parsing": "p",
    "Converting": "⚙",
    "OCR": "◎",
    "Chunking": "⚙",
    "Context": "ctx",
    "Metadata": "meta",
    "Embedding": "emb",
    "Storing": "···",
    "Deleting": "del",
    "Done": "✓",
    "Skipped": "↷",
    "Error": "✗",
}


def _stage_label(stage: str) -> str:
    """Icon + stage name for the progress row stage column."""
    return f"{_STAGE_ICONS.get(stage, '·')} {stage}"


class _ProgressTracker:
    """Overall + per-file tasks with a bounded rolling terminal-row window."""

    def __init__(self, progress: Progress, total: int | None) -> None:
        self._progress = progress
        self.overall = progress.add_task("[bold]Overall", total=total, stage="", detail="")
        self._file_tasks: dict[str, TaskID] = {}
        self._terminal_order: OrderedDict[str, TaskID] = OrderedDict()
        self.done_files: set[str] = set()

    def set_total(self, total: int) -> None:
        self._progress.update(self.overall, total=total)

    def _evict(self) -> None:
        while len(self._terminal_order) > _MAX_TERMINAL_ROWS:
            src, tid = self._terminal_order.popitem(last=False)
            if src in self._file_tasks:
                with contextlib.suppress(Exception):
                    self._progress.remove_task(tid)
                del self._file_tasks[src]

    def mark_active(self, src: str, stage: str) -> None:
        tid = self._file_tasks.get(src)
        if tid is None:
            tid = self._file_tasks[src] = self._progress.add_task(
                os.path.basename(src), total=None, stage=_stage_label(stage), detail=""
            )
        self._progress.update(tid, stage=_stage_label(stage))

    def mark_done(self, src: str, stage: str, detail: str = "") -> None:
        tid = self._file_tasks.get(src)
        if tid is None:
            tid = self._file_tasks[src] = self._progress.add_task(
                os.path.basename(src), total=None, stage=_stage_label(stage), detail=detail
            )
        self._progress.update(tid, total=1, completed=1, stage=_stage_label(stage), detail=detail)
        self._terminal_order[src] = tid
        self._evict()
        if src not in self.done_files:
            self.done_files.add(src)
            self._progress.update(self.overall, completed=len(self.done_files))


def _make_progress() -> Progress:
    """Progress with per-file rows (indeterminate) + overall bar (determinate).

    Text columns use Column(overflow="ellipsis") so rows never wrap —
    line wrapping breaks live redraw and causes duplicated rows.
    """

    def _ellipsis_col(text_format: str, style: str = "none") -> TextColumn:
        return TextColumn(text_format, style=style, table_column=Column(overflow="ellipsis"))

    return Progress(
        SpinnerColumn(),
        _ellipsis_col("[bold]{task.description}"),
        _ellipsis_col("{task.fields[stage]}", style="cyan"),
        BarColumn(bar_width=20),
        TaskProgressColumn(),
        _ellipsis_col("{task.fields[detail]}"),
        console=console,
        refresh_per_second=10,
    )


def _setup_logging(verbose: bool, *, quiet: bool = False) -> None:
    """Configure logging; quiet suppresses INFO noise during live displays."""
    from memex.engine.core.logging_setup import setup_logging

    setup_logging(verbose=verbose, level=logging.WARNING if (quiet and not verbose) else None)


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
    _setup_logging(verbose, quiet=True)

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

    with _make_progress() as progress:
        tracker = _ProgressTracker(progress, total=total)

        def _on_progress(p: FileProgress) -> None:
            if p.stage in _TERMINAL_STAGES:
                detail = ""
                if p.stage == "Error" and p.error:
                    detail = p.error[:60]
                elif p.chunks:
                    detail = f"{p.chunks} chunks"
                tracker.mark_done(p.path, p.stage, detail)
            else:
                tracker.mark_active(p.path, p.stage)

        for file_path in files:
            src = str(file_path)
            status_store.mark_pending(src, source_name=source_name or target.name)
            tracker.mark_active(src, "Checking")

            try:
                can_skip, chunk_count = engine.check_unmodified_local(src)
                if can_skip:
                    status_store.mark_skipped(src, reason="unchanged")
                    tracker.mark_done(src, "Skipped", f"{chunk_count} chunks")
                    continue

                tracker.mark_active(src, "Parsing")
                result = parse_file(src)
                if not result.ok:
                    err = f"{file_path}: {result.status} — {result.errors}"
                    errors.append(err)
                    status_store.mark_failed(src, str(result.errors))
                    tracker.mark_done(src, "Error", str(result.errors)[:60])
                    continue

                tracker.mark_active(src, "Hashing")
                content_hash = engine.compute_file_hash(result.markdown.encode())
                already, existing_chunks = engine.is_already_ingested(src, content_hash)
                if already:
                    status_store.mark_skipped(src, reason="dedup")
                    tracker.mark_done(src, "Skipped", f"{existing_chunks} chunks")
                    continue

                tracker.mark_active(src, "Converting")
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
                tracker.mark_done(src, "Done", f"{chunks} chunks")
            except Exception as exc:
                errors.append(f"{file_path}: {exc}")
                status_store.mark_failed(src, str(exc), exc=exc)
                tracker.mark_done(src, "Error", str(exc)[:60])

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
    _setup_logging(verbose, quiet=True)

    from memex.engine.core.yaml_config import YamlConfig
    from memex.engine.sources.sync import SyncStats
    from memex.engine.sources.sync import sync as rag_sync

    yaml_config = YamlConfig(config_path)

    with _make_progress() as progress:
        tracker = _ProgressTracker(progress, total=None)
        seen_total = 0

        def _on_progress(p: FileProgress) -> None:
            nonlocal seen_total
            if p.total > 0:
                seen_total = p.total
                tracker.set_total(p.total)
            if p.stage in _TERMINAL_STAGES:
                detail = ""
                if p.stage == "Error" and p.error:
                    detail = p.error[:60]
                elif p.chunks:
                    detail = f"{p.chunks} chunks"
                tracker.mark_done(p.path, p.stage, detail)
            else:
                tracker.mark_active(p.path, p.stage)

        async def _run() -> SyncStats:
            return await rag_sync(yaml_config, source_name=source_name, dry_run=dry_run, progress_cb=_on_progress)  # type: ignore[return-value]

        stats = asyncio.run(_run())
        if seen_total == 0:
            tracker.set_total(1)
            progress.update(tracker.overall, completed=1)

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
