"""CLI commands for Memex RAG."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import threading
import time
from collections import OrderedDict
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
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

# Icons use only widely-supported glyphs (DejaVu/Noto/common terminal fonts).
_STAGE_ICONS: dict[str, str] = {
    "Checking": "?",
    "Scanning": "≡",
    "Reconciling": "↻",
    "Hashing": "#",
    "Parsing": "↓",
    "Converting": "⚙",
    "OCR": "◎",
    "Chunking": "▶",
    "Context": ">",
    "Metadata": "✦",
    "Embedding": "◆",
    "Storing": "↑",
    "Deleting": "✕",
    "Done": "✓",
    "Skipped": "↷",
    "Error": "✗",
}


def _stage_label(stage: str) -> str:
    """Icon + stage name for the progress row stage column."""
    return f"{_STAGE_ICONS.get(stage, '·')} {stage}"


def _short_name(path: str, max_len: int = 40) -> str:
    """Basename truncated to max_len — long names must never wrap the row."""
    name = os.path.basename(path)
    if len(name) <= max_len:
        return name
    return f"{name[: max_len - 3]}..."


def _fmt_dur(seconds: float) -> str:
    """Human duration: 2.3s / 1m05s / 1h02m."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m{secs:02.0f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02}m"


class _ProgressTracker:
    """Overall + per-file tasks with a bounded rolling terminal-row window.

    Tracks per-file wall time (first activity → terminal stage) and the
    total run time for the final summary.
    """

    def __init__(self, progress: Progress, total: int | None) -> None:
        self._progress = progress
        self.overall = progress.add_task("[bold]Overall", total=total, stage="", detail="")
        self._file_tasks: dict[str, TaskID] = {}
        self._terminal_order: OrderedDict[str, TaskID] = OrderedDict()
        self._start_times: dict[str, float] = {}
        self.file_times: dict[str, float] = {}
        self.file_stage: dict[str, str] = {}
        self.done_files: set[str] = set()
        self._started_ts = time.monotonic()

    def set_total(self, total: int) -> None:
        self._progress.update(self.overall, total=total)

    def total_elapsed(self) -> float:
        return time.monotonic() - self._started_ts

    def _display_budget(self) -> int:
        """Max rows this display can show without scrolling the terminal.

        Overall + 2 queue rows are fixed; the rest is shared by active and
        terminal file rows. Eviction keeps the total inside the budget so
        the live region always redraws in place (never appends).
        """
        height = console.size.height or 24
        return max(6, height - 4)

    def _evict(self) -> None:
        while len(self._file_tasks) > self._display_budget():
            if not self._terminal_order:
                break
            src, tid = self._terminal_order.popitem(last=False)
            if src in self._file_tasks:
                with contextlib.suppress(Exception):
                    self._progress.remove_task(tid)
                del self._file_tasks[src]

    def mark_active(self, src: str, stage: str) -> None:
        if src not in self._start_times:
            self._start_times[src] = time.monotonic()
        tid = self._file_tasks.get(src)
        if tid is None:
            tid = self._file_tasks[src] = self._progress.add_task(
                _short_name(src), total=None, stage=_stage_label(stage), detail=""
            )
            self._evict()
        self._progress.update(tid, stage=_stage_label(stage))

    def mark_done(self, src: str, stage: str, detail: str = "") -> None:
        start = self._start_times.setdefault(src, time.monotonic())
        self.file_times[src] = time.monotonic() - start
        self.file_stage[src] = stage
        tid = self._file_tasks.get(src)
        if tid is None:
            tid = self._file_tasks[src] = self._progress.add_task(
                _short_name(src), total=None, stage=_stage_label(stage), detail=detail
            )
        self._progress.update(tid, total=1, completed=1, stage=_stage_label(stage), detail=detail)
        self._terminal_order[src] = tid
        self._evict()
        if src not in self.done_files:
            self.done_files.add(src)
            self._progress.update(self.overall, completed=len(self.done_files))


def _make_progress() -> Progress:
    """Progress with per-file rows + overall bar (determinate).

    Minimal pip/git-style layout — description · stage · live elapsed ·
    bar · percent. No spinner, no red pulse: per-file rows are
    indeterminate (total=None) and pulse in dim gray; the overall bar
    fills green as files finish. Elapsed time shows live on every row.
    Text columns use Column(overflow="ellipsis") so rows never wrap —
    line wrapping breaks live redraw and causes duplicated rows.
    """

    def _ellipsis_col(text_format: str, style: str = "none", min_width: int = 0) -> TextColumn:
        return TextColumn(
            text_format,
            style=style,
            table_column=Column(overflow="ellipsis", min_width=min_width or None),
        )

    return Progress(
        _ellipsis_col("[bold]{task.description}"),
        _ellipsis_col("{task.fields[stage]}", style="cyan", min_width=16),
        TimeElapsedColumn(),
        BarColumn(
            bar_width=12,
            complete_style="green",
            finished_style="green",
            pulse_style="grey50",
        ),
        TaskProgressColumn(),
        _ellipsis_col("{task.fields[detail]}"),
        console=console,
        refresh_per_second=10,
    )


class _QueueDisplay:
    """Live converter queue rows: which file is converting now, which wait.

    Polls ``GET {service}/queue`` (MarkItDown + OCR) in background threads
    and updates one Progress row per service. Rows sit right after the
    Overall row and are always visible. Conversions can finish in under a
    second, so the poll is fast and the last activity is remembered for a
    few seconds — the row shows "◎ now: file" while busy and "· done:
    file" right after, so work is always visible.
    """

    POLL_INTERVAL_S = 0.25
    LAST_ACTIVITY_HOLD_S = 5.0

    def __init__(self, progress: Progress) -> None:
        self._progress = progress
        self._stop = threading.Event()
        self._tasks: dict[str, TaskID] = {}
        self._last_seen: dict[str, float] = {}
        self._last_file: dict[str, str] = {}

    def start(self) -> None:
        from memex.engine.core import config as engine_config

        services = {
            "MarkItDown": engine_config.MARKITDOWN_URL,
            "OCR": engine_config.OCR_URL,
        }
        for label, base_url in services.items():
            task = self._progress.add_task(
                f"[bold]{label} queue",
                total=None,
                stage="· idle",
                detail="",
            )
            self._tasks[label] = task
            threading.Thread(
                target=self._poll,
                args=(label, base_url, task),
                daemon=True,
                name=f"{label.lower()}-queue-poll",
            ).start()

    def _poll(self, label: str, base_url: str, task: TaskID) -> None:
        import httpx

        url = f"{base_url.rstrip('/')}/queue"
        while not self._stop.wait(self.POLL_INTERVAL_S):
            try:
                resp = httpx.get(url, timeout=1.5)
                if resp.status_code != 200:
                    continue
                data = resp.json()
            except Exception:
                continue

            current = data.get("current")
            pending = list(data.get("pending") or [])
            if current or pending:
                self._last_seen[label] = time.monotonic()
                self._last_file[label] = current or (pending[0] if pending else "")
                detail = f"waiting: {', '.join(pending)}" if pending else ""
                self._progress.update(
                    task,
                    stage=f"◎ now: {_short_name(self._last_file[label])}" if current else "◎ starting…",
                    detail=detail,
                )
            else:
                last = self._last_seen.get(label, 0.0)
                if last and time.monotonic() - last < self.LAST_ACTIVITY_HOLD_S:
                    self._progress.update(
                        task,
                        stage=f"· done: {_short_name(self._last_file.get(label, ''))}",
                        detail="",
                    )
                else:
                    self._progress.update(task, stage="· idle", detail="")

    def stop(self) -> None:
        self._stop.set()


def _setup_logging(verbose: bool, *, quiet: bool = False) -> None:
    """Configure logging; quiet suppresses INFO noise during live displays."""
    from memex.engine.core.logging_setup import setup_logging

    setup_logging(verbose=verbose, level=logging.WARNING if (quiet and not verbose) else None)


def _progress_columns() -> list:
    return [
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
    ]


def _stop_running_syncs() -> None:
    """Stop every running `memex sync` process.

    Escalation: SIGINT (graceful, finishes the current file) → wait 10s →
    SIGTERM → wait 5s → SIGKILL. Per-file statuses are checkpointed, so a
    hard kill loses at most the file being processed at that moment.
    """
    import signal
    import subprocess

    me = os.getpid()

    def _pids() -> list[int]:
        try:
            out = subprocess.run(
                ["pgrep", "-f", "memex sync"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            pids = []
            for p in out.stdout.split():
                pid = p.strip()
                if not pid or pid == str(me):
                    continue
                try:
                    cmd = Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="ignore")
                except OSError:
                    continue
                if "--stop" in cmd:  # never signal ourselves
                    continue
                pids.append(int(pid))
            return pids
        except Exception:
            return []

    for sig, grace in ((signal.SIGINT, 10), (signal.SIGTERM, 5), (signal.SIGKILL, 0)):
        pids = _pids()
        if not pids:
            break
        for pid in pids:
            try:
                os.kill(pid, sig)
            except (ProcessLookupError, PermissionError, ValueError):
                continue
        console.print(
            f"  [dim]{sig.name} sent to {len(pids)} process(es)…[/dim]"
        )
        if grace:
            time.sleep(grace)

    remaining = _pids()
    if remaining:
        console.print(f"[red]✗ {len(remaining)} sync process(es) still running — check manually[/red]")
        raise typer.Exit(code=1)
    console.print("[bold green]✓ sync stopped[/]")
    console.print("  Run `memex sync` again to resume (pending files are re-processed).")
    raise typer.Exit(code=0)


def _print_run_summary(
    *,
    title: str,
    elapsed: float,
    stats_line: str,
    errors: list[str],
) -> None:
    """Minimal one-line summary — per-file times already live on their rows."""
    console.print()
    console.print(
        f"[bold green]✓ {title}[/] in [bold]{_fmt_dur(elapsed)}[/] — {stats_line}"
    )
    for err in errors:
        console.print(f"  [red]✗ {err}[/]")


def _stats_line_ingest(ingested: int, total_chunks: int, error_count: int) -> str:
    if error_count:
        return (
            f"[green]{ingested}[/] ingested · [green]{total_chunks}[/] chunks"
            f" · [red]{error_count} errors[/]"
        )
    return f"[green]{ingested}[/] ingested · [green]{total_chunks}[/] chunks · 0 errors"


def _stats_line_sync(stats, dry_run: bool) -> str:
    prefix = "would " if dry_run else ""
    changed_verb = "change" if dry_run else "changed"
    deleted_verb = "delete" if dry_run else "deleted"
    errors = (
        f" · [red]{len(stats.errors)} errors[/]" if stats.errors else " · 0 errors"
    )
    return (
        f"[green]{stats.added}[/] added · [yellow]{stats.changed}[/] {prefix}{changed_verb}"
        f" · [red]{stats.deleted}[/] {prefix}{deleted_verb} · {stats.unchanged} unchanged{errors}"
    )


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
        queue_display = _QueueDisplay(progress)
        queue_display.start()
        try:
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
        finally:
            queue_display.stop()

    _print_run_summary(
        title="ingest complete",
        elapsed=tracker.total_elapsed(),
        stats_line=_stats_line_ingest(ingested, total_chunks, len(errors)),
        errors=errors,
    )

    if errors:
        raise typer.Exit(code=1)


@app.command()
def sync(
    source_name: str | None = typer.Option(None, "--source-name", "-s", help="Sync specific source only"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would change without writing"),
    config_path: str = typer.Option("config.yaml", "--config", "-c", help="Path to config.yaml"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
    stop: bool = typer.Option(False, "--stop", help="Stop a running sync gracefully (SIGINT) and exit"),
) -> None:
    """Sync collection against configured document sources.

    Use --stop to terminate a running sync from another terminal — the
    sync finishes the in-flight file and exits (progress is saved).
    """
    _setup_logging(verbose, quiet=True)

    if stop:
        _stop_running_syncs()
        return

    from memex.engine.core.yaml_config import YamlConfig
    from memex.engine.sources.sync import SyncStats
    from memex.engine.sources.sync import sync as rag_sync

    yaml_config = YamlConfig(config_path)

    with _make_progress() as progress:
        tracker = _ProgressTracker(progress, total=None)
        queue_display = _QueueDisplay(progress)
        queue_display.start()
        try:
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

            # Run the sync in a worker thread with its own event loop. The
            # main thread waits on interruptible time.sleep(), so Ctrl+C
            # raises KeyboardInterrupt here instantly even while the sync
            # is blocked inside C-level futures (future.result() would
            # swallow the signal until the file finished).
            sync_holder: dict[str, SyncStats] = {}

            def _run_sync_thread() -> None:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    sync_holder["stats"] = loop.run_until_complete(_run())
                finally:
                    loop.close()

            worker = threading.Thread(target=_run_sync_thread, daemon=True, name="sync-worker")
            worker.start()
            try:
                while worker.is_alive():
                    time.sleep(0.5)
                stats = sync_holder["stats"]
            except KeyboardInterrupt:
                # Force-exit: pool threads may still be mid-LLM-call and the
                # interpreter would wait on them. Per-file statuses are
                # checkpointed — the next sync resumes pending files.
                console.print()
                console.print("[yellow]! sync interrupted — progress saved[/]")
                console.print("[yellow]  run `memex sync` again to resume[/]")
                os._exit(130)
        finally:
            queue_display.stop()
        if seen_total == 0:
            tracker.set_total(1)
            progress.update(tracker.overall, completed=1)

    _print_run_summary(
        title="sync complete",
        elapsed=tracker.total_elapsed(),
        stats_line=_stats_line_sync(stats, dry_run),
        errors=stats.errors,
    )

    if stats.errors:
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
