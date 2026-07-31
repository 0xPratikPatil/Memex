"""CLI commands for Memex RAG."""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

import typer

from memex import __version__

app = typer.Typer(help="Memex RAG — CLI commands")


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        stream=sys.stderr,
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
    _setup_logging(verbose)
    log = logging.getLogger("memex.cli.ingest")

    from rag.docling_client import parse_file
    from rag.pipeline import RAGEngine
    from rag.yaml_config import YamlConfig

    target = Path(path)
    if not target.exists():
        typer.echo(f"Error: path does not exist: {path}", err=True)
        raise typer.Exit(code=1)

    YamlConfig(config_path)

    if target.is_dir():
        log.info("Ingesting directory: %s (recursive=%s)", target, recursive)
        files = sorted(target.rglob("*") if recursive else target.iterdir())
        files = [f for f in files if f.is_file()]
    else:
        log.info("Ingesting file: %s", target)
        files = [target]

    if not files:
        typer.echo("No files found to ingest.", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"Found {len(files)} file(s) to ingest.")

    engine = RAGEngine()
    engine._get_qdrant()  # ensure collection exists

    ingested = 0
    errors: list[str] = []
    for file_path in files:
        try:
            result = parse_file(str(file_path))
            if not result.ok:
                errors.append(f"{file_path}: {result.status} — {result.errors}")
                continue
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
            log.info("Ingested %s (%d chunks)", file_path.name, chunks)
        except Exception as exc:
            errors.append(f"{file_path}: {exc}")

    typer.echo(f"Ingested: {ingested}, Errors: {len(errors)}")
    if errors:
        for err in errors:
            typer.echo(f"  failed: {err}", err=True)
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

    from rag.sync import sync as rag_sync
    from rag.yaml_config import YamlConfig

    yaml_config = YamlConfig(config_path)

    async def _run() -> object:
        return await rag_sync(yaml_config, source_name=source_name, dry_run=dry_run)

    stats = asyncio.run(_run())

    prefix = "would " if dry_run else ""
    typer.echo(
        f"added={stats.added} {prefix}changed={stats.changed} "
        f"{prefix}deleted={stats.deleted} unchanged={stats.unchanged} "
        f"errors={len(stats.errors)}"
    )
    if stats.errors:
        for err in stats.errors:
            typer.echo(f"  failed: {err}", err=True)
        raise typer.Exit(code=1)


@app.command()
def eval(
    golden_set: str = typer.Argument(..., help="Path to golden set YAML/JSON file"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of results per query"),
    compare_rerank: bool = typer.Option(False, "--compare-rerank", help="Compare with/without reranking"),
    config_path: str = typer.Option("config.yaml", "--config", "-c", help="Path to config.yaml"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
) -> None:
    """Evaluate retrieval quality against a golden set."""
    _setup_logging(verbose)

    from rag.yaml_config import YamlConfig

    golden_path = Path(golden_set)
    if not golden_path.exists():
        typer.echo(f"Error: golden set file not found: {golden_set}", err=True)
        raise typer.Exit(code=1)

    YamlConfig(config_path)

    typer.echo(f"Loaded golden set: {golden_set}")
    typer.echo(f"Top-K: {top_k}, Compare rerank: {compare_rerank}")
    typer.echo("(evaluation framework not yet integrated)")


@app.command()
def serve(
    config_path: str = typer.Option("config.yaml", "--config", "-c", help="Path to config.yaml"),
) -> None:
    """Start the MCP server (stdio transport)."""
    from memex.server import mcp

    mcp.run(transport="stdio")


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
