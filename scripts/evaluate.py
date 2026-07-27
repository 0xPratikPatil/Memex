#!/usr/bin/env python3
"""CLI tool for running RAG pipeline evaluations.

Usage:
    python scripts/evaluate.py                          # Run full evaluation
    python scripts/evaluate.py --dataset eval.jsonl     # Use specific dataset
    python scripts/evaluate.py --top-k 10               # Set top-K
    python scripts/evaluate.py --compare report1.json report2.json
    python scripts/evaluate.py --add-query              # Interactive query addition
    python scripts/evaluate.py --query "my question"    # Single query eval
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def cmd_run(args: argparse.Namespace) -> None:
    """Run evaluation against the RAG pipeline."""
    from rag import config as eval_config
    from rag.pipeline import RAGEngine
    from rag.services.evaluation import EvalRunner

    engine = RAGEngine()
    try:
        runner = EvalRunner(engine)
        if args.dataset:
            runner.dataset.path = args.dataset
            runner.dataset.load()

        if not runner.dataset.queries:
            print("No evaluation queries found. Use --add-query to add some.")
            return

        print(f"Running evaluation with {len(runner.dataset.queries)} queries...")
        results = runner.run(top_k=args.top_k)

        print("\n=== Evaluation Results ===")
        for key, value in sorted(results.items()):
            if key.startswith("avg_"):
                print(f"  {key}: {value:.4f}")

        if "benchmarks" in results:
            print("\n=== Performance Benchmarks ===")
            for op, bench in results["benchmarks"].items():
                print(f"  {op}: avg={bench['avg_ms']:.1f}ms  p95={bench['p95_ms']:.1f}ms  n={bench['count']}")

        print(f"\nReport saved to {eval_config.EVAL_OUTPUT_DIR}/")
    finally:
        engine.close()


def cmd_single(args: argparse.Namespace) -> None:
    """Evaluate a single query."""
    from rag.pipeline import RAGEngine
    from rag.services.evaluation import EvalRunner

    engine = RAGEngine()
    try:
        runner = EvalRunner(engine)
        result = runner.run_single(query=args.query, top_k=args.top_k)

        print(f"\nQuery: {result['query']}")
        print(f"Latency: {result['latency_ms']:.1f}ms")
        print(f"Results: {result['results_count']}")
        if result.get("mrr") is not None:
            print(f"MRR: {result['mrr']:.4f}")
            print(f"Hit@{args.top_k}: {result.get(f'hit@{args.top_k}', 'N/A')}")
        print("\nTop results:")
        for r in result["results"][:5]:
            print(f"  {r['rrf_score']:.4f}  {r['source']}")
    finally:
        engine.close()


def cmd_compare(args: argparse.Namespace) -> None:
    """Compare two evaluation report files."""
    with open(args.report_a) as f:
        report_a = json.load(f)
    with open(args.report_b) as f:
        report_b = json.load(f)

    from rag.services.evaluation import EvalRunner

    label_a = args.label_a or Path(args.report_a).stem
    label_b = args.label_b or Path(args.report_b).stem
    comparison = EvalRunner.compare(report_a, report_b, label_a, label_b)

    print(f"\n=== A/B Comparison: {label_a} vs {label_b} ===")
    for metric, vals in comparison["metrics"].items():
        delta = vals["delta"]
        sign = "+" if delta >= 0 else ""
        print(f"  {metric}:  {vals[label_a]:.4f} -> {vals[label_b]:.4f}  ({sign}{delta:.4f})")


def cmd_add_query(args: argparse.Namespace) -> None:
    """Interactively add queries to the evaluation dataset."""
    from rag.services.evaluation import EvalDataset

    dataset = EvalDataset(args.dataset)
    dataset.load()

    print(f"Current dataset: {len(dataset.queries)} queries")
    print("Add evaluation queries (empty line to finish):\n")

    while True:
        query = input("Query: ").strip()
        if not query:
            break
        sources_raw = input("Expected sources (comma-separated, optional): ").strip()
        keywords_raw = input("Expected keywords (comma-separated, optional): ").strip()
        category = input("Category (optional): ").strip()
        difficulty = input("Difficulty (easy/medium/hard, optional): ").strip()

        sources = [s.strip() for s in sources_raw.split(",") if s.strip()] if sources_raw else []
        keywords = [k.strip() for k in keywords_raw.split(",") if k.strip()] if keywords_raw else []

        metadata: dict[str, str] = {}
        if category:
            metadata["category"] = category
        if difficulty:
            metadata["difficulty"] = difficulty

        dataset.add_query(query, sources, keywords, metadata if metadata else None)
        dataset.save()
        print(f"  Added. Total: {len(dataset.queries)} queries\n")

    print(f"Dataset saved with {len(dataset.queries)} queries to {args.dataset}")


def cmd_stats(args: argparse.Namespace) -> None:
    """Show statistics about the evaluation dataset."""
    from rag.services.evaluation import EvalDataset

    dataset = EvalDataset(args.dataset)
    dataset.load()

    print(f"\nDataset: {args.dataset}")
    print(f"Total queries: {len(dataset.queries)}")

    if not dataset.queries:
        return

    categories: dict[str, int] = {}
    difficulties: dict[str, int] = {}
    for q in dataset.queries:
        cat = q.metadata.get("category", "uncategorized")
        diff = q.metadata.get("difficulty", "unknown")
        categories[cat] = categories.get(cat, 0) + 1
        difficulties[diff] = difficulties.get(diff, 0) + 1

    print("\nBy category:")
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count}")

    print("\nBy difficulty:")
    for diff, count in sorted(difficulties.items()):
        print(f"  {diff}: {count}")

    with_sources = sum(1 for q in dataset.queries if q.expected_sources)
    with_keywords = sum(1 for q in dataset.queries if q.expected_keywords)
    print(f"\nWith expected sources: {with_sources}")
    print(f"With expected keywords: {with_keywords}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RAG Evaluation Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", help="Available commands")

    # run
    p_run = sub.add_parser("run", help="Run full evaluation")
    p_run.add_argument("--dataset", default="", help="Eval dataset path")
    p_run.add_argument("--top-k", type=int, default=10, help="Top-K results")
    p_run.add_argument("--output", default="", help="Output directory")

    # single
    p_single = sub.add_parser("single", help="Evaluate a single query")
    p_single.add_argument("--query", required=True, help="Query to evaluate")
    p_single.add_argument("--top-k", type=int, default=10, help="Top-K results")

    # compare
    p_compare = sub.add_parser("compare", help="Compare two evaluation reports")
    p_compare.add_argument("report_a", help="Path to first report")
    p_compare.add_argument("report_b", help="Path to second report")
    p_compare.add_argument("--label-a", default="", help="Label for first report")
    p_compare.add_argument("--label-b", default="", help="Label for second report")

    # add-query
    p_add = sub.add_parser("add-query", help="Interactively add queries")
    p_add.add_argument("--dataset", default="", help="Eval dataset path")

    # stats
    p_stats = sub.add_parser("stats", help="Show dataset statistics")
    p_stats.add_argument("--dataset", default="", help="Eval dataset path")

    args = parser.parse_args()

    # Apply defaults from config
    from rag import config

    if hasattr(args, "dataset") and not args.dataset:
        args.dataset = config.EVAL_DATASET_PATH
    if hasattr(args, "output") and args.output:
        config.EVAL_OUTPUT_DIR = args.output

    if args.command == "run":
        cmd_run(args)
    elif args.command == "single":
        cmd_single(args)
    elif args.command == "compare":
        cmd_compare(args)
    elif args.command == "add-query":
        cmd_add_query(args)
    elif args.command == "stats":
        cmd_stats(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
