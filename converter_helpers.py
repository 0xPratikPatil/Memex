"""Mode-aware model/processor filtering for Marker converter.

build_converter_args() is the single source of truth for how fast vs
balanced mode affects which models and processors PdfConverter uses.

Standalone module — no memex.engine imports — so it can be tested
without pulling in numpy, qdrant_client, pydantic, etc.
"""

from __future__ import annotations

import gc
from typing import Any


def build_converter_args(
    mode: str,
    config_parser: Any,
    artifact_dict: dict[str, Any],
) -> tuple[dict[str, Any], list[str], Any]:
    """Prepare arguments for PdfConverter based on conversion mode.

    In fast mode, removes table_rec_model and TableProcessor to save GPU VRAM.
    In balanced mode, all models and processors are kept.

    Returns:
        (artifact_dict, processor_list, renderer)
    """
    from marker.converters.pdf import PdfConverter
    from marker.processors.table import TableProcessor

    processor_list = config_parser.get_processors()

    if mode == "fast" and "table_rec_model" in artifact_dict:
        del artifact_dict["table_rec_model"]
        gc.collect()

        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass  # torch not available (e.g. test environment)

        default_procs = list(PdfConverter.default_processors)
        processor_list = [
            f"{p.__module__}.{p.__qualname__}"
            for p in default_procs
            if p is not TableProcessor
        ]

    return artifact_dict, processor_list, config_parser.get_renderer()
