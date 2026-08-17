"""Single-file Marker conversion worker — runs as an isolated subprocess.

Loads the marker models, converts one document to markdown, writes the result
JSON and exits. Because it is a short-lived process, all model memory is
released on exit — an OOM or hang kills only this process, never the server.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--format", default="markdown")
    parser.add_argument("--mode", default="fast")
    parser.add_argument("--force-ocr", action="store_true")
    args = parser.parse_args()

    try:
        from marker.config.parser import ConfigParser
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict
        from marker.output import text_from_rendered

        config = {
            "output_format": args.format,
            "mode": args.mode,
            "force_ocr": args.force_ocr,
        }
        config_parser = ConfigParser(config)

        artifact_dict = create_model_dict()

        # In fast mode, skip table_rec_model + TableProcessor to save ~500MB
        # GPU VRAM.  TableProcessor requires table_rec_model; without it the
        # PdfConverter init fails with "Cannot resolve dependency".  Fast mode
        # handles tables via text extraction, not the table recognition model.
        processor_list = config_parser.get_processors()
        if args.mode == "fast" and "table_rec_model" in artifact_dict:
            import gc

            import torch
            from marker.processors.table import TableProcessor

            del artifact_dict["table_rec_model"]
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Build processor list excluding TableProcessor
            default_procs = list(PdfConverter.default_processors)
            processor_list = [
                f"{p.__module__}.{p.__qualname__}"
                for p in default_procs
                if p is not TableProcessor
            ]

        converter = PdfConverter(
            config=config_parser.generate_config_dict(),
            artifact_dict=artifact_dict,
            processor_list=processor_list,
            renderer=config_parser.get_renderer(),
            llm_service=config_parser.get_llm_service(),
        )

        rendered = converter(args.input)
        text, _, images = text_from_rendered(rendered)
        if not (text or "").strip():
            result = {"success": False, "error": "conversion produced empty output"}
        else:
            result = {
                "format": args.format,
                "output": text,
                "images": {k: "" for k in images},  # image bytes omitted — markdown embeds refs
                "metadata": getattr(rendered, "metadata", {}),
                "success": True,
            }
        Path(args.output).write_text(json.dumps(result))
        return 0
    except Exception as exc:
        result = {"success": False, "error": f"{type(exc).__name__}: {exc}"}
        with __import__("contextlib").suppress(OSError):
            Path(args.output).write_text(json.dumps(result))
        return 1


if __name__ == "__main__":
    sys.exit(main())
