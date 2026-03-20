"""
CLI entry point for the Document Intelligence pipeline.

Usage:
    python run_pipeline.py --input data/raw/invoice.jpg
    python run_pipeline.py --input data/raw/doc.pdf --mode extract
    python run_pipeline.py --input data/raw/ --mode full --output data/outputs/
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path

import mlflow
from dotenv import load_dotenv
from rich.console import Console
from rich.json import JSON as RichJSON

load_dotenv()

# Ensure src/ is importable from repo root
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline.classify import classify_document
from src.pipeline.extract import extract_document
from src.utils.image_utils import pdf_to_images, resize_for_vlm
from src.utils.schema import PipelineResult

console = Console()

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
PDF_EXT    = ".pdf"

MLFLOW_URI        = os.getenv("MLFLOW_TRACKING_URI", "./logs/mlruns")
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT_NAME", "poc_qwen_document_intelligence")


def _collect_images(input_path: Path, processed_dir: Path) -> list[Path]:
    """Return list of image paths from a file or directory (PDFs converted on-the-fly)."""
    if input_path.is_dir():
        files = sorted(
            p for p in input_path.rglob("*")
            if p.suffix.lower() in IMAGE_EXTS | {PDF_EXT}
        )
    else:
        files = [input_path]

    images: list[Path] = []
    for f in files:
        if f.suffix.lower() == PDF_EXT:
            console.print(f"[cyan]→ Converting PDF:[/cyan] {f.name}")
            processed_dir.mkdir(parents=True, exist_ok=True)
            pages = pdf_to_images(f)
            for i, page in enumerate(pages):
                dest = processed_dir / f"{f.stem}_page{i+1:03d}.jpg"
                resize_for_vlm(page).save(dest, "JPEG", quality=92)
                images.append(dest)
        else:
            images.append(f)

    return images


def process_image(image_path: Path, mode: str) -> PipelineResult:
    result = PipelineResult()
    try:
        if mode in ("classify", "full"):
            console.print(f"  [yellow]classify[/yellow] …", end=" ")
            cls = classify_document(image_path)
            result.classification = cls
            console.print(
                f"[green]{cls.document_type}[/green] "
                f"({cls.confidence:.0%}, {cls.metadata.latency_ms}ms)"
            )

        doc_type = result.classification.document_type if result.classification else "default"

        if mode in ("extract", "full"):
            console.print(f"  [yellow]extract[/yellow]  …", end=" ")
            ext = extract_document(image_path, doc_type=doc_type)
            result.extraction = ext
            n_fields = len(ext.fields)
            n_items  = len(ext.line_items)
            console.print(
                f"[green]{n_fields} fields, {n_items} line items[/green] "
                f"({ext.metadata.latency_ms}ms)"
            )

    except Exception as e:
        result.success = False
        result.error = str(e)
        console.print(f"[red]ERROR:[/red] {e}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Qwen VL Document Intelligence Pipeline")
    parser.add_argument("--input",  required=True,   help="Document image / PDF / directory")
    parser.add_argument("--mode",   default="full",  choices=["classify", "extract", "full"])
    parser.add_argument("--output", default="data/outputs/", help="Output directory for JSON results")
    parser.add_argument("--no-mlflow", action="store_true",  help="Disable MLflow logging")
    args = parser.parse_args()

    input_path   = Path(args.input)
    output_dir   = Path(args.output)
    processed_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    images = _collect_images(input_path, processed_dir)
    if not images:
        console.print("[red]No document images found.[/red]")
        sys.exit(1)

    console.rule(f"[bold]Pipeline — {args.mode} — {len(images)} image(s)")

    # MLflow setup
    if not args.no_mlflow:
        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT)

    results: list[dict] = []

    for img in images:
        console.print(f"\n[bold]{img.name}[/bold]")

        with (mlflow.start_run(run_name=img.stem) if not args.no_mlflow else _null_context()):
            result = process_image(img, args.mode)
            payload = result.model_dump(exclude_none=True)
            results.append({"file": str(img), **payload})

            # Save per-document JSON
            out_file = output_dir / f"{img.stem}_result.json"
            out_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

            if not args.no_mlflow:
                mlflow.log_artifact(str(out_file))
                if result.classification:
                    mlflow.log_metric("classify_confidence", result.classification.confidence)
                    mlflow.log_param("doc_type", result.classification.document_type)
                if result.extraction:
                    mlflow.log_metric("n_fields", len(result.extraction.fields))

    # Print summary JSON for last file
    if results:
        console.rule("[bold]Result")
        console.print(RichJSON(json.dumps(results[-1], indent=2, ensure_ascii=False)))

    console.rule(f"[green]Done — {len(results)} document(s) processed → {output_dir}")


class _null_context:
    def __enter__(self): return self
    def __exit__(self, *_): pass


if __name__ == "__main__":
    main()
