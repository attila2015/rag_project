"""
Data extraction using Qwen2.5-VL.

Extracts structured fields, key-value pairs, and tables from document images.
Prompt templates are adapted based on the document type detected by classify.py.
"""
from __future__ import annotations
import os
from pathlib import Path

from src.pipeline.infer import call_vlm, parse_json_from_response
from src.pipeline.prompt_registry import get_extract_prompts, EXTRACT_SYSTEM, EXTRACT_TEMPLATES
from src.utils.image_utils import load_and_encode
from src.utils.schema import ExtractionMetadata, ExtractionResult, LineItem

SYSTEM_PROMPT = EXTRACT_SYSTEM
# Rétrocompatibilité : _PROMPTS exposé pour finetune_xpu et tests
_PROMPTS: dict[str, str] = {k: v["user"] for k, v in EXTRACT_TEMPLATES.items()}

_MODEL_NAME = os.getenv("MODEL_NAME", "qwen2.5-vl-7b-q4_k_m")


def extract_document(
    image_path: str | Path,
    doc_type: str = "default",
) -> ExtractionResult:
    """
    Extract structured data from a document image.

    Args:
        image_path: path to the image file
        doc_type: document type (from classify.py) to select the right prompt

    Returns:
        ExtractionResult with fields, line_items, tables
    """
    image_b64 = load_and_encode(image_path)
    prompt = _PROMPTS.get(doc_type, _PROMPTS["default"])
    raw, latency_ms = call_vlm(image_b64, SYSTEM_PROMPT, prompt)

    data = parse_json_from_response(raw)

    line_items = [
        LineItem(**item) if isinstance(item, dict) else LineItem(description=str(item))
        for item in data.get("line_items", [])
    ]

    return ExtractionResult(
        document_type=data.get("document_type", doc_type),
        confidence=float(data.get("confidence", 0.0)),
        fields=data.get("fields", {}),
        line_items=line_items,
        tables=data.get("tables", []),
        raw_text_snippet=data.get("raw_text_snippet", ""),
        metadata=ExtractionMetadata(
            model=_MODEL_NAME,
            latency_ms=latency_ms,
            source_file=str(image_path),
        ),
    )
