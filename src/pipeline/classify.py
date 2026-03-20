"""
Document classification using Qwen2.5-VL.

Determines document type (invoice, contract, bank statement, etc.)
and broad category (financial, legal, administrative, technical).
"""
from __future__ import annotations
from pathlib import Path

from src.pipeline.infer import call_vlm, parse_json_from_response
from src.pipeline.prompt_registry import get_classify_prompts
from src.utils.image_utils import load_and_encode
from src.utils.schema import ClassificationResult, ExtractionMetadata

SYSTEM_PROMPT, USER_PROMPT = get_classify_prompts()


def classify_document(image_path: str | Path) -> ClassificationResult:
    """
    Classify a document from an image file path.

    Args:
        image_path: path to the document image (jpg, png, etc.)

    Returns:
        ClassificationResult with document_type, category, confidence
    """
    image_b64 = load_and_encode(image_path)
    raw, latency_ms = call_vlm(image_b64, SYSTEM_PROMPT, USER_PROMPT)

    data = parse_json_from_response(raw)
    return ClassificationResult(
        document_type=data.get("document_type", "unknown"),
        category=data.get("category", ""),
        confidence=float(data.get("confidence", 0.0)),
        language=data.get("language", ""),
        metadata=ExtractionMetadata(
            model=Path(image_path).name,
            latency_ms=latency_ms,
            source_file=str(image_path),
        ),
    )
