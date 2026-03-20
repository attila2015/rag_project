"""
Pydantic schemas for pipeline outputs.
"""
from __future__ import annotations
from typing import Any
from datetime import datetime
from pydantic import BaseModel, Field


class ExtractionMetadata(BaseModel):
    model: str
    latency_ms: int
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    source_file: str = ""
    page: int = 1


class ClassificationResult(BaseModel):
    document_type: str            # e.g. "invoice", "contract", "bank_statement"
    category: str = ""            # e.g. "financial", "legal", "administrative"
    confidence: float = 0.0
    language: str = "fr"
    metadata: ExtractionMetadata | None = None


class LineItem(BaseModel):
    description: str = ""
    qty: float | str = ""
    unit_price: str = ""
    total: str = ""


class ExtractionResult(BaseModel):
    document_type: str = ""
    confidence: float = 0.0
    fields: dict[str, Any] = Field(default_factory=dict)
    line_items: list[LineItem] = Field(default_factory=list)
    tables: list[list[Any]] = Field(default_factory=list)
    raw_text_snippet: str = ""
    metadata: ExtractionMetadata | None = None


class PipelineResult(BaseModel):
    classification: ClassificationResult | None = None
    extraction: ExtractionResult | None = None
    success: bool = True
    error: str = ""
