# Prompt engineering guide — Qwen2.5-VL

All prompts are versioned in [`prompts/registry.json`](../prompts/registry.json).
This file documents the design decisions and tuning notes behind each template.

---

## Core principles

1. **JSON-only output** — System prompt must state: *"Respond only with valid JSON — no prose, no markdown fences."*
   The parser in `src/pipeline/infer.py` strips fences, but clean output is more reliable.

2. **Temperature = 0.1** — Forces deterministic, structured output. Higher values produce creative but malformed JSON.

3. **Explicit schema in the user prompt** — Include the target JSON structure with field names and types.
   Qwen VL follows explicit schemas far better than open-ended instructions.

4. **Null over hallucination** — For any field not visible in the image, instruct the model to return `null`.
   Add: *"If a field is not visible, use null — do not guess."*

---

## Primary use case — CAC 40 chart extraction

**Registry key:** `extract.chart` (v1.0.0)

The main task is reading candlestick chart screenshots (daily timeframe, TradingView source)
and extracting OHLC values, trend direction, and chart metadata.

### System prompt
```
You are an expert financial chart reader.
Analyze the chart image carefully: read the title/symbol, the timeframe selector,
the date axis, and the price scale.
Always respond with valid JSON only — no prose, no markdown fences.
```

### User prompt
```
Read this financial chart image and extract the visible price data.

Return JSON:
{
  "document_type": "chart",
  "confidence": <float 0.0-1.0>,
  "fields": {
    "symbol":     "<ticker visible on chart, e.g. TVC:CAC40>",
    "chart_date": "<YYYY-MM-DD of the active candle/cursor>",
    "timeframe":  "<1m|5m|15m|30m|1H|4H|1D|1W|1M>",
    "open":       "<opening price as string>",
    "high":       "<high price as string>",
    "low":        "<low price as string>",
    "close":      "<closing/last price as string>",
    "volume":     "<volume if visible, else null>",
    "chart_type": "<candlestick|bar|line>",
    "trend":      "<bullish|bearish|sideways>"
  }
}
```

### Expected output
```json
{
  "document_type": "chart",
  "confidence": 0.96,
  "fields": {
    "symbol": "TVC:CAC40",
    "chart_date": "2025-03-12",
    "timeframe": "1D",
    "open": "8012.5",
    "high": "8087.3",
    "low": "7968.1",
    "close": "8054.7",
    "volume": null,
    "chart_type": "candlestick",
    "trend": "bullish"
  }
}
```

### Eval fields (used for fine-tuning accuracy scoring)
`symbol`, `chart_date`, `open`, `high`, `low`, `close`

---

## Classification prompt

**Registry key:** `classify` (v1.0.0)

Determines the document type before routing to the correct extraction prompt.
For this project, the expected output is almost always `"document_type": "chart"`.

```
Classify this document.

Return a JSON object with exactly these fields:
{
  "document_type": "<chart|invoice|contract|bank_statement|report|form|receipt|id_document|letter|other>",
  "category":      "<financial|legal|administrative|technical|personal|other>",
  "confidence":    <float 0.0–1.0>,
  "language":      "<fr|en|de|es|other>",
  "notes":         "<optional short observation>"
}
```

---

## Secondary document types

The pipeline also supports invoices, bank statements, and contracts.
These are secondary use cases — not the focus of the fine-tuning dataset.

### Invoice (`extract.invoice`)
```
Extract all data from this invoice.
Return JSON:
{
  "document_type": "invoice",
  "confidence": <float>,
  "fields": {
    "invoice_number": "", "date": "", "due_date": "",
    "vendor": "", "vendor_address": "",
    "client": "", "client_address": "",
    "total_amount": "", "vat_amount": "", "currency": ""
  },
  "line_items": [{"description":"","qty":"","unit_price":"","total":""}],
  "raw_text_snippet": "<first 200 chars of visible text>"
}
```

### Bank statement (`extract.bank_statement`)
```
Extract all data from this bank statement.
Return JSON:
{
  "document_type": "bank_statement",
  "confidence": <float>,
  "fields": {
    "account_holder": "", "iban": "", "bank_name": "",
    "period_start": "", "period_end": "",
    "opening_balance": "", "closing_balance": "", "currency": ""
  },
  "line_items": [{"description":"<label>","qty":"<date>","unit_price":"<debit>","total":"<credit>"}]
}
```

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| Model ignores schema | Repeat schema twice: once in system prompt, once in user prompt |
| OHLC values swapped | Specify axes: *"The Y-axis is the price scale. Read high as the top of the candle body/wick."* |
| Wrong date extracted | Add: *"The chart_date is the date shown on the X-axis under the active cursor, format YYYY-MM-DD."* |
| `"trend": null` | Add one-shot example in the prompt: `"trend": "bullish"` |
| Truncated JSON | Increase `MAX_TOKENS` to 4096 |
| Hallucinated values | Add: *"If a field is not visible in the image, use null — do not guess."* |
| Mixed language output | Add: *"All field values must be exactly as they appear on the chart."* |

---

## Fine-tuning notes

The dataset (`finetuning/dataset/`) contains **100 CAC 40 daily chart screenshots**
with ground-truth OHLC values in `finetuning/dataset/labels.json`.

Fine-tuning target: improve accuracy on `open`, `high`, `low`, `close` vs base model.

- Eval metric: exact match on rounded values (±0.5 point tolerance)
- LoRA config: r=4, alpha=16, target modules = `q_proj`, `v_proj`
- Training script: `scripts/finetune_xpu.py`
- Expected gain: +15–25% exact match on chart_date and OHLC fields after 3 epochs

See `notebooks/finetune_qwen_vl.ipynb` for the full walkthrough.
