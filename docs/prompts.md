# Prompt engineering guide — Qwen2.5-VL

## Principles

1. **JSON-only output** — Always instruct the model to return only valid JSON.
   The system prompt should state: *"Respond only with valid JSON — no prose, no markdown fences."*
   The parser in `src/pipeline/infer.py` strips fences, but clean output is more reliable.

2. **Temperature = 0.1** — Low temperature forces deterministic, structured output.
   Higher temperatures may produce creative but malformed JSON.

3. **Explicit schema** — Include the target JSON structure in the user prompt.
   Qwen VL follows explicit schemas much better than open-ended instructions.

4. **Field coverage** — Name every expected field explicitly, even empty ones.
   The model is less likely to hallucinate fields when given a fixed schema.

---

## Prompt templates by document type

### Invoice
```
Extract all data from this invoice.
Return JSON:
{
  "invoice_number": "", "date": "", "due_date": "",
  "vendor": "", "vendor_address": "",
  "client": "", "client_address": "",
  "total_amount": "", "vat_amount": "", "currency": "",
  "line_items": [{"description":"","qty":"","unit_price":"","total":""}]
}
```

### Bank statement
```
Extract all data from this bank statement.
Return JSON:
{
  "account_holder": "", "iban": "", "bank_name": "",
  "period": {"start": "", "end": ""},
  "opening_balance": "", "closing_balance": "",
  "transactions": [{"date":"","label":"","debit":"","credit":"","balance":""}]
}
```

### Generic document
```
Extract all structured information from this document.
Return JSON with keys: document_type, language, fields (dict), tables (list of lists), summary.
```

---

## Troubleshooting prompts

| Symptom | Fix |
|---|---|
| Model ignores schema | Repeat schema twice: once in system, once in user |
| Truncated JSON | Increase `MAX_TOKENS` (to 4096) |
| Wrong field names | Add examples: `"invoice_number": "INV-001"` |
| Hallucinated values | Add: *"If a field is not visible, use null."* |
| Mixed lang output | Add: *"All field values must be in the original document language."* |

---

## Fine-tuning notes

For domain-specific documents (e.g., financial filings, medical records):
- Collect 200–500 labeled image+JSON pairs
- Fine-tune with LoRA using `trl` SFTTrainer
- See `notebooks/finetune_qwen_vl.ipynb`
