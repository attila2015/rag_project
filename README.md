# POC — Document Intelligence with Qwen 3 VL

Transform raw document images into structured, exploitable data using **Qwen 3 VL** (vision + language) running fully locally via `llama.cpp`.

---

## Project structure

```
poc_qwen/
├── data/
│   ├── raw/          # Input: PDF pages / scans / photos
│   ├── processed/    # Resized/normalized images ready for inference
│   └── outputs/      # Extracted JSON / CSV results
├── docs/             # Reference docs, prompt templates
├── logs/             # Inference logs, MLflow artifacts
├── models/           # GGUF model weights (downloaded separately)
├── notebooks/        # Exploration / fine-tuning experiments
├── scripts/
│   ├── start_server.sh        # Start llama.cpp server (Linux/macOS)
│   ├── start_server.ps1       # Start llama.cpp server (Windows)
│   ├── download_model.py      # Download Qwen3-VL GGUF from HF
│   └── convert_pdf.py         # PDF → image preprocessing
├── src/
│   ├── pipeline/
│   │   ├── classify.py        # Document classification
│   │   ├── extract.py         # Field / table extraction
│   │   └── infer.py           # Low-level VLM call
│   └── utils/
│       ├── image_utils.py     # Image loading, resizing, encoding
│       └── schema.py          # Pydantic output schemas
├── .env.example
├── requirements.txt
└── run_pipeline.py            # CLI entry point
```

---

## Hardware requirements

| Config | Minimum | Recommended |
|---|---|---|
| RAM | 16 GB | 32 GB |
| VRAM (GPU) | 8 GB (Q4_K_M) | 16 GB (Q5_K_M/Q8) |
| CPU fallback | Slow but works | — |
| Disk | 10 GB free | 20 GB |

> Qwen3-VL 7B at Q4_K_M ≈ 5.2 GB VRAM. Qwen3-VL 72B requires 40+ GB.

---

## Quick start

### 1. Install dependencies

```bash
cd poc_qwen
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Download Qwen3-VL GGUF model

```bash
python scripts/download_model.py
```

Or manually from Hugging Face:
- [Qwen/Qwen2.5-VL-7B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct-GGUF)
- Place `.gguf` file in `models/`

### 3. Install llama.cpp

```bash
# Option A: pip (CPU)
pip install llama-cpp-python

# Option B: pip (CUDA GPU)
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall --no-cache-dir

# Option C: prebuilt server binary
# https://github.com/ggerganov/llama.cpp/releases
```

### 4. Configure environment

```bash
cp .env.example .env
# Edit .env with your model path
```

### 5. Start the model server

```bash
# Windows PowerShell
scripts/start_server.ps1

# Linux/macOS
bash scripts/start_server.sh
```

Server runs at `http://localhost:8080`.

### 6. Run the pipeline

```bash
# Classify + extract a single document
python run_pipeline.py --input data/raw/my_document.pdf

# Extract only
python run_pipeline.py --input data/raw/invoice.jpg --mode extract

# Classify only
python run_pipeline.py --input data/raw/contract.png --mode classify
```

---

## Output example

```json
{
  "document_type": "invoice",
  "confidence": 0.94,
  "fields": {
    "invoice_number": "INV-2024-00142",
    "date": "2024-11-15",
    "vendor": "Acme Corp",
    "total_amount": "4 250,00 EUR",
    "vat": "708,33 EUR"
  },
  "line_items": [
    {"description": "Consulting services", "qty": 5, "unit_price": "750.00", "total": "3750.00"},
    {"description": "Travel expenses", "qty": 1, "unit_price": "500.00", "total": "500.00"}
  ],
  "raw_text_snippet": "FACTURE N° INV-2024-00142...",
  "extraction_metadata": {
    "model": "qwen2.5-vl-7b-q4_k_m",
    "latency_ms": 3420,
    "timestamp": "2025-01-15T10:32:45"
  }
}
```

---

## Optional: Fine-tuning Qwen3-VL

See `notebooks/finetune_qwen_vl.ipynb` for a LoRA fine-tuning walkthrough using:
- `transformers` + `peft` + `trl`
- Your labeled documents in `data/processed/`

Minimum for fine-tuning: 24 GB VRAM (A10G / RTX 3090+).

---

## Troubleshooting

| Error | Fix |
|---|---|
| `Connection refused :8080` | Server not started — run `start_server` script |
| `CUDA out of memory` | Reduce `--n-gpu-layers`, use smaller quant (Q4_K_S) |
| `Image too large` | Resize to max 1024px — `src/utils/image_utils.py` handles this |
| `JSON parse error` | Model returned malformed JSON — retry or adjust `--temp 0.1` |
| Slow on CPU | Normal — expect 5-30s/image on CPU; use GPU offload |
| `DLL load failed` (Windows) | Install Visual C++ Redistributable 2019+ |
