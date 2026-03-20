# POC — Document Intelligence with Qwen2.5-VL

Transform raw document images into structured, exploitable data using **Qwen2.5-VL** (vision + language) running fully locally via `llama.cpp`.

Primary use case: **CAC 40 chart analysis** — extract OHLC values, trend signals, and indicators from candlestick screenshots. The same pipeline handles invoices, forms, and any structured document.

---

## Overview

| Component | Technology |
|-----------|-----------|
| Vision-Language Model | Qwen2.5-VL-7B (Q4_K_M GGUF) |
| Inference server | llama.cpp (OpenAI-compatible API) |
| UI | Streamlit multi-page app |
| Fine-tuning | LoRA r=4 via IPEX-LLM on Intel Arc XPU |
| Dataset | 100 CAC 40 chart images + JSON labels |
| Export | JSON / CSV + PowerPoint report |

---

## Project structure

```
poc_qwen/
├── app.py                         # Streamlit entry point (server health + navigation)
├── pages/
│   ├── 01_Guide.py                # Setup guide and llama.cpp install instructions
│   ├── 02_Pipeline.py             # Document upload → classify → extract → export
│   └── 03_FineTuning.py           # 6-step LoRA fine-tuning workflow (Intel Arc XPU)
├── src/
│   ├── pipeline/
│   │   ├── infer.py               # Low-level VLM call (OpenAI-compatible client)
│   │   ├── classify.py            # Document type classification
│   │   ├── extract.py             # Structured field / table extraction
│   │   └── prompt_registry.py     # Load prompts from prompts/registry.json
│   └── utils/
│       ├── image_utils.py         # Image loading, resizing, base64 encoding
│       ├── schema.py              # Pydantic output schemas
│       ├── monitoring.py          # MLflow / latency tracking
│       └── vector_store.py        # Embedding store (ChromaDB)
├── scripts/
│   ├── download_model.py          # Download Qwen2.5-VL GGUF from Hugging Face
│   ├── start_server.ps1           # Start llama.cpp server (Windows PowerShell)
│   ├── start_server.sh            # Start llama.cpp server (Linux/macOS)
│   ├── capture_chart_dataset.py   # Auto-capture CAC 40 screenshots for dataset
│   ├── finetune_xpu.py            # Fine-tuning script (Intel Arc XPU via IPEX-LLM)
│   ├── convert_pdf.py             # PDF → image preprocessing
│   ├── setup.ps1 / setup.sh       # Environment bootstrap
│   └── start_mlflow.ps1           # Start MLflow tracking server
├── finetuning/
│   ├── dataset/
│   │   ├── images/                # 100 CAC 40 chart images (chart_cac40_YYYY-MM-DD.jpg)
│   │   └── labels.json            # OHLC + indicator ground truth for each image
│   └── ft_*/                      # Fine-tuned adapter checkpoints (git-ignored)
├── models/                        # GGUF weights — downloaded separately (git-ignored)
├── data/
│   ├── raw/                       # Input: PDF pages / scans / chart screenshots
│   ├── processed/                 # Normalized images ready for inference
│   ├── outputs/                   # Extracted JSON / CSV results
│   └── vectorstore/               # ChromaDB embeddings
├── prompts/
│   └── registry.json              # Prompt templates for each document type
├── reports/
│   └── generate_ppt.py            # Auto-generate PowerPoint report from results
├── notebooks/
│   └── finetune_qwen_vl.ipynb     # LoRA fine-tuning walkthrough
├── add_oneapi_path.ps1            # Add Intel oneAPI to PATH (required for XPU)
├── start_with_xpu.bat             # Launch llama.cpp with Intel Arc GPU offload
├── test_xpu.py                    # Verify XPU availability
├── example_inference.py           # Minimal inference example (no UI)
├── run_pipeline.py                # CLI entry point
├── MODELS.md                      # Model download guide
├── requirements.txt
└── .env.example
```

---

## Hardware requirements

| Config | Minimum | Recommended |
|--------|---------|-------------|
| RAM | 16 GB | 32 GB |
| VRAM (GPU) | 8 GB Q4_K_M | 16 GB Q5_K_M/Q8 |
| CPU fallback | Slow but works | — |
| Disk | 10 GB free | 20 GB |

> **Qwen2.5-VL 7B at Q4_K_M ≈ 4.4 GB VRAM.** Tested on Intel Arc A770 (16 GB) via IPEX-LLM.
> See `add_oneapi_path.ps1` and `start_with_xpu.bat` for Arc setup.

---

## Quick start

### 1. Install dependencies

```bash
cd poc_qwen
python -m venv .venv

# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Download the model

```bash
python scripts/download_model.py
```

Downloads `Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf` and `mmproj-F16.gguf` into `models/`.
See [MODELS.md](MODELS.md) for other sizes (72B, Q5_K_M, Q8_0) and manual download instructions.

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env — set MODEL_SERVER_URL, MODEL_NAME, TEMPERATURE, MAX_TOKENS
```

### 4. Start the llama.cpp server

```bash
# Windows PowerShell
scripts/start_server.ps1

# Intel Arc XPU (Windows)
start_with_xpu.bat

# Linux / macOS
bash scripts/start_server.sh
```

Server runs at `http://localhost:8080` (OpenAI-compatible `/v1` API).

### 5. Launch the Streamlit UI

```bash
streamlit run app.py --server.port 8505
```

Navigate to `http://localhost:8505`.

---

## Streamlit UI — pages

| Page | Description |
|------|-------------|
| **Home** (`app.py`) | Server health check, auto-start llama.cpp, navigation |
| **Pipeline** (`02_Pipeline.py`) | Upload image/PDF → classify → extract → JSON/CSV export |
| **Fine-tuning** (`03_FineTuning.py`) | 6-step guided LoRA workflow on Intel Arc XPU |
| **Guide** (`01_Guide.py`) | Setup instructions, llama.cpp install, XPU config |

### Pipeline page — 5-step flow

```
Upload → Classification → Extraction → Validation → Export
```

- Drag-drop any image or PDF
- Model detects document type (chart, invoice, form, …)
- Extracts structured fields as JSON
- Export to CSV or download JSON
- History of previous extractions in session

---

## CLI usage

```bash
# Extract OHLC data from a CAC 40 chart screenshot
python run_pipeline.py --input data/raw/chart_cac40_2025-03-12.jpg

# Extract only (skip classification step)
python run_pipeline.py --input data/raw/chart_cac40_2025-03-14.jpg --mode extract

# Batch: process all charts in a folder
python run_pipeline.py --input data/raw/ --mode extract
```

---

## Output example — CAC 40 chart

```json
{
  "document_type": "financial_chart",
  "confidence": 0.96,
  "fields": {
    "index": "CAC 40",
    "date": "2025-03-12",
    "open": 8012.5,
    "high": 8087.3,
    "low":  7968.1,
    "close": 8054.7,
    "trend": "bullish",
    "rsi_14": 58.2,
    "macd_signal": "buy"
  },
  "extraction_metadata": {
    "model": "qwen2.5-vl-7b-q4_k_m",
    "latency_ms": 3240,
    "timestamp": "2025-03-12T17:05:12"
  }
}
```

---

## Fine-tuning on Intel Arc XPU

The `03_FineTuning.py` page guides you through a 6-step LoRA fine-tuning workflow:

1. **Dataset** — review the 100 CAC 40 chart images + `finetuning/dataset/labels.json`
2. **Base model** — download Qwen2.5-VL-3B (lighter base for fine-tuning)
3. **Training config** — LoRA r=4, batch size, learning rate, epochs
4. **Training** — launch `scripts/finetune_xpu.py` via IPEX-LLM on Intel Arc
5. **Evaluation** — compare predictions vs ground-truth labels
6. **Export** — merge adapter, convert to GGUF for llama.cpp serving

**Estimated time:** 30–45 min on Intel Core Ultra 5 125H (Arc integrated GPU).

```bash
# Direct fine-tuning script
python scripts/finetune_xpu.py --epochs 3 --lora-r 4
```

---

## Dataset — CAC 40 charts

```
finetuning/dataset/
├── images/                        # 100 daily candlestick screenshots
│   ├── chart_cac40_2025-03-12.jpg
│   ├── chart_cac40_2025-03-14.jpg
│   └── …
└── labels.json                    # OHLC + indicators ground truth
```

Images are captured automatically with `scripts/capture_chart_dataset.py` (screenshots of Euronext / TradingView charts).

---

## Intel Arc XPU setup

```powershell
# 1. Add Intel oneAPI to PATH
.\add_oneapi_path.ps1

# 2. Verify Arc GPU is detected
python test_xpu.py

# 3. Start llama.cpp with XPU offload
.\start_with_xpu.bat
```

Requires [Intel oneAPI Base Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html) installed.

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| `Connection refused :8080` | Run `start_server.ps1` or `start_with_xpu.bat` first |
| `CUDA out of memory` | Reduce `--n-gpu-layers`, use Q4_K_S instead of Q4_K_M |
| `Image too large` | `src/utils/image_utils.py` auto-resizes to 1024px max |
| `JSON parse error` | Model returned malformed JSON — retry or lower `--temp 0.1` |
| Slow on CPU | Expected (5–30 s/image) — use GPU or XPU offload |
| `DLL load failed` (Windows) | Install Visual C++ Redistributable 2019+ |
| `oneAPI not found` | Run `add_oneapi_path.ps1` then restart terminal |
| XPU not detected | Check Intel Graphics driver ≥ 31.0.101.5186 |

---

## Model download

See [MODELS.md](MODELS.md) for full instructions including:
- Automatic download via `scripts/download_model.py`
- Manual `huggingface-cli` commands
- Quantization comparison table (Q4_K_M / Q5_K_M / Q8_0)
- Hugging Face token setup for gated repos
