# Models — Download Guide

The model weights are **not included in this repository** (files > 5 GB).
You must download them separately before running the app or the pipeline.

---

## Models used

| File | Size | Role |
|------|------|------|
| `Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf` | ~4.4 GB | Main vision-language model (quantized Q4_K_M) |
| `mmproj-F16.gguf` | ~1.3 GB | Multimodal projector — required for image understanding in llama.cpp |

Both files must be placed in the `models/` directory.

---

## Option 1 — Automatic download (recommended)

```bash
pip install huggingface-hub

# Default: Qwen2.5-VL 7B Q4_K_M
python scripts/download_model.py

# Other sizes / quantizations
python scripts/download_model.py --model 7b --quant Q5_K_M
python scripts/download_model.py --model 7b --quant Q8_0
python scripts/download_model.py --model 72b --quant Q4_K_M
```

The script downloads both the model and the mmproj projector automatically.

---

## Option 2 — Manual download

### 7B model (recommended for 8–16 GB VRAM)

**Source:** [unsloth/Qwen2.5-VL-7B-Instruct-GGUF](https://huggingface.co/unsloth/Qwen2.5-VL-7B-Instruct-GGUF)

```bash
# Via huggingface-hub CLI
huggingface-cli download unsloth/Qwen2.5-VL-7B-Instruct-GGUF \
    Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf \
    mmproj-F16.gguf \
    --local-dir models/
```

### 72B model (requires 40+ GB VRAM or CPU offload)

**Source:** [unsloth/Qwen2.5-VL-72B-Instruct-GGUF](https://huggingface.co/unsloth/Qwen2.5-VL-72B-Instruct-GGUF)

```bash
huggingface-cli download unsloth/Qwen2.5-VL-72B-Instruct-GGUF \
    Qwen2.5-VL-72B-Instruct-Q4_K_M.gguf \
    mmproj-F16.gguf \
    --local-dir models/
```

---

## Quantization guide

| Quant | Size (7B) | VRAM | Quality |
|-------|-----------|------|---------|
| Q4_K_M | ~4.4 GB | 6–8 GB | Good — recommended default |
| Q5_K_M | ~5.2 GB | 8–10 GB | Better |
| Q8_0 | ~7.7 GB | 10–12 GB | Near-lossless |

> **Intel Arc XPU users:** Q4_K_M runs well on Arc A770 (16 GB).
> See `add_oneapi_path.ps1` and `start_with_xpu.bat` for XPU setup.

---

## Verify download

```bash
python - <<'EOF'
from pathlib import Path
models = Path("models")
for f in ["Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf", "mmproj-F16.gguf"]:
    p = models / f
    status = f"{p.stat().st_size / 1e9:.1f} GB  OK" if p.exists() else "MISSING"
    print(f"{f}: {status}")
EOF
```

---

## Hugging Face token (private models / gated repos)

If prompted for authentication:

```bash
huggingface-cli login
# or
export HF_TOKEN=hf_your_token_here
```
