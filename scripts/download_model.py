"""
Download Qwen2.5-VL GGUF weights from Hugging Face.

Usage:
    python scripts/download_model.py
    python scripts/download_model.py --model 7b --quant Q5_K_M
"""
import argparse
import os
from pathlib import Path

MODELS_DIR = Path(__file__).parent.parent / "models"

AVAILABLE = {
    "7b": {
        "repo": "unsloth/Qwen2.5-VL-7B-Instruct-GGUF",
        "quants": {
            "Q4_K_M": "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf",
            "Q5_K_M": "Qwen2.5-VL-7B-Instruct-Q5_K_M.gguf",
            "Q8_0":   "Qwen2.5-VL-7B-Instruct-Q8_0.gguf",
        },
        # mmproj = multimodal projector, required for vision in llama.cpp
        "mmproj": "mmproj-F16.gguf",
    },
    "72b": {
        "repo": "unsloth/Qwen2.5-VL-72B-Instruct-GGUF",
        "quants": {
            "Q4_K_M": "Qwen2.5-VL-72B-Instruct-Q4_K_M.gguf",
        },
        "mmproj": "mmproj-F16.gguf",
    },
}


def download(model_size: str = "7b", quant: str = "Q4_K_M") -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise SystemExit("Install huggingface-hub: pip install huggingface-hub")

    spec = AVAILABLE.get(model_size)
    if spec is None:
        raise ValueError(f"Unknown model size '{model_size}'. Choose from: {list(AVAILABLE)}")

    filename = spec["quants"].get(quant)
    if filename is None:
        raise ValueError(f"Unknown quant '{quant}'. Choose from: {list(spec['quants'])}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dest = MODELS_DIR / filename
    if dest.exists():
        print(f"[skip] Already downloaded: {dest}")
        return dest

    print(f"Downloading {spec['repo']} / {filename} -> {dest}")
    hf_hub_download(repo_id=spec["repo"], filename=filename, local_dir=str(MODELS_DIR))
    print(f"[ok] Saved to {dest}")

    # Download mmproj (vision projector) — required for llama.cpp VL inference
    mmproj_name = spec.get("mmproj")
    if mmproj_name:
        mmproj_dest = MODELS_DIR / mmproj_name
        if mmproj_dest.exists():
            print(f"[skip] mmproj already present: {mmproj_dest.name}")
        else:
            print(f"Downloading mmproj {mmproj_name} ...")
            hf_hub_download(repo_id=spec["repo"], filename=mmproj_name, local_dir=str(MODELS_DIR))
            print(f"[ok] mmproj saved to {mmproj_dest}")

    return dest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="7b", choices=list(AVAILABLE))
    parser.add_argument("--quant", default="Q4_K_M")
    args = parser.parse_args()
    download(args.model, args.quant)
