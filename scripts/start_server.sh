#!/usr/bin/env bash
# Start llama.cpp OpenAI-compatible server for Qwen2.5-VL
# Usage: bash scripts/start_server.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$SCRIPT_DIR/.."

# Load .env if present
if [ -f "$ROOT/.env" ]; then
    export $(grep -v '^#' "$ROOT/.env" | xargs)
fi

MODEL_PATH="${MODEL_PATH:-}"
N_GPU_LAYERS="${N_GPU_LAYERS:-35}"
N_CTX="${N_CTX:-8192}"
N_THREADS="${N_THREADS:-8}"
PORT=8080

# Auto-detect model if not set
if [ -z "$MODEL_PATH" ]; then
    MODEL_PATH=$(find "$ROOT/models" -name "*.gguf" | head -1)
    if [ -z "$MODEL_PATH" ]; then
        echo "[ERROR] No .gguf model found in $ROOT/models/"
        echo "        Run: python scripts/download_model.py"
        exit 1
    fi
fi

echo "[info] Model  : $MODEL_PATH"
echo "[info] GPU layers: $N_GPU_LAYERS"
echo "[info] Context: $N_CTX tokens"
echo "[info] Port   : $PORT"
echo ""

# Try llama-server (newer) then llama.cpp server binary
if command -v llama-server &>/dev/null; then
    BINARY="llama-server"
elif command -v ./llama-server &>/dev/null; then
    BINARY="./llama-server"
else
    # Fallback: Python llama-cpp-python server
    echo "[info] llama-server binary not found — using llama-cpp-python HTTP server"
    python -m llama_cpp.server \
        --model "$MODEL_PATH" \
        --n_gpu_layers "$N_GPU_LAYERS" \
        --n_ctx "$N_CTX" \
        --n_threads "$N_THREADS" \
        --port "$PORT" \
        --chat_format chatml
    exit 0
fi

$BINARY \
    --model "$MODEL_PATH" \
    --n-gpu-layers "$N_GPU_LAYERS" \
    --ctx-size "$N_CTX" \
    --threads "$N_THREADS" \
    --port "$PORT" \
    --host 0.0.0.0 \
    --mmproj "$(dirname "$MODEL_PATH")/mmproj-qwen2.5-vl.gguf" \
    --log-disable 2>/dev/null || true \
    --verbose-prompt
