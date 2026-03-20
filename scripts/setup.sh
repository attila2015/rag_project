#!/usr/bin/env bash
# POC Qwen VL — Setup automatisé (Linux/macOS)
# Usage: bash scripts/setup.sh [--gpu] [--skip-model] [--quant Q5_K_M]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

GPU=0; SKIP_MODEL=0; QUANT="Q4_K_M"; MODEL_SIZE="7b"

while [[ $# -gt 0 ]]; do
  case $1 in
    --gpu)          GPU=1 ;;
    --skip-model)   SKIP_MODEL=1 ;;
    --quant)        QUANT="$2"; shift ;;
    --model-size)   MODEL_SIZE="$2"; shift ;;
  esac; shift
done

ok()   { echo "  ✓ $*"; }
warn() { echo "  ⚠ $*"; }
step() { echo ""; echo "  [$1] $2"; echo "  $(printf '─%.0s' {1..50})"; }

echo ""
echo "  Document Intelligence — Qwen2.5-VL POC"
echo "  Setup automatisé"

step "1/6" "Vérification Python"
python3 --version || { echo "Python 3 requis"; exit 1; }
ok "$(python3 --version)"

step "2/6" "Environnement virtuel"
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
  ok "Créé .venv/"
else
  ok "Déjà existant .venv/"
fi
source .venv/bin/activate
ok "Activé"

step "3/6" "Installation des dépendances"
pip install --upgrade pip --quiet

if [ "$GPU" -eq 1 ]; then
  warn "Mode GPU — compilation avec CUDA"
  CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall --no-cache-dir --quiet
else
  pip install llama-cpp-python --quiet
fi

grep -v "^#\|torch\|transformers\|peft\|trl\|datasets\|accelerate\|bitsandbytes\|llama-cpp-python\|^$" \
  requirements.txt > /tmp/req_slim.txt
pip install -r /tmp/req_slim.txt --quiet
ok "Dépendances installées"

step "4/6" "Configuration .env"
if [ ! -f ".env" ]; then
  cp .env.example .env
  ok "Créé .env"
else
  ok ".env déjà présent"
fi

step "5/6" "Modèle GGUF"
if [ "$SKIP_MODEL" -eq 1 ]; then
  warn "Skipped. Placez un .gguf dans models/"
elif ls models/*.gguf 2>/dev/null | head -1 | grep -q .; then
  ok "Modèle déjà présent : $(ls models/*.gguf | head -1 | xargs basename)"
else
  echo "  Téléchargement Qwen2.5-VL $MODEL_SIZE $QUANT..."
  python scripts/download_model.py --model "$MODEL_SIZE" --quant "$QUANT"
  ok "Modèle téléchargé"
fi

step "6/6" "Résumé"
echo ""
echo "  Setup terminé !"
echo "  Terminal 1 : bash scripts/start_server.sh"
echo "  Terminal 2 : bash scripts/start_ui.sh"
echo "  UI         : http://localhost:8501"
