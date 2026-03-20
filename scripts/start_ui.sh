#!/usr/bin/env bash
# Launch the Streamlit UI
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

[ -f ".venv/bin/activate" ] && source ".venv/bin/activate"

echo ""
echo "  Document Intelligence — Qwen2.5-VL"
echo "  ────────────────────────────────────"
echo "  UI:     http://localhost:8501"
echo "  Modele: http://localhost:8080"
echo ""

streamlit run app.py --server.port 8501
