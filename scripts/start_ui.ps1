# Launch the Streamlit UI for the Document Intelligence POC
# Usage: pwsh scripts/start_ui.ps1

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

# Activate venv if present
$venv = Join-Path $Root ".venv\Scripts\Activate.ps1"
if (Test-Path $venv) { & $venv }

Write-Host ""
Write-Host "  Document Intelligence — Qwen2.5-VL"
Write-Host "  ──────────────────────────────────────"
Write-Host "  UI:     http://localhost:8501" -ForegroundColor Green
Write-Host "  Modele: http://localhost:8080  (demarrez d'abord start_server.ps1)"
Write-Host "  MLflow: http://localhost:5000  (optionnel: pwsh scripts/start_mlflow.ps1)"
Write-Host ""

streamlit run app.py --server.port 8501 --server.headless false
