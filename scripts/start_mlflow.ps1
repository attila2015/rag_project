# Start MLflow UI for the Document Intelligence POC
# Usage: pwsh scripts/start_mlflow.ps1 [-Port 5000]

param(
    [int]$Port = 5000
)

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

# Activate venv if present
$venv = Join-Path $Root ".venv\Scripts\Activate.ps1"
if (Test-Path $venv) { & $venv }

# Create mlruns dir if needed
$MlrunsDir = Join-Path $Root "logs\mlruns"
if (-not (Test-Path $MlrunsDir)) {
    New-Item -ItemType Directory -Path $MlrunsDir -Force | Out-Null
    Write-Host "  Cree: $MlrunsDir" -ForegroundColor Green
}

Write-Host ""
Write-Host "  MLflow UI" -ForegroundColor Cyan
Write-Host "  ─────────────────────────────────────"
Write-Host "  Backend : $MlrunsDir"
Write-Host "  UI      : http://localhost:$Port" -ForegroundColor Green
Write-Host ""

mlflow ui --backend-store-uri "file:///$($MlrunsDir.Replace('\','/'))" --port $Port --host 0.0.0.0
