# ============================================================
#  POC Qwen VL — Setup automatisé (Windows PowerShell)
#  Usage: pwsh scripts/setup.ps1
#  Options:
#    -SkipModel    ne pas télécharger le modèle GGUF
#    -Quant Q5_K_M choisir la quantization (Q4_K_M par défaut)
#    -ModelSize 7b  taille du modèle (7b par défaut)
#    -GPU          installer llama-cpp-python avec support CUDA
# ============================================================
param(
    [switch]$SkipModel,
    [string]$Quant     = "Q4_K_M",
    [string]$ModelSize = "7b",
    [switch]$GPU
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

function Write-Step($n, $msg) {
    Write-Host ""
    Write-Host "  [$n] $msg" -ForegroundColor Cyan
    Write-Host "  $('─' * 50)" -ForegroundColor DarkGray
}

function Write-OK($msg)  { Write-Host "  ✓ $msg" -ForegroundColor Green }
function Write-Warn($msg){ Write-Host "  ⚠ $msg" -ForegroundColor Yellow }
function Write-Err($msg) { Write-Host "  ✗ $msg" -ForegroundColor Red }

Write-Host ""
Write-Host "  ╔══════════════════════════════════════════════╗" -ForegroundColor Magenta
Write-Host "  ║   Document Intelligence — Qwen2.5-VL POC    ║" -ForegroundColor Magenta
Write-Host "  ║   Setup automatisé                          ║" -ForegroundColor Magenta
Write-Host "  ╚══════════════════════════════════════════════╝" -ForegroundColor Magenta

# ── Step 1: Python version check ─────────────────────────────
Write-Step "1/6" "Vérification Python"
$pyver = python --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Err "Python introuvable. Installez Python 3.10+ depuis python.org"
    exit 1
}
Write-OK $pyver

# ── Step 2: Virtual environment ───────────────────────────────
Write-Step "2/6" "Environnement virtuel"
if (-not (Test-Path ".venv")) {
    python -m venv .venv
    Write-OK "Créé : .venv/"
} else {
    Write-OK "Déjà existant : .venv/"
}

$activate = ".venv\Scripts\Activate.ps1"
if (Test-Path $activate) {
    & $activate
    Write-OK "Activé"
} else {
    Write-Warn "Impossible d'activer le venv automatiquement — activez manuellement"
}

# ── Step 3: pip upgrade + core deps ──────────────────────────
Write-Step "3/6" "Installation des dépendances"
python -m pip install --upgrade pip --quiet

if ($GPU) {
    Write-Host "  Mode GPU (CUDA) — compilation llama-cpp-python avec CUDA..." -ForegroundColor Yellow
    $env:CMAKE_ARGS = "-DGGML_CUDA=on"
    pip install llama-cpp-python --force-reinstall --no-cache-dir --quiet
} else {
    Write-Host "  Mode CPU — pour GPU, relancez avec -GPU" -ForegroundColor DarkGray
    pip install llama-cpp-python --quiet
}

# Install remaining deps (excluding heavy ML ones)
$reqs = Get-Content requirements.txt | Where-Object {
    $_ -notmatch "^#" -and
    $_ -notmatch "torch" -and
    $_ -notmatch "transformers" -and
    $_ -notmatch "peft" -and
    $_ -notmatch "trl" -and
    $_ -notmatch "datasets" -and
    $_ -notmatch "accelerate" -and
    $_ -notmatch "bitsandbytes" -and
    $_ -notmatch "llama-cpp-python" -and  # déjà installé
    $_.Trim() -ne ""
}
$tmpReq = [System.IO.Path]::GetTempFileName() + ".txt"
$reqs | Set-Content $tmpReq
pip install -r $tmpReq --quiet
Remove-Item $tmpReq
Write-OK "Dépendances installées"

# ── Step 4: .env ──────────────────────────────────────────────
Write-Step "4/6" "Configuration .env"
if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-OK "Créé .env depuis .env.example"
} else {
    Write-OK ".env déjà présent"
}

# ── Step 5: Download model ────────────────────────────────────
Write-Step "5/6" "Modèle GGUF"
if ($SkipModel) {
    Write-Warn "Skipped (--SkipModel). Placez manuellement un .gguf dans models/"
} else {
    $existing = Get-ChildItem "models" -Filter "*.gguf" -ErrorAction SilentlyContinue
    if ($existing) {
        Write-OK "Modèle déjà présent : $($existing[0].Name)"
    } else {
        Write-Host "  Téléchargement Qwen2.5-VL $ModelSize $Quant (~5 GB)..." -ForegroundColor Yellow
        python scripts/download_model.py --model $ModelSize --quant $Quant
        Write-OK "Modèle téléchargé"
    }
}

# ── Step 6: Summary ───────────────────────────────────────────
Write-Step "6/6" "Résumé"

Write-Host ""
Write-Host "  ┌─────────────────────────────────────────────────┐" -ForegroundColor Green
Write-Host "  │  Setup terminé !                                │" -ForegroundColor Green
Write-Host "  │                                                 │" -ForegroundColor Green
Write-Host "  │  Pour démarrer :                                │" -ForegroundColor Green
Write-Host "  │                                                 │" -ForegroundColor Green
Write-Host "  │  Terminal 1 — Serveur modèle :                  │" -ForegroundColor Green
Write-Host "  │    pwsh scripts/start_server.ps1                │" -ForegroundColor Green
Write-Host "  │                                                 │" -ForegroundColor Green
Write-Host "  │  Terminal 2 — Interface UI :                    │" -ForegroundColor Green
Write-Host "  │    pwsh scripts/start_ui.ps1                    │" -ForegroundColor Green
Write-Host "  │                                                 │" -ForegroundColor Green
Write-Host "  │  Puis ouvrir : http://localhost:8501            │" -ForegroundColor Green
Write-Host "  └─────────────────────────────────────────────────┘" -ForegroundColor Green
Write-Host ""
