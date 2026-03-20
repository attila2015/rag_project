# Start llama.cpp OpenAI-compatible server for Qwen2.5-VL (Windows)
# Détecte automatiquement : Intel Arc (Vulkan/SYCL), NVIDIA (CUDA), CPU fallback
# Usage: pwsh scripts/start_server.ps1 [-Backend cpu|vulkan|sycl|cuda]

param(
    [string]$Backend = "auto",   # auto | cpu | vulkan | sycl | cuda
    [int]$Port       = 8080,
    [int]$NCtx       = 4096,
    [int]$NThreads   = 8,
    [int]$NGpuLayers = -1        # -1 = auto selon backend
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot

# Load .env
$envFile = Join-Path $Root ".env"
if (Test-Path $envFile) {
    Get-Content $envFile | Where-Object { $_ -notmatch "^#" -and $_ -match "=" } | ForEach-Object {
        $k, $v = $_ -split "=", 2
        [System.Environment]::SetEnvironmentVariable($k.Trim(), $v.Trim(), "Process")
    }
}

# ── Détection GPU ─────────────────────────────────────────────────────────────
function Get-GpuInfo {
    $gpus = Get-CimInstance Win32_VideoController | Select-Object Name, AdapterRAM
    return $gpus
}

$gpus = Get-GpuInfo
$hasNvidia = $gpus | Where-Object { $_.Name -match "NVIDIA|GeForce|RTX|GTX" }
$hasIntelArc = $gpus | Where-Object { $_.Name -match "Intel.*Arc|Arc.*Graphics" }
$hasAMD = $gpus | Where-Object { $_.Name -match "AMD|Radeon|RX " }

Write-Host ""
Write-Host "  GPU(s) detectes :" -ForegroundColor Cyan
$gpus | ForEach-Object {
    $vram = if ($_.AdapterRAM) { [math]::Round($_.AdapterRAM/1GB, 1) } else { "?" }
    Write-Host "    - $($_.Name) ($vram GB)" -ForegroundColor White
}

# ── Sélection automatique du backend ─────────────────────────────────────────
if ($Backend -eq "auto") {
    if ($hasNvidia) {
        $Backend = "cuda"
        Write-Host "  Backend auto: CUDA (NVIDIA detecte)" -ForegroundColor Green
    } elseif ($hasIntelArc) {
        # Vulkan fonctionne sans installation supplémentaire sur Intel Arc
        $Backend = "vulkan"
        Write-Host "  Backend auto: Vulkan (Intel Arc detecte - drivers Intel natifs)" -ForegroundColor Yellow
    } elseif ($hasAMD) {
        $Backend = "vulkan"
        Write-Host "  Backend auto: Vulkan (AMD detecte)" -ForegroundColor Yellow
    } else {
        $Backend = "cpu"
        Write-Host "  Backend auto: CPU (pas de GPU discret detecte)" -ForegroundColor DarkGray
    }
}

# ── Répertoire du binaire ──────────────────────────────────────────────────────
$BinDir = switch ($Backend) {
    "vulkan" { Join-Path $Root "bin\vulkan" }
    "sycl"   { Join-Path $Root "bin\sycl" }
    "cuda"   { Join-Path $Root "bin\cuda" }
    default  { Join-Path $Root "bin\cpu" }
}

$ServerBin = Join-Path $BinDir "llama-server.exe"
if (-not (Test-Path $ServerBin)) {
    Write-Host "  Binaire $Backend non trouve dans $BinDir" -ForegroundColor Red
    Write-Host "  Fallback sur CPU..." -ForegroundColor Yellow
    $BinDir = Join-Path $Root "bin\cpu"
    $ServerBin = Join-Path $BinDir "llama-server.exe"
    $Backend = "cpu"
}

# ── Auto-detect model et mmproj ───────────────────────────────────────────────
$ModelPath = $env:MODEL_PATH
if (-not $ModelPath) {
    $ModelPath = Get-ChildItem "$Root\models" -Filter "*.gguf" `
        | Where-Object { $_.Name -notmatch "mmproj" } `
        | Sort-Object LastWriteTime -Descending `
        | Select-Object -First 1 -ExpandProperty FullName
}
if (-not $ModelPath) {
    Write-Host "[ERREUR] Aucun modele .gguf dans $Root\models\" -ForegroundColor Red
    Write-Host "         Lancez: python scripts/download_model.py" -ForegroundColor Yellow
    exit 1
}

$MmprojPath = Get-ChildItem "$Root\models" -Filter "mmproj*.gguf" `
    | Select-Object -First 1 -ExpandProperty FullName

# ── GPU layers selon backend ───────────────────────────────────────────────────
if ($NGpuLayers -eq -1) {
    $NGpuLayers = switch ($Backend) {
        "cuda"   { 35 }
        "vulkan" { 20 }   # Intel Arc integrée ~2GB partagé — prudent
        "sycl"   { 20 }
        default  { 0  }   # CPU: 0 layers GPU
    }
}

# ── Affichage ─────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "  ─────────────────────────────────────────────" -ForegroundColor DarkGray
Write-Host "  Backend   : $Backend" -ForegroundColor Cyan
Write-Host "  Binaire   : $ServerBin"
Write-Host "  Modele    : $(Split-Path $ModelPath -Leaf)"
if ($MmprojPath) { Write-Host "  MMProj    : $(Split-Path $MmprojPath -Leaf)" -ForegroundColor Green }
Write-Host "  GPU layers: $NGpuLayers  [0=CPU only]"
Write-Host "  Contexte  : $NCtx tokens"
Write-Host "  Threads   : $NThreads"
Write-Host "  Port      : $Port"
Write-Host "  ─────────────────────────────────────────────" -ForegroundColor DarkGray
Write-Host ""
Write-Host "  Serveur : http://localhost:$Port" -ForegroundColor Green
Write-Host "  UI      : http://localhost:8503" -ForegroundColor Green
Write-Host ""

# ── Lancement ─────────────────────────────────────────────────────────────────
$args_list = @(
    "--model", $ModelPath,
    "--n-gpu-layers", $NGpuLayers,
    "--ctx-size", $NCtx,
    "--threads", $NThreads,
    "--port", $Port,
    "--host", "0.0.0.0",
    "--chat-template", "chatml"
)

if ($MmprojPath) {
    $args_list += @("--mmproj", $MmprojPath)
}

# Ajouter DLL Vulkan au PATH si nécessaire
if ($Backend -eq "vulkan") {
    $env:PATH = "$BinDir;$env:PATH"
}

& $ServerBin @args_list
