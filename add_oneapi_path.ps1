$oneapiPaths = @(
    'C:\Program Files (x86)\Intel\oneAPI\2025.1\bin',
    'C:\Program Files (x86)\Intel\oneAPI\compiler\2025.1\bin',
    'C:\Program Files (x86)\Intel\oneAPI\mkl\2025.1\bin',
    'C:\Program Files (x86)\Intel\oneAPI\tbb\2022.1\bin',
    'C:\Program Files (x86)\Intel\oneAPI\dnnl\2025.1\bin',
    'C:\Program Files (x86)\Intel\oneAPI\dal\2025.5\bin',
    'C:\Program Files (x86)\Intel\oneAPI\ipp\2022.1\bin',
    'C:\Program Files (x86)\Intel\oneAPI\ippcp\2025.1\bin',
    'C:\Program Files (x86)\Intel\oneAPI\umf\0.10\bin',
    'C:\Program Files (x86)\Intel\oneAPI\tcm\1.3\bin'
)

# Filtre les chemins existants seulement
$existingPaths = $oneapiPaths | Where-Object { Test-Path $_ }

# Mise a jour PATH utilisateur (permanent)
$currentPath = [Environment]::GetEnvironmentVariable('PATH', 'User')
$toAdd = $existingPaths | Where-Object { $currentPath -notlike "*$_*" }
if ($toAdd.Count -gt 0) {
    $newPath = ($toAdd -join ';') + ';' + $currentPath
    [Environment]::SetEnvironmentVariable('PATH', $newPath, 'User')
    Write-Host "PATH utilisateur mis a jour (+$($toAdd.Count) dossiers oneAPI)"
} else {
    Write-Host "Tous les chemins oneAPI sont deja dans PATH"
}

# Aussi mettre a jour le PATH de ce process pour le test immediat
$env:PATH = ($existingPaths -join ';') + ';' + $env:PATH

# Test IPEX
Write-Host ""
Write-Host "--- Test IPEX ---"
$result = & python -c "
import os
for p in [r'C:\Program Files (x86)\Intel\oneAPI\2025.1\bin',
          r'C:\Program Files (x86)\Intel\oneAPI\compiler\2025.1\bin',
          r'C:\Program Files (x86)\Intel\oneAPI\mkl\2025.1\bin',
          r'C:\Program Files (x86)\Intel\oneAPI\tbb\2022.1\bin',
          r'C:\Program Files (x86)\Intel\oneAPI\dnnl\2025.1\bin']:
    if os.path.exists(p) and hasattr(os, 'add_dll_directory'):
        os.add_dll_directory(p)
import torch
print('torch:', torch.__version__)
import intel_extension_for_pytorch as ipex
print('IPEX:', ipex.__version__)
print('XPU:', torch.xpu.is_available())
" 2>&1
Write-Host $result
