@echo off
echo ============================================================
echo  Lancement Streamlit avec Intel Arc XPU (oneAPI + IPEX)
echo ============================================================
echo.

REM Ajouter les DLL oneAPI au PATH de ce process
set "ONEAPI_ROOT=C:\Program Files (x86)\Intel\oneAPI"
set "PATH=%ONEAPI_ROOT%\2025.1\bin;%ONEAPI_ROOT%\compiler\2025.1\bin;%ONEAPI_ROOT%\mkl\2025.1\bin;%ONEAPI_ROOT%\tbb\2022.1\bin;%ONEAPI_ROOT%\dnnl\2025.1\bin;%PATH%"

echo [1/2] Test Intel Arc XPU...
python "%~dp0test_xpu.py"
if errorlevel 1 (
    echo [WARN] XPU non accessible - mode CPU
) else (
    echo [OK] Intel Arc XPU pret
)

echo.
echo [2/2] Demarrage Streamlit...
cd /d "%~dp0"
streamlit run app.py
pause
