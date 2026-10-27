@echo off
REM ============================================================
REM  One-time setup for the HD-MEA practical (Windows).
REM  Double-click this file. It installs uv (if missing) and
REM  builds the Python environment from the lockfile.
REM ============================================================
setlocal
cd /d "%~dp0"

echo.
echo [1/2] Checking for uv ...
where uv >nul 2>nul
if errorlevel 1 (
    echo       uv not found - installing it ...
    powershell -ExecutionPolicy ByPass -NoProfile -Command "irm https://astral.sh/uv/install.ps1 | iex"
    REM add uv to PATH for the rest of this script
    set "PATH=%USERPROFILE%\.local\bin;%PATH%"
)

echo.
echo [2/2] Building the environment (downloads Python + libraries once) ...
uv sync
if errorlevel 1 (
    echo.
    echo  *** Setup FAILED. Check your internet connection and try again. ***
    pause
    exit /b 1
)

echo.
echo  Done!  Double-click  start_jupyter.bat  to open the notebooks.
echo.
pause
