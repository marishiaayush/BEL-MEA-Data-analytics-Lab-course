@echo off
REM Launch JupyterLab in the course environment.
setlocal
cd /d "%~dp0"
where uv >nul 2>nul || set "PATH=%USERPROFILE%\.local\bin;%PATH%"
uv run jupyter lab notebooks
