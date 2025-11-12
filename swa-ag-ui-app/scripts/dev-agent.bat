@echo off
REM Navigate to the agent directory
cd /d "%~dp0\..\agent" || exit /b 1

REM Check if .venv exists, if not use the global Python
if exist ".venv\Scripts\activate.bat" (
    REM Activate the virtual environment
    call .venv\Scripts\activate.bat
) else (
    echo Warning: .venv not found in agent directory
)

REM Start langgraph dev server
langgraph dev --port 8123 --no-browser
