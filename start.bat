@echo off
rem Starts the Factory Energy agent on this PC and opens it in the browser.
rem   1. finds Python (3.11+ preferred), makes .venv and installs packages the first time
rem   2. loads the 5-quarter history into data\energy.duckdb the first time
rem   3. starts the API (port 8000) and the UI (port 8501) in their own windows
rem   4. opens http://localhost:8501
rem Close the two windows "Energy API" and "Energy UI" to stop it.
setlocal
cd /d "%~dp0"
title Factory Energy - starter

set "API_PORT=8000"
set "UI_PORT=8501"
set "VENV_PY=.venv\Scripts\python.exe"

rem ---- 1. Python and packages ------------------------------------------------
if exist "%VENV_PY%" goto have_venv
set "BASE_PY="
for %%V in (3.13 3.12 3.11 3.10) do (
    if not defined BASE_PY (
        py -%%V -c "import sys" >nul 2>&1 && set "BASE_PY=py -%%V"
    )
)
if not defined BASE_PY (
    python -c "import sys; assert sys.version_info >= (3, 10)" >nul 2>&1 && set "BASE_PY=python"
)
if not defined BASE_PY (
    echo Python 3.10 or newer was not found. Install it from https://www.python.org and run start.bat again.
    pause
    exit /b 1
)
echo Creating the virtual environment with %BASE_PY% ...
%BASE_PY% -m venv .venv || (echo Could not create .venv & pause & exit /b 1)

:have_venv
"%VENV_PY%" -c "import fastapi, uvicorn, multipart, langgraph, duckdb, pandas, numpy, statsmodels, plotly, streamlit, httpx, requests, yaml" >nul 2>&1
if not errorlevel 1 goto deps_ok
echo Installing packages (first run only, takes a few minutes) ...
"%VENV_PY%" -m pip install --upgrade pip >nul 2>&1
"%VENV_PY%" -m pip install -r requirements.txt
if errorlevel 1 (
    echo The pinned versions in requirements.txt do not fit this Python; installing compatible versions instead ...
    "%VENV_PY%" -m pip install fastapi uvicorn pydantic python-multipart langgraph duckdb pandas numpy statsmodels plotly streamlit httpx requests PyYAML pytest
    if errorlevel 1 (echo Package installation failed. & pause & exit /b 1)
)
:deps_ok

rem ---- settings ---------------------------------------------------------------
if not exist ".env" (
    copy ".env.example" ".env" >nul
    echo Created .env from .env.example ^(MOCK_LLM=true: no LLM key needed^).
)

rem ---- 2. data and 3. API -----------------------------------------------------
call :health
if not errorlevel 1 (
    echo The API is already running on port %API_PORT%.
    goto start_ui
)
echo Loading the history data if the store is empty ...
"%VENV_PY%" scripts\load_data.py data\history --if-empty
echo Starting the API on http://localhost:%API_PORT% ...
start "Energy API" cmd /k ""%VENV_PY%" -m uvicorn app.api:app --host 127.0.0.1 --port %API_PORT%"
echo Waiting for the API to answer ...
powershell -NoProfile -Command "for($i=0;$i -lt 120;$i++){try{Invoke-WebRequest -UseBasicParsing http://127.0.0.1:%API_PORT%/health -TimeoutSec 3 | Out-Null; exit 0}catch{Start-Sleep -Seconds 1}}; exit 1"
if errorlevel 1 (
    echo The API did not start. Look at the "Energy API" window for the error.
    pause
    exit /b 1
)

rem ---- 3. UI and 4. browser ---------------------------------------------------
:start_ui
echo Starting the UI on http://localhost:%UI_PORT% ...
start "Energy UI" cmd /k "set API_URL=http://127.0.0.1:%API_PORT%&& "%VENV_PY%" -m streamlit run ui\streamlit_app.py --server.port %UI_PORT% --server.headless true --browser.gatherUsageStats false"
powershell -NoProfile -Command "for($i=0;$i -lt 60;$i++){try{Invoke-WebRequest -UseBasicParsing http://127.0.0.1:%UI_PORT% -TimeoutSec 3 | Out-Null; exit 0}catch{Start-Sleep -Seconds 1}}; exit 1"
start "" "http://localhost:%UI_PORT%"
echo.
echo   UI   http://localhost:%UI_PORT%
echo   API  http://localhost:%API_PORT%/docs
echo   Close the "Energy API" and "Energy UI" windows to stop.
echo.
timeout /t 8 >nul
exit /b 0

:health
powershell -NoProfile -Command "try{Invoke-WebRequest -UseBasicParsing http://127.0.0.1:%API_PORT%/health -TimeoutSec 2 | Out-Null; exit 0}catch{exit 1}" >nul 2>&1
exit /b %errorlevel%
