@echo off
setlocal
cd /d "%~dp0"

set HOST=127.0.0.1
set PORT=8000

if not exist .venv (
  py -3 -m venv .venv
)

call .venv\Scripts\activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo Starting server on http://%HOST%:%PORT% ...
start "" "http://%HOST%:%PORT%/"

python -m uvicorn backend.app.main:app --host %HOST% --port %PORT%

echo.
echo Server stopped. Press any key to exit.
pause >nul

