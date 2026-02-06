@echo off
setlocal
cd /d "%~dp0\.."

if not exist .venv (
  py -3 -m venv .venv
)

call .venv\Scripts\activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install pyinstaller

pyinstaller --noconfirm --onefile --name PaperBotDashboard ^
  --add-data "backend\app\web\index.html;backend\app\web" ^
  -m uvicorn backend.app.main:app

echo.
echo Done. EXE is in dist\PaperBotDashboard.exe
pause
