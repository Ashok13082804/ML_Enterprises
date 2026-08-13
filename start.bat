@echo off
REM MLVerse X — Local Start Script (Windows)

echo Starting MLVerse X...

if not exist .env (
    echo Creating .env file from .env.example...
    copy .env.example .env
)

if not exist .venv (
    echo Creating Python virtual environment (.venv)...
    python -m venv .venv
)

call .venv\Scripts\activate.bat

echo Installing Python dependencies...
python -m pip install -r backend\requirements.txt

echo Starting FastAPI Backend on http://localhost:8000 ...
start "MLVerse Backend" .venv\Scripts\python.exe -m uvicorn app.main:app --app-dir backend --port 8000

echo Starting Next.js Frontend on http://localhost:3000 ...
cd frontend
if not exist node_modules (
    echo Installing npm dependencies...
    npm install
)
npm run dev
