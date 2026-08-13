# MLVerse X — Local Start Script (Windows PowerShell)

Write-Host "🚀 Starting MLVerse X..." -ForegroundColor Cyan

if (-not (Test-Path .env)) {
    Write-Host "📝 Creating .env file from .env.example..." -ForegroundColor Yellow
    Copy-Item .env.example .env
}

if (-not (Test-Path .venv)) {
    Write-Host "🐍 Creating Python virtual environment (.venv)..." -ForegroundColor Yellow
    python -m venv .venv
}

Write-Host "📦 Installing Python dependencies..." -ForegroundColor Cyan
& .\.venv\Scripts\python.exe -m pip install -r backend/requirements.txt

Write-Host "⚡ Starting FastAPI Backend on http://localhost:8000 ..." -ForegroundColor Green
Start-Process .\.venv\Scripts\python.exe -ArgumentList "-m uvicorn app.main:app --app-dir backend --port 8000"

Write-Host "🌐 Starting Next.js Frontend on http://localhost:3000 ..." -ForegroundColor Green
Set-Location frontend
if (-not (Test-Path node_modules)) {
    Write-Host "📦 Installing npm dependencies..." -ForegroundColor Cyan
    npm install
}
npm run dev
