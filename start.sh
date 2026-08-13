#!/usr/bin/env bash
# MLVerse X — Local Start Script (macOS / Linux)
set -e

echo "🚀 Starting MLVerse X..."

# Free ports 8000 & 3000 if occupied
lsof -ti:8000 | xargs kill -9 2>/dev/null || true
lsof -ti:3000 | xargs kill -9 2>/dev/null || true

if [ ! -f .env ]; then
  echo "📝 Creating .env file from .env.example..."
  cp .env.example .env
fi

if [ ! -d .venv ]; then
  echo "🐍 Creating Python virtual environment (.venv)..."
  python3 -m venv .venv
fi

source .venv/bin/activate
echo "📦 Installing Python dependencies..."
pip install -q -r backend/requirements.txt

echo "⚡ Starting FastAPI Backend on http://localhost:8000 ..."
.venv/bin/python3 -m uvicorn app.main:app --app-dir backend --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

trap "kill $BACKEND_PID 2>/dev/null" EXIT INT TERM

echo "🌐 Starting Next.js Frontend on http://localhost:3000 ..."
cd frontend
if [ ! -d node_modules ]; then
  echo "📦 Installing npm dependencies..."
  npm install
fi
npm run dev
