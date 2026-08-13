# MLVerse X 🚀
## The Ultimate Offline AI, ML, DL, NLP & Computer Vision Platform

> **Enterprise-grade AI/ML platform with 100+ modules, Local LLM, RAG, Multi-Agent AI — runs 100% offline, no external API keys required.**

---

## ⚡ Quick Start (Without Docker — Single Line Command)

Run both the FastAPI Backend (port 8000) and Next.js Frontend (port 3000) with a **single command**:

### 🍏 macOS / Linux (Without Docker):
```bash
cp -n .env.example .env 2>/dev/null; pip install -r backend/requirements.txt && (python3 -m uvicorn app.main:app --app-dir backend --host 0.0.0.0 --port 8000 &) && (cd frontend && npm install && npm run dev)
```

### 🪟 Windows PowerShell (Without Docker):
```powershell
if (-not (Test-Path .env)) { Copy-Item .env.example .env }; pip install -r backend/requirements.txt; Start-Process python -ArgumentList "-m uvicorn app.main:app --app-dir backend --port 8000"; cd frontend; npm install; npm run dev
```

---

## 🐳 Quick Start (With Docker Compose)

### macOS / Linux (Docker):
```bash
git clone https://github.com/mlversex/platform.git mlverse-x && cd mlverse-x && cp .env.example .env && docker compose up --build
```

### Windows PowerShell (Docker):
```powershell
git clone https://github.com/mlversex/platform.git mlverse-x; cd mlverse-x; Copy-Item .env.example .env; docker compose up --build
```

---


## 🏗️ Architecture

```
MLVerse X
├── frontend/          # Next.js 14 + TypeScript + Tailwind + Shadcn
├── backend/           # FastAPI + SQLAlchemy + Celery + Redis
├── ml/                # Shared ML pipeline engine (sklearn/XGB/LGB/TF/PyTorch)
├── ai/
│   ├── agents/        # 15 autonomous AI agents
│   ├── rag/           # RAG pipeline (ChromaDB/FAISS)
│   ├── ollama/        # Local LLM client
│   └── embeddings/    # sentence-transformers (local)
├── docker/            # Dockerfiles + nginx config
├── scripts/           # Setup, migration, seed scripts
├── docs/              # API docs, architecture diagrams
└── tests/             # Unit + integration tests
```

---

## 🤖 Ollama Setup (Required for AI Assistant & RAG)

### 1. Install Ollama
```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.com/install.sh | sh

# Windows — Download from https://ollama.com/download
```

### 2. Start Ollama Service
```bash
ollama serve
```

### 3. Pull Recommended Models
```bash
# Main chat model (4B — fast, good quality)
ollama pull llama3.2

# Coding model
ollama pull codellama

# Lightweight model (for low-resource systems)
ollama pull tinyllama

# Embedding model (for RAG)
ollama pull nomic-embed-text

# Optional: More powerful models
ollama pull mistral
ollama pull deepseek-r1
ollama pull phi4
ollama pull qwen2.5
ollama pull gemma3
```

### 4. Verify
```bash
ollama list
# Should show your downloaded models

curl http://localhost:11434/api/tags
# Should return JSON with model list
```

---

## 🐳 Docker Deployment (Recommended)

### Prerequisites
- Docker Desktop 4.x+ (with Docker Compose v2)
- 8GB+ RAM recommended
- 20GB+ free disk space
- Ollama installed and running (for AI features)

### Start All Services
```bash
# Copy environment template
cp .env.example .env

# Edit .env if needed (defaults work out of the box)
nano .env  # or code .env

# Build and start all services
docker compose up --build

# Or run in background
docker compose up --build -d

# View logs
docker compose logs -f
```

### Service URLs
| Service | URL |
|---------|-----|
| **Frontend** | http://localhost:3000 |
| **Backend API** | http://localhost:8000 |
| **API Docs** | http://localhost:8000/docs |
| **MLflow** | http://localhost:5000 |
| **MinIO Console** | http://localhost:9001 |
| **Flower (Celery)** | http://localhost:5555 |
| **ChromaDB** | http://localhost:8001 |

### Stop Services
```bash
docker compose down

# Remove all data (fresh start)
docker compose down -v
```

---

## 💻 Local Development

### Backend
```bash
cd backend

# Create virtual environment
python -m venv .venv

# Activate (macOS/Linux)
source .venv/bin/activate

# Activate (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env

# Run database migrations
alembic upgrade head

# Seed initial data
python scripts/seed.py

# Start backend
uvicorn app.main:app --reload --port 8000
```

### Frontend
```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build
npm start
```

### Celery Worker (for async training jobs)
```bash
cd backend
celery -A app.workers.celery_app worker --loglevel=info
```

### Celery Beat (for scheduled tasks)
```bash
cd backend
celery -A app.workers.celery_app beat --loglevel=info
```

---

## 🔐 Default Credentials

After first startup and DB seed:

| Role | Email | Password |
|------|-------|----------|
| Super Admin | admin@mlverse.ai | Admin@123! |
| Developer | dev@mlverse.ai | Dev@123! |
| Student | student@mlverse.ai | Student@123! |

> **Change these immediately in production!**

---

## 📦 100 AI/ML Modules

### 🟢 Beginner ML (20)
House Price Prediction, Student Performance, Salary Prediction, Employee Attrition, Loan Approval, Credit Risk, Customer Churn, Insurance Premium, Car Price, Used Bike Price, Medical Insurance, Flight Fare, Weather, Rainfall, Electricity Consumption, Energy Demand, Movie Recommendation, Book Recommendation, Music Recommendation, Product Recommendation

### 🔵 NLP (20)
Fake News Detection, Spam Email, SMS Spam, Sentiment Analysis, Emotion Detection, Language Detection, News Classification, Document Categorization, Resume Screening, AI Interview Assistant, Intent Detection, Hate Speech Detection, Toxic Comment Detection, Grammar Correction, Essay Scoring, Keyword Extraction, Text Summarization, Machine Translation, Speech Emotion Recognition, Voice Command Recognition

### 🟣 Computer Vision (20)
Face Attendance, Face Mask Detection, Driver Drowsiness, Helmet Detection, Vehicle Detection, License Plate Recognition, Traffic Sign Detection, YOLO Object Detection, Human Pose Estimation, Crowd Counting, Fire Detection, Smoke Detection, PPE Detection, Animal Detection, Crop Disease Detection, Plant Species Identification, Skin Disease Detection, Brain Tumor Detection, Pneumonia Detection, Diabetic Retinopathy Detection

### 🟡 Time Series (10)
Stock Prediction, Cryptocurrency Forecasting, Sales Forecasting, Demand Forecasting, Weather Forecasting, Air Pollution Prediction, Traffic Prediction, Water Quality Prediction, Solar Energy Prediction, Wind Energy Prediction

### 🟠 Finance AI (10)
Credit Card Fraud Detection, Transaction Fraud, Loan Default Prediction, Portfolio Risk Analysis, Customer Lifetime Value, Dynamic Pricing, Invoice Classification, Expense Categorization, Stock Trend Prediction, Financial Sentiment Analysis

### 🔴 Healthcare AI (10)
Disease Prediction, Diabetes Prediction, Heart Disease Prediction, Kidney Disease Prediction, Cancer Prediction, Medical Report Summarization, Drug Recommendation, Hospital Readmission Prediction, ICU Mortality Prediction, Patient Risk Stratification

### ⚙️ Industrial AI (10)
Predictive Maintenance, Machine Failure Prediction, Quality Inspection, Defect Detection, Supply Chain Optimization, Inventory Prediction, Warehouse Analytics, Smart Manufacturing, Production Yield Prediction, Equipment Health Monitoring

---

## 🧠 AI Features

- **AI Copilot** — Offline LLM via Ollama (Llama 3, Mistral, DeepSeek, etc.)
- **RAG System** — Upload docs, PDFs, code and ask questions with citations
- **Multi-Agent AI** — 15 specialized agents working autonomously
- **AutoML** — Auto algorithm selection, hyperparameter tuning, leaderboard
- **Explainable AI** — SHAP, LIME, feature importance for every prediction

---

## 🎨 Themes
- 🌙 **Dark Mode** — Deep navy gradients
- ☀️ **Light Mode** — Clean professional white
- ⚫ **OLED Mode** — True black for OLED displays

---

## 🔒 Security
- JWT + Refresh Tokens
- Role-Based Access Control (Admin/Developer/Researcher/Student/Organization)
- Two-Factor Authentication (TOTP)
- Email Verification + OTP
- Rate Limiting
- CORS Protection
- SQL Injection Protection

---

## 🧪 Testing
```bash
# Backend tests
cd backend && pytest tests/ -v --cov=app

# Frontend tests
cd frontend && npm test

# E2E tests
npm run test:e2e
```

---

## 📄 License
MIT License — See [LICENSE](LICENSE)

---

## 🤝 Contributing
See [CONTRIBUTING.md](docs/CONTRIBUTING.md)

---

Built with ❤️ by the MLVerse X Team




