#!/usr/bin/env bash
# MLVerse X — Ollama Auto-Installer & Starter Script (macOS / Linux)
set -e

echo "🦙 Checking Ollama installation status..."

if command -v ollama &> /dev/null; then
    echo "✅ Ollama binary is already installed at: $(which ollama)"
else
    echo "📦 Installing Ollama..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        if command -v brew &> /dev/null; then
            brew install ollama
        else
            echo "Installing via Official script..."
            curl -fsSL https://ollama.com/install.sh | sh
        fi
    else
        curl -fsSL https://ollama.com/install.sh | sh
    fi
fi

echo "🚀 Starting Ollama service ('ollama serve')..."
ollama serve &
OLLAMA_PID=$!

sleep 3

echo "📥 Pulling default model 'llama3.2'..."
ollama pull llama3.2 || echo "⚠️ Model pull can be completed once service starts."

echo "🎉 Ollama setup script complete! Ollama is running at http://localhost:11434"
