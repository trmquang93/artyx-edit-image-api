#!/bin/bash

# Qwen-Image AI Editing Server Startup Script

set -e

# Configuration
DEFAULT_HOST="0.0.0.0"
DEFAULT_PORT="8000"
DEFAULT_LOG_LEVEL="INFO"

# Parse command line arguments
HOST=${1:-$DEFAULT_HOST}
PORT=${2:-$DEFAULT_PORT}
LOG_LEVEL=${3:-$DEFAULT_LOG_LEVEL}

echo "🚀 Starting Qwen-Image AI Editing Server"
echo "   Host: $HOST"
echo "   Port: $PORT"
echo "   Log Level: $LOG_LEVEL"
echo "=" * 50

# Check if GPU is available
if command -v nvidia-smi &> /dev/null; then
    echo "🔍 GPU Status:"
    nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader,nounits
    echo
fi

# Check Python environment
echo "🐍 Python Environment:"
echo "   Python: $(python --version)"
echo "   PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "   CUDA Available: $(python -c 'import torch; print(torch.cuda.is_available())')"

if python -c 'import torch; torch.cuda.is_available()' | grep -q "True"; then
    echo "   GPU Count: $(python -c 'import torch; print(torch.cuda.device_count())')"
    echo "   GPU Name: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")')"
fi

echo

# Set environment variables
export LOG_LEVEL=$LOG_LEVEL
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Create necessary directories
mkdir -p logs
mkdir -p /tmp/.torch
mkdir -p /tmp/.huggingface
mkdir -p /tmp/.transformers

# Start the server
echo "🌟 Starting FastAPI server..."
echo

if [ "$RUNPOD_MODE" = "true" ]; then
    echo "🚀 Running in RunPod serverless mode"
    python runpod/handler.py
else
    echo "🌐 Running in FastAPI server mode"
    python -m uvicorn main:app \
        --host $HOST \
        --port $PORT \
        --log-level $(echo $LOG_LEVEL | tr '[:upper:]' '[:lower:]') \
        --access-log \
        --loop uvloop \
        --http httptools
fi