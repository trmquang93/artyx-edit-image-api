#!/bin/bash

# Entrypoint script for AI Image Editing Server
# Supports both RunPod serverless and FastAPI modes

echo "🚀 Starting AI Image Editing Server..."
echo "Environment: ${SERVER_MODE:-runpod}"
echo "Python Version: $(python --version)"
echo "Working Directory: $(pwd)"

# Check server mode
if [ "${SERVER_MODE}" = "fastapi" ]; then
    echo "📡 Starting FastAPI server with multipart upload support..."
    echo "Available at: http://0.0.0.0:8000"
    echo "API Docs: http://0.0.0.0:8000/docs"
    exec python main.py
elif [ "${SERVER_MODE}" = "debug" ]; then
    echo "🐛 Starting in debug mode..."
    python -c "
import sys
print(f'Python path: {sys.path}')
try:
    import torch
    print(f'✅ PyTorch version: {torch.__version__}')
    print(f'✅ CUDA available: {torch.cuda.is_available()}')
except ImportError:
    print('❌ PyTorch not available')
try:
    import transformers
    print(f'✅ Transformers version: {transformers.__version__}')
except ImportError:
    print('❌ Transformers not available')
"
    echo "Debug complete. Exiting..."
else
    echo "🤖 Starting RunPod serverless worker with REAL AI processing..."
    exec python -u runpod_worker.py
fi