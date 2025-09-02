#!/bin/bash

# Enhanced entrypoint script for AI Image Editing Server
# Based on proven production patterns for reliability and performance

# Exit immediately if a command exits with a non-zero status
set -e

echo "🚀 Starting AI Image Editing Server (Enhanced)"
echo "=================================================="

# CUDA 검사 및 설정 (Enhanced pattern)
echo "🔍 Checking CUDA availability..."

# Python을 통한 CUDA 검사 (Enhanced pattern)
python_cuda_check() {
    python3 -c "
import torch
try:
    if torch.cuda.is_available():
        print('CUDA_AVAILABLE')
        exit(0)
    else:
        print('CUDA_NOT_AVAILABLE')
        exit(1)
except Exception as e:
    print(f'CUDA_ERROR: {e}')
    exit(2)
" 2>/dev/null
}

# CUDA 검사 실행 (Enhanced pattern)
cuda_status=$(python_cuda_check)
case $? in
    0)
        echo "✅ CUDA is available and working (Python check)"
        export CUDA_VISIBLE_DEVICES=0
        export FORCE_CUDA=1
        ;;
    1)
        echo "❌ CUDA is not available (Python check)"
        echo "Error: CUDA is required but not available. Exiting..."
        exit 1
        ;;
    2)
        echo "❌ CUDA check failed (Python check)"
        echo "Error: CUDA initialization failed. Exiting..."
        exit 1
        ;;
esac

# 추가적인 nvidia-smi 검사 (Enhanced pattern)
if command -v nvidia-smi &> /dev/null; then
    if nvidia-smi &> /dev/null; then
        echo "✅ NVIDIA driver working (nvidia-smi check)"
        echo "📊 GPU Information:"
        nvidia-smi --query-gpu=name,memory.total,memory.used,temperature.gpu --format=csv,noheader,nounits 2>/dev/null || echo "Could not query GPU details"
    else
        echo "❌ NVIDIA driver found but not working"
        echo "Error: NVIDIA driver is not working properly. Exiting..."
        exit 1
    fi
else
    echo "❌ NVIDIA driver not found"
    echo "Error: NVIDIA driver is required but not found. Exiting..."
    exit 1
fi

# CUDA 환경 변수 설정 확인
echo "🔧 CUDA Environment:"
echo "   CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "   FORCE_CUDA: $FORCE_CUDA"
echo "   CUDA_HOME: $CUDA_HOME"

# Python dependencies check
echo "🐍 Checking Python dependencies..."
python_deps_check() {
    python3 -c "
import sys
missing_deps = []
required_deps = ['torch', 'diffusers', 'transformers', 'PIL', 'runpod']

for dep in required_deps:
    try:
        __import__(dep)
        print(f'✅ {dep} imported successfully')
    except ImportError as e:
        print(f'❌ {dep} import failed: {e}')
        missing_deps.append(dep)

if missing_deps:
    print(f'Missing dependencies: {missing_deps}')
    sys.exit(1)
else:
    print('✅ All required dependencies available')
    sys.exit(0)
" 2>/dev/null
}

if ! python_deps_check; then
    echo "❌ Python dependencies check failed"
    echo "Error: Required dependencies are missing. Exiting..."
    exit 1
fi

# PyTorch CUDA compatibility check
echo "🧪 Testing PyTorch CUDA integration..."
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    print(f'Current device: {torch.cuda.current_device()}')
    print(f'Device name: {torch.cuda.get_device_name(0)}')
    # Test tensor creation on GPU
    try:
        x = torch.randn(2, 2).cuda()
        print('✅ GPU tensor creation successful')
    except Exception as e:
        print(f'❌ GPU tensor creation failed: {e}')
        exit(1)
" || {
    echo "❌ PyTorch CUDA integration test failed"
    exit 1
}

# Model cache directory setup
echo "📁 Setting up model cache directories..."
mkdir -p /runpod-volume/.torch/hub
mkdir -p /runpod-volume/.huggingface/transformers
mkdir -p /runpod-volume/.transformers
echo "✅ Cache directories ready"

# Handler validation
echo "🔍 Validating handler module..."
python3 -c "
try:
    import handler_enhanced
    print('✅ Handler module imported successfully')
    
    # Test basic initialization
    manager = handler_enhanced.QwenImageManager()
    print('✅ QwenImageManager created successfully')
    
except Exception as e:
    print(f'❌ Handler validation failed: {e}')
    import traceback
    traceback.print_exc()
    exit(1)
" || {
    echo "❌ Handler module validation failed"
    exit 1
}

# Health check before starting main service
echo "🏥 Running health check..."
python3 -c "
import handler_enhanced
import json

try:
    # Test health endpoint
    test_job = {'input': {'task': 'health'}}
    result = handler_enhanced.handler(test_job)
    
    if result.get('success'):
        print('✅ Health check passed')
        print(f'Server type: {result.get(\"environment\", {}).get(\"server_type\", \"unknown\")}')
        print(f'GPU available: {result.get(\"environment\", {}).get(\"gpu_available\", False)}')
    else:
        print(f'❌ Health check failed: {result}')
        exit(1)
        
except Exception as e:
    print(f'❌ Health check exception: {e}')
    import traceback
    traceback.print_exc()
    exit(1)
" || {
    echo "❌ Health check failed"
    exit 1
}

# Memory usage check
echo "💾 Checking memory usage..."
python3 -c "
import torch
if torch.cuda.is_available():
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    print(f'GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB, Total: {total:.2f}GB')
    
    if total < 8.0:  # Less than 8GB
        print('⚠️  Warning: GPU has less than 8GB memory, some models may not load')
    else:
        print('✅ GPU memory sufficient for AI models')
else:
    print('⚠️  No GPU detected, running on CPU (will be slower)')
"

# Final startup message
echo "🎯 All checks passed! Starting RunPod serverless handler..."
echo "=================================================="

# Start the handler in the foreground
# This script becomes the main process for the container
exec python /app/handler_enhanced.py