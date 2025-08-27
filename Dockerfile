# Optimized Dockerfile for Qwen-Image AI Editing Server
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Set working directory
WORKDIR /app

# Set environment variables for non-interactive installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV TORCH_HOME=/tmp/.torch
ENV HF_HOME=/tmp/.huggingface
ENV TRANSFORMERS_CACHE=/tmp/.transformers
ENV PIP_NO_CACHE_DIR=1

# Pre-configure timezone to prevent interactive prompts
RUN echo 'tzdata tzdata/Areas select Etc' | debconf-set-selections && \
    echo 'tzdata tzdata/Zones/Etc select UTC' | debconf-set-selections

# Install system dependencies with non-interactive mode and clean up in one layer
RUN apt-get update && apt-get install -y \
    tzdata \
    git \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && apt-get autoremove -y

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies in a single layer to reduce disk usage
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir \
        accelerate \
        safetensors \
        invisible-watermark && \
    pip cache purge && \
    find /opt/conda -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# Copy application code
COPY . .

# Create cache directories with proper permissions
RUN mkdir -p /tmp/.torch /tmp/.huggingface /tmp/.transformers && \
    chmod -R 777 /tmp/.torch /tmp/.huggingface /tmp/.transformers

# Pre-download models (optional, for faster cold starts)
# RUN python -c "from diffusers import DiffusionPipeline; DiffusionPipeline.from_pretrained('Qwen/Qwen-Image', torch_dtype=torch.float16)"

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Default command - RunPod worker with enhanced error handling
CMD ["python", "runpod_worker.py"]

# Alternative commands:
# CMD ["python", "startup_debug.py"]     # Debug startup
# CMD ["python", "main.py"]              # FastAPI server