# UserTurnLoRA Docker Image
# Optimized for GPU training on cloud instances (GCloud, RunPod, etc.)

# Use --platform to ensure x86_64 build for cloud deployment
FROM --platform=linux/amd64 nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HUB_ENABLE_HF_TRANSFER=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    git \
    wget \
    unzip \
    curl \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# Set working directory
WORKDIR /app

# Copy requirements first for layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install flash-attn from pre-built wheel (CUDA 12.1, Python 3.11, PyTorch 2.5)
RUN pip install --no-cache-dir https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.3.11/flash_attn-2.8.0%2Bcu121torch2.5-cp311-cp311-linux_x86_64.whl

# Install BLEURT from git (not on PyPI)
RUN pip install --no-cache-dir git+https://github.com/google-research/bleurt.git

# Download BLEURT checkpoint
RUN wget -q https://storage.googleapis.com/bleurt-oss-21/BLEURT-20.zip \
    && unzip -q BLEURT-20.zip -d /app \
    && rm BLEURT-20.zip

# Copy source code (ablation is now in src/)
COPY src/ ./src/
COPY create_plots.py .

# Create output directory
RUN mkdir -p /app/output

# Set environment variables
ENV BLEURT_CHECKPOINT=/app/BLEURT-20
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:64

# Default command: run the pipeline
ENTRYPOINT ["python", "-m", "src.main"]

# Default arguments (can be overridden)
CMD ["--help"]
