# Use Python 3.12 slim image (lightweight)
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies (incl. Pillow build deps and fonts)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    build-essential \
    python3-dev \
    libgl1 \
    libglib2.0-0 \
    libjpeg-dev \
    zlib1g-dev \
    libpng-dev \
    libtiff-dev \
    libopenjp2-7-dev \
    libwebp-dev \
    libfreetype6-dev \
    fonts-liberation \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install PyTorch CPU-only first
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies (ignore dependency conflicts since it works locally)
RUN pip install -r requirements.txt

# Copy application code
COPY . .

# Set environment variables to disable oneDNN/MKL-DNN optimizations
ENV FLAGS_use_mkldnn=false
ENV PADDLE_USE_MKLDNN=0
ENV FLAGS_use_onednn=false

# Disable NumPy/OpenBLAS threading and SIMD to prevent SIGFPE on Cloud Run
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV NUMEXPR_NUM_THREADS=1
ENV KMP_DUPLICATE_LIB_OK=TRUE
ENV OPENBLAS_CORETYPE=generic

# Pre-download PaddleOCR models during build
# This caches models in /root/.paddlex/ so they don't need to download at runtime
RUN python download_models.py

# Cloud Run requires the app to listen on 0.0.0.0 and port 8080
EXPOSE 8080

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]