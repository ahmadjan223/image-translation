# Use Python 3.12 slim image (lightweight)
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libgl1 \
    libglib2.0-0 \
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

# Pre-download PaddleOCR models during build
# This caches models in /root/.paddlex/ so they don't need to download at runtime
RUN python download_models.py

# Cloud Run requires the app to listen on 0.0.0.0 and port 8080
EXPOSE 8080

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]