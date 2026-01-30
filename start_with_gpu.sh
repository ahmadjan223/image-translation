#!/bin/bash
# Startup script for Image Translation API with GPU support
# This sets LD_LIBRARY_PATH before starting Python

# Set CUDA library paths from venv
export LD_LIBRARY_PATH="/root/image-translation/venv/lib/python3.11/site-packages/nvidia/cublas/lib:/root/image-translation/venv/lib/python3.11/site-packages/nvidia/cudnn/lib:/root/image-translation/venv/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH"

echo "✅ LD_LIBRARY_PATH set to: $LD_LIBRARY_PATH"
echo "🚀 Starting FastAPI with GPU support..."

# Activate venv and start the server
source venv/bin/activate
python main.py
