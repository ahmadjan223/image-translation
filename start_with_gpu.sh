#!/bin/bash
# Startup script for Image Translation API with GPU support
# Uses tmux for persistent session that survives SSH disconnections

# Set CUDA library paths from venv
export LD_LIBRARY_PATH="/root/image-translation/venv/lib/python3.11/site-packages/nvidia/cublas/lib:/root/image-translation/venv/lib/python3.11/site-packages/nvidia/cudnn/lib:/root/image-translation/venv/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH"

echo "✅ LD_LIBRARY_PATH set"
echo "🚀 Starting FastAPI with GPU support in tmux session..."

# Kill existing session if it exists
tmux kill-session -t translation 2>/dev/null

# Create tmux session and run server
tmux new-session -d -s translation -c /root/image-translation \
  "source venv/bin/activate && export LD_LIBRARY_PATH=/root/image-translation/venv/lib/python3.11/site-packages/nvidia/cublas/lib:/root/image-translation/venv/lib/python3.11/site-packages/nvidia/cudnn/lib:/root/image-translation/venv/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:\$LD_LIBRARY_PATH && python main.py"

echo "✅ Server started in tmux session 'translation'"
echo "📌 To attach: tmux attach -t translation"
echo "📌 To detach: Press Ctrl+B then D"
echo "📌 To stop: tmux kill-session -t translation"
