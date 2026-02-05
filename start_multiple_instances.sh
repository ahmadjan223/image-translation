#!/bin/bash
# Start multiple FastAPI instances for load balancing
# Each instance runs on a different port to handle concurrent requests

# Number of instances (adjust based on your GPU memory)
NUM_INSTANCES=${1:-4}  # Default 4 instances, override with: ./start_multiple_instances.sh 6

# Base port
BASE_PORT=8080

# Set CUDA library paths from venv
export LD_LIBRARY_PATH="/root/image-translation/venv/lib/python3.11/site-packages/nvidia/cublas/lib:/root/image-translation/venv/lib/python3.11/site-packages/nvidia/cudnn/lib:/root/image-translation/venv/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH"

echo "✅ LD_LIBRARY_PATH set"
echo "🚀 Starting $NUM_INSTANCES FastAPI instances..."

# Activate venv
source venv/bin/activate

# Create logs directory if it doesn't exist
mkdir -p logs

# Array to store PIDs
PIDS=()

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Stopping all instances..."
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null
    done
    echo "✅ All instances stopped"
    exit 0
}

# Trap Ctrl+C and other termination signals
trap cleanup SIGINT SIGTERM

# Start instances
for i in $(seq 0 $((NUM_INSTANCES - 1))); do
    PORT=$((BASE_PORT + i))
    LOG_FILE="logs/instance_${PORT}.log"
    
    echo "  ➜ Starting instance $((i + 1))/$NUM_INSTANCES on port $PORT"
    PORT=$PORT python main.py > "$LOG_FILE" 2>&1 &
    PID=$!
    PIDS+=($PID)
    
    # Small delay to avoid startup conflicts
    sleep 2
done

echo ""
echo "✅ All $NUM_INSTANCES instances started!"
echo "📋 Ports: $BASE_PORT-$((BASE_PORT + NUM_INSTANCES - 1))"
echo "📁 Logs: logs/instance_*.log"
echo ""
echo "Press Ctrl+C to stop all instances"
echo ""

# Wait for all background processes
wait
