# Running Multiple API Instances for Load Balancing

## Overview
The translation pipeline now supports running multiple FastAPI instances to distribute concurrent requests and maximize GPU utilization.

## Quick Start

### 1. Start Multiple API Instances
```bash
# Start 4 instances (default) on ports 8080-8083
./start_multiple_instances.sh

# Or specify a custom number of instances
./start_multiple_instances.sh 6  # Starts 6 instances on ports 8080-8085
```

### 2. Run Translation Script
```bash
# The script automatically load-balances across all running instances
python translate_offer.py
```

## How It Works

### Load Balancing Strategy
- **Round-robin distribution**: Requests are evenly distributed across all API instances
- Each instance handles its own GPU operations with internal semaphores
- Total throughput scales linearly with number of instances (up to GPU limits)

### Architecture
```
translate_offer.py (40 concurrent requests)
       ↓
   Round-robin
       ↓
├─ Instance 1 (port 8080) ─ 8 OCR + 5 inpainting concurrent ops
├─ Instance 2 (port 8081) ─ 8 OCR + 5 inpainting concurrent ops  
├─ Instance 3 (port 8082) ─ 8 OCR + 5 inpainting concurrent ops
└─ Instance 4 (port 8083) ─ 8 OCR + 5 inpainting concurrent ops
```

### Configuration

**In translate_offer.py:**
```python
NUM_API_INSTANCES = 4  # Number of API instances to use
BASE_PORT = 8080       # Starting port number
MAX_CONCURRENT_REQUESTS = 40  # Total concurrent requests across all instances
```

**In start_multiple_instances.sh:**
```bash
./start_multiple_instances.sh 4  # Number of instances
```

## Monitoring

### View Logs
Each instance has its own log file:
```bash
tail -f logs/instance_8080.log
tail -f logs/instance_8081.log
# etc.
```

### Check Running Instances
```bash
lsof -i :8080-8083  # Check which ports are in use
```

## Performance Tuning

### GPU Memory Optimization
- **2x RTX 5060 Ti**: Start with 4 instances
- Monitor GPU memory: `nvidia-smi -l 1`
- Reduce instances if you see OOM errors

### Request Concurrency
Adjust `MAX_CONCURRENT_REQUESTS` in [translate_offer.py](translate_offer.py):
- **4 instances**: 40-60 concurrent requests recommended
- **6 instances**: 60-90 concurrent requests recommended
- Formula: `~10-15 requests per instance`

### Internal Semaphores (per instance)
Configured in [routes/api.py](routes/api.py):
- `ocr_semaphore`: 8 concurrent OCR operations
- `inpainting_semaphore`: 5 concurrent inpainting operations

## Stopping Instances

Press `Ctrl+C` in the terminal running `start_multiple_instances.sh` to gracefully stop all instances.

Or kill manually:
```bash
pkill -f "python main.py"
```

## Troubleshooting

### Connection Errors
If you see "Connection error" in logs:
1. Check if instances are running: `lsof -i :8080-8083`
2. Check instance logs: `cat logs/instance_8080.log`
3. Ensure correct number of instances matches `NUM_API_INSTANCES`

### GPU OOM Errors
If GPU memory errors occur:
1. Reduce number of instances: `./start_multiple_instances.sh 2`
2. Or reduce `MAX_CONCURRENT_REQUESTS` in translate_offer.py
3. Check GPU usage: `nvidia-smi`

### Uneven Load Distribution
The round-robin strategy ensures even distribution. Check logs to verify:
```bash
grep "port 80" logs/instance_*.log | cut -d: -f1 | sort | uniq -c
```
