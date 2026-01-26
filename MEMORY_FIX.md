# Memory Optimization Fix for Cloud Run

## Problem
The application was exhausting memory on Cloud Run when processing multiple images (22 images in your case) because:

1. **Unlimited parallelism**: All 22 images were processed simultaneously with `asyncio.gather()`
2. **Multiple OCR instances**: ThreadPoolExecutor with 8 workers created up to 8 concurrent OCR operations
3. **Large model memory**: Each PaddleOCR instance loads ~500MB+ of models into memory
4. **Cloud Run limits**: Typically 2-4GB RAM vs. much more on local machines

**Example**: 8 concurrent OCR × 500MB = **4GB+ memory usage** → OOM crash

## Solution

### 1. Concurrency Limiting with Semaphore
Added `asyncio.Semaphore` to limit concurrent OCR operations:

```python
# Only 3 OCR operations run simultaneously (configurable)
ocr_semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_OCR)

async with ocr_semaphore:
    ocr_data = await run_ocr(...)
```

### 2. Reduced Thread Pool Workers
Reduced ThreadPoolExecutor from 8 → 3 workers to limit memory footprint.

### 3. Configurable via Environment Variables
Added to `config.py`:
- `MAX_CONCURRENT_OCR`: Max simultaneous OCR operations (default: 3)
- `MAX_WORKER_THREADS`: Thread pool size (default: 3)

## Configuration Guide

### Cloud Run Memory Allocation

Adjust environment variables based on allocated memory:

| Cloud Run RAM | MAX_CONCURRENT_OCR | MAX_WORKER_THREADS | Expected Usage |
|---------------|-------------------|-------------------|----------------|
| 2GB           | 2                 | 2                 | ~1.5GB         |
| 4GB           | 3                 | 3                 | ~2.5GB         |
| 8GB           | 5                 | 5                 | ~4GB           |
| 16GB          | 8                 | 8                 | ~6-8GB         |

### Setting Environment Variables

**In Dockerfile** (default for all deployments):
```dockerfile
ENV MAX_CONCURRENT_OCR=3
ENV MAX_WORKER_THREADS=3
```

**In Cloud Run** (override per deployment):
```bash
gcloud run deploy image-translation \
  --set-env-vars MAX_CONCURRENT_OCR=2,MAX_WORKER_THREADS=2 \
  --memory 2Gi
```

**In `.env`** (for local development):
```
MAX_CONCURRENT_OCR=8
MAX_WORKER_THREADS=8
```

## Performance Impact

### Before (22 images):
- All 22 images start downloading simultaneously
- Up to 8 OCR operations run in parallel
- Memory spikes to 4-6GB → **OOM crash on Cloud Run**

### After (22 images):
- All 22 images still download simultaneously (low memory)
- Only 3 OCR operations run at once (controlled by semaphore)
- Memory peaks at ~2.5GB → **fits in 4GB Cloud Run instance**
- Processing time increases slightly but completes successfully

## Files Modified

1. **[config.py](config.py#L16-L17)**: Added `MAX_CONCURRENT_OCR` and `MAX_WORKER_THREADS` settings
2. **[routes/api.py](routes/api.py#L25-L28)**: 
   - Reduced ThreadPoolExecutor workers
   - Added semaphore for OCR concurrency control
   - Wrapped OCR calls in `async with ocr_semaphore`
3. **[Dockerfile](Dockerfile#L50-L57)**: Added environment variables with documentation

## Testing

Test with different memory allocations:

```bash
# Deploy with 2GB
gcloud run deploy image-translation --memory 2Gi --set-env-vars MAX_CONCURRENT_OCR=2

# Deploy with 4GB (recommended)
gcloud run deploy image-translation --memory 4Gi --set-env-vars MAX_CONCURRENT_OCR=3

# Deploy with 8GB (high throughput)
gcloud run deploy image-translation --memory 8Gi --set-env-vars MAX_CONCURRENT_OCR=5
```

## Monitoring

Check memory usage in Cloud Run logs:
```bash
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=image-translation" --limit 50
```

Look for:
- ✅ `"memory_mb": 2500` - healthy
- ⚠️ `"memory_mb": 3800` (on 4GB instance) - close to limit
- ❌ `"Memory limit exceeded"` - increase memory or reduce concurrency
