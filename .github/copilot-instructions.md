# Image Translation API - Copilot Instructions

## Project Overview
FastAPI service that translates Chinese text in product images to English using OCR, LLM translation, and inpainting. Deploys to Google Cloud Run with GPU support for local development.

**Core Pipeline:** Download image → PaddleOCR (detect Chinese) → Gemini (translate) → SimpleLama (inpaint) → Pillow (overlay English text) → GCP Storage (optional upload)

## Architecture & Key Components

### Service Layer ([services/](services/))
- **ocr.py**: PaddleOCR wrapper, filters Chinese text using `CJK_RE` regex (`[\u4e00-\u9fff]`)
- **translation.py**: Gemini API client with structured JSON output (`{"i": idx, "en": text}`)
- **inpainting.py**: SimpleLama model removes Chinese text using dilated masks (`cv2.dilate` with `pad=6`)
- **text_overlay.py**: Pillow text rendering with smart wrapping, shadow effects, and dynamic font sizing (12-120px)

### API Routes ([routes/api.py](routes/api.py))
**Critical Pattern:** Pre-initialized global model instances (`OCR_INSTANCE`, `LAMA_INSTANCE`) passed to async workers to avoid repeated model loading. GPU operations use `asyncio.Semaphore(1)` for sequential execution (prevents OOM).

- `/download` - Downloads images via httpx with User-Agent spoofing
- `/ocr` - Returns Chinese text detections with confidence scores
- `/translate` - Full single-image pipeline
- `/translate-batch` - Parallel HTML batch processing with `asyncio.gather()` and semaphore-controlled GPU ops

### Configuration ([config.py](config.py))
**Environment Dependencies:** 
- Sets `LD_LIBRARY_PATH` for CUDA libraries in venv (`nvidia/cublas/lib`, `nvidia/cudnn/lib`)
- Disables threading/SIMD (`OMP_NUM_THREADS=1`, `OPENBLAS_CORETYPE=generic`) to prevent Cloud Run SIGFPE crashes
- Disables PaddlePaddle oneDNN (`FLAGS_use_mkldnn=false`) for Docker compatibility

**Key Settings:**
- `OCR_CONFIG`: PaddleOCR parameters - CPU mode, 4000px side limit, 0.6 confidence threshold
- `TRANSLATION_SYSTEM_PROMPT`: Gemini prompt emphasizes brevity (`en length <= max_chars`)
- `FONT_PATH`: DejaVuSans-Bold at `/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf`

### Data Flow
```
HTML → extract_image_urls() → [parallel downloads]
  ↓ (sequential per image via semaphore)
PaddleOCR → get_chinese_items() → filter by CJK regex
  ↓ (parallel across images)
Gemini batch translate → translate_items_gemini()
  ↓ (sequential per image via semaphore)
SimpleLama inpaint → overlay_english_text() → WebP bytes
  ↓ (parallel uploads)
GCP Storage → replace_image_urls() → translated HTML
```

## Development Workflows

### Local Development (GPU)
```bash
# One-time setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python download_models.py  # Cache PaddleOCR models to ~/.paddlex/

# Run with GPU support
./start_with_gpu.sh  # Sets CUDA paths, runs uvicorn on port 8080
```

### Cloud Run Deployment
```bash
./deploy.sh  # Builds Docker, pre-downloads models, pushes to GCR, deploys to Cloud Run
```
**Critical:** [Dockerfile](Dockerfile) runs `python download_models.py` during build to cache models in `/root/.paddlex/` (ephemeral storage would cause cold-start failures).

### Testing
```bash
# Test GCP Storage integration
python test_gcp_storage.py

# Manual API testing
curl -X POST http://localhost:8080/translate-batch \
  -H "Content-Type: application/json" \
  -d '{"offer_id": "test", "description": "<img src=\"https://example.com/image.jpg\">"}'
```

## Project-Specific Patterns

### Async/Thread Pool Hybrid
**Why:** OCR, inpainting, and Pillow are synchronous CPU-bound operations. Running them directly in async functions blocks the event loop, causing FastAPI to return 200 OK before processing completes (Cloud Run issue).

**Pattern:** Use `loop.run_in_executor(None, sync_func, args)` to run blocking operations in default ThreadPoolExecutor:
```python
async with ocr_semaphore:  # Prevent concurrent GPU ops
    ocr_data = await loop.run_in_executor(None, run_ocr_on_image, path, ocr_dir, OCR_INSTANCE)
```

### Model Instance Passing (Not Singletons)
**Anti-pattern:** Using `get_ocr_instance()` or thread-local storage patterns.
**Correct:** Pre-initialize at module load, pass instances as function parameters:
```python
# At module level in routes/api.py
OCR_INSTANCE = PaddleOCR(**OCR_CONFIG)
LAMA_INSTANCE = SimpleLama()

# Pass to workers
await process_image(image_path, ocr=OCR_INSTANCE, lama=LAMA_INSTANCE)
```

### Text Overlay Sizing Strategy
[text_overlay.py](services/text_overlay.py) uses binary search for font size (12-120px, step=2) that maximizes size while fitting text in bounding box with wrapping. Falls back to 2 lines if `len(text) > SPLIT_THRESHOLD` (12 chars).

### GCP Storage Pattern
[gcp_storage.py](gcp_storage.py) uploads images with `{folder}/{offer_id}/image_{idx}.webp` structure. CDN URLs replace original HTML image URLs via `replace_image_urls()` regex substitution.

## Common Issues & Solutions

1. **SIGFPE crashes in Cloud Run**: Threading/SIMD conflicts resolved by `OMP_NUM_THREADS=1` in [config.py](config.py#L51)
2. **Models re-downloading**: Run [download_models.py](download_models.py) in Dockerfile, NOT at runtime
3. **OOM errors**: Use `asyncio.Semaphore(1)` for GPU operations (`ocr_semaphore`, `inpainting_semaphore`)
4. **Incomplete responses**: Wrap CPU-bound ops in `run_in_executor()` to avoid blocking event loop

## Key Files
- [CLOUD_RUN_FIX.md](CLOUD_RUN_FIX.md) - Detailed deployment troubleshooting
- [utils/html_parser.py](utils/html_parser.py) - BeautifulSoup image URL extraction/replacement
- [models.py](models.py) - Pydantic request/response schemas
