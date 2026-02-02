"""
API routes - SIMPLIFIED VERSION using direct instance passing.

Your intuition is correct! Simply create instances once and pass them.
No need for get_ocr_instance() or get_simple_lama() wrappers.

Performance optimizations:
- Create OCR and SimpleLama instances ONCE at startup
- Pass instances to all workers (Python passes by reference, no copies)
- GPU operations use semaphores for sequential execution
- I/O operations run in parallel
"""
import sys
import uuid
import json
import asyncio
import logging
import io
from pathlib import Path
from typing import Dict, List, Optional

import httpx
import cv2
import numpy as np
from PIL import Image
from fastapi import APIRouter, HTTPException
from paddleocr import PaddleOCR
from simple_lama_inpainting import SimpleLama

from config import (
    DOWNLOADS_DIR,
    FONT_PATH,
    OCR_CONFIG,
    WEBP_LOSSLESS,
    WEBP_QUALITY,
    WEBP_METHOD,
    OCR_CONFIDENCE_THRESH,
    MASK_PAD,
    HTTP_TIMEOUT,
    USER_AGENT,
    CJK_RE
)
from models import (
    BatchTranslateRequest,
    BatchTranslateResponse,
    ImageTranslationResult
)
from utils.image import load_image_to_bgr, save_image, get_image_dimensions, convert_to_webp, get_file_extension
from utils.html_parser import (
    extract_image_urls,
    replace_image_urls,
    extract_chinese_text_from_html,
    replace_chinese_with_markers,
    replace_markers_with_translations
)
from services.ocr import get_chinese_items, run_ocr_on_image
from services.translation import translate_items_gemini
from services.inpainting import create_mask_from_items, inpaint_with_lama
from services.text_overlay import overlay_english_text
from gcp_storage import gcp_storage

# Configure structured logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s] %(message)s'
)

router = APIRouter()

# ============================================================================
# Pre-initialize models at module load - SIMPLE AND CLEAR
# ============================================================================

logger.info("🔧 Initializing models at startup...")

# Create OCR instance once - will be passed to all workers
try:
    OCR_INSTANCE = PaddleOCR(**OCR_CONFIG)
    logger.info("✅ OCR model loaded successfully")
except Exception as e:
    logger.error(f"❌ Failed to load OCR model: {e}")
    raise RuntimeError(f"OCR model initialization failed: {e}") from e

# Create SimpleLama instance once - will be passed to all workers
LAMA_INSTANCE: Optional[SimpleLama] = None
try:
    LAMA_INSTANCE = SimpleLama()
    logger.info("✅ SimpleLama model loaded successfully")
except Exception as e:
    logger.warning(f"⚠️  SimpleLama initialization failed: {e}")
    logger.warning("   Will use OpenCV inpainting as fallback")
    # Continue without SimpleLama - will use cv2.inpaint fallback

# GPU operation semaphores - ensure sequential execution to prevent OOM
ocr_semaphore = asyncio.Semaphore(1)
inpainting_semaphore = asyncio.Semaphore(1)


@router.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Image Translation API",
        "endpoints": {
            "/translate-batch": "POST - Batch translate images in HTML content",
            "/health": "GET - Health check"
        }
    }


@router.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


# ============================================================================
# Pipeline stage helper functions
# ============================================================================

async def download_image(
    image_url: str,
    output_path: Path,
    request_id: str
) -> None:
    """Download image from URL and save to disk."""
    logger.info(f"[{request_id}] Downloading image")
    
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        response = await client.get(image_url, headers={"User-Agent": USER_AGENT})
        response.raise_for_status()
    
    with open(output_path, "wb") as f:
        f.write(response.content)


async def handle_no_chinese_text(
    original_path: Path,
    output_path: Path,
    offer_id: str,
    image_index: int,
    request_id: str
) -> tuple[bytes, Optional[str]]:
    """Convert image to WebP when no Chinese text detected."""
    logger.info(f"[{request_id}] No Chinese text detected, converting to WebP")
    
    img_bytes = convert_to_webp(
        original_path,
        output_path,
        lossless=WEBP_LOSSLESS,
        quality=WEBP_QUALITY,
        method=WEBP_METHOD
    )
    
    # Upload to GCS
    public_url = None
    if gcp_storage.is_available():
        logger.info(f"[{request_id}] Uploading to GCS")
        loop = asyncio.get_running_loop()
        public_url = await loop.run_in_executor(
            None,
            gcp_storage.upload_from_bytes,
            img_bytes,
            f"translated/{offer_id}",
            f"image_{image_index}"
        )
    
    return img_bytes, public_url


async def translate_and_inpaint(
    original_path: Path,
    ch_items: List[Dict],
    output_path: Path,
    offer_id: str,
    image_index: int,
    request_id: str,
    lama: Optional[SimpleLama]
) -> tuple[bytes, Optional[str]]:
    """Translate Chinese text and inpaint image."""
    # Translation
    logger.info(f"[{request_id}] Translating {len(ch_items)} text regions")
    loop = asyncio.get_running_loop()
    en_lines = await loop.run_in_executor(None, translate_items_gemini, ch_items)
    
    for it, en in zip(ch_items, en_lines):
        it["en"] = en
    
    # Inpainting and overlay
    logger.info(f"[{request_id}] Inpainting and overlaying text")
    
    async with inpainting_semaphore:
        def inpaint_and_overlay():
            img_bgr = load_image_to_bgr(str(original_path))
            H, W = get_image_dimensions(img_bgr)
            mask = create_mask_from_items(ch_items, H, W, pad=MASK_PAD)
            inpainted_bgr = inpaint_with_lama(img_bgr, mask, lama, request_id)
            final_pil = overlay_english_text(inpainted_bgr, ch_items, FONT_PATH)
            
            # Convert to WebP
            return convert_to_webp(
                final_pil,
                lossless=WEBP_LOSSLESS,
                quality=WEBP_QUALITY,
                method=WEBP_METHOD
            )
        
        img_bytes = await loop.run_in_executor(None, inpaint_and_overlay)
    
    # Save locally
    with open(output_path, "wb") as f:
        f.write(img_bytes)
    
    # Upload to GCS
    public_url = None
    if gcp_storage.is_available():
        logger.info(f"[{request_id}] Uploading to GCS")
        public_url = await loop.run_in_executor(
            None,
            gcp_storage.upload_from_bytes,
            img_bytes,
            f"translated/{offer_id}",
            f"image_{image_index}"
        )
    
    return img_bytes, public_url


# ============================================================================
# Streaming pipeline worker - receives model instances as parameters
# ============================================================================

async def process_single_image_pipeline(
    image_url: str,
    image_index: int,
    images_dir: Path,
    offer_id: str,
    results: List,
    url_mapping: Dict[str, str],
    ocr: PaddleOCR,                    # ← Receive OCR instance
    lama: Optional[SimpleLama]         # ← Receive SimpleLama instance (can be None)
):
    """
    Process single image through the translation pipeline.
    
    Coordinates the full pipeline: download → OCR → translate → inpaint → upload.
    """
    request_id = f"{offer_id}-img{image_index}"
    original_path = None
    
    try:
        # Stage 1: Download
        print(f"   [{image_index + 1}] 📥 Downloading...")
        
        # Get content type for extension detection
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            response = await client.get(image_url, headers={"User-Agent": USER_AGENT})
            response.raise_for_status()
            content_type = response.headers.get("content-type", "")
        
        ext = get_file_extension(image_url, content_type)
        original_path = images_dir / f"original_{image_index}{ext}"
        
        with open(original_path, "wb") as f:
            f.write(response.content)
        
        print(f"   [{image_index + 1}] ✅ Downloaded")
        
        # Stage 2: OCR
        print(f"   [{image_index + 1}] 🔍 Running OCR...")
        
        async with ocr_semaphore:
            ocr_dir = images_dir / f"ocr_{image_index}"
            ocr_dir.mkdir(exist_ok=True)
            
            loop = asyncio.get_running_loop()
            ocr_data, fed_path = await loop.run_in_executor(
                None,
                run_ocr_on_image,
                str(original_path),
                str(ocr_dir),
                ocr
            )
            ch_items = get_chinese_items(ocr_data, conf_thresh=OCR_CONFIDENCE_THRESH)
            
            print(f"   [{image_index + 1}] ✅ OCR: {len(ch_items)} Chinese regions")
        
        output_path = images_dir / f"translated_{image_index}.webp"
        
        # Stage 3: Process based on Chinese text presence
        if not ch_items:
            print(f"   [{image_index + 1}] ⭐️ No Chinese text")
            img_bytes, public_url = await handle_no_chinese_text(
                original_path, output_path, offer_id, image_index, request_id
            )
            chinese_count = 0
        else:
            print(f"   [{image_index + 1}] 🌐 Translating...")
            img_bytes, public_url = await translate_and_inpaint(
                original_path, ch_items, output_path, offer_id, image_index, request_id, lama
            )
            chinese_count = len(ch_items)
        
        # Store results
        results[image_index] = ImageTranslationResult(
            original_url=image_url,
            local_path=str(output_path),
            public_url=public_url,
            chinese_count=chinese_count,
            success=True,
            error=None
        )
        url_mapping[image_url] = public_url or str(output_path)
        print(f"   [{image_index + 1}] ✅ Complete!")
    
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        logger.error(f"[{request_id}] Pipeline failed: {error_msg}")
        print(f"   [{image_index + 1}] ❌ {error_msg}")
        print(f"   [{image_index + 1}] ❌ {error_msg}")
        
        results[image_index] = ImageTranslationResult(
            original_url=image_url,
            local_path=str(original_path) if original_path else "",
            public_url=None,
            chinese_count=0,
            success=False,
            error=error_msg[:200]
        )



@router.post("/translate-batch", response_model=BatchTranslateResponse)
async def translate_batch(request: BatchTranslateRequest):
    """
    Batch translate images AND Chinese text in HTML.
    
    Flow:
    1. Extract and translate Chinese text from HTML
    2. Process images (OCR, translate, inpaint)
    3. Return HTML with both text and images translated
    """
    try:
        offer_id = request.offer_id or str(uuid.uuid4())
        logger.info(f"[{offer_id}] Starting batch translation request")
        session_dir = DOWNLOADS_DIR / offer_id
        session_dir.mkdir(exist_ok=True)
        
        # ============ Step 0: Extract image URLs from ORIGINAL HTML first ============
        original_html = request.description
        image_urls = extract_image_urls(original_html)
        logger.info(f"[{offer_id}] Found {len(image_urls)} images in HTML")
        
        # ============ Step 1: Extract and translate Chinese text from HTML ============
        html_content = original_html
        chinese_items, soup = extract_chinese_text_from_html(html_content, CJK_RE)
        
        if chinese_items:
            logger.info(f"[{offer_id}] Found {len(chinese_items)} Chinese text segments in HTML")
            print(f"📝 Found {len(chinese_items)} Chinese text segments")
            
            # Replace Chinese text with markers
            html_content = replace_chinese_with_markers(soup, chinese_items)
            
            # Translate text using Gemini
            print(f"   Translating HTML text...")
            text_items = [{"text": item["original"], "i": item["index"]} for item in chinese_items]
            loop = asyncio.get_running_loop()
            translations = await loop.run_in_executor(None, translate_items_gemini, text_items)
            
            # Replace markers with translations
            html_content = replace_markers_with_translations(html_content, chinese_items, translations)
            logger.info(f"[{offer_id}] Translated {len(translations)} text segments")
            print(f"   ✅ HTML text translated")
        else:
            logger.info(f"[{offer_id}] No Chinese text found in HTML")
        
        # ============ Step 2: Process images ============
        
        if not image_urls:
            # Save HTML with translated text (even if no images)
            html_output_path = session_dir / f"{offer_id}.html"
            with open(html_output_path, "w", encoding="utf-8") as f:
                f.write(html_content)
            
            return BatchTranslateResponse(
                offer_id=offer_id,
                message="⚠️ No images found in HTML content. Text translated.",
                total_images=0,
                successful_translations=0,
                failed_translations=0,
                translated_html=html_content,
                image_results=[]
            )
        
        images_dir = session_dir / "images"
        images_dir.mkdir(exist_ok=True)
        
        logger.info(f"[{offer_id}] Processing {len(image_urls)} images")
        print(f"🚀 Processing {len(image_urls)} images")
        
        # Shared state
        results = [None] * len(image_urls)
        url_mapping: Dict[str, str] = {}
        
        # Create tasks - pass model instances to each worker
        pipeline_tasks = [
            process_single_image_pipeline(
                url,
                idx,
                images_dir,
                offer_id,
                results,
                url_mapping,
                ocr=OCR_INSTANCE,      # ← Pass OCR instance
                lama=LAMA_INSTANCE     # ← Pass SimpleLama instance
            )
            for idx, url in enumerate(image_urls)
        ]
        
        # Execute all pipelines
        await asyncio.gather(*pipeline_tasks, return_exceptions=True)
        
        logger.info(f"[{offer_id}] All {len(image_urls)} images processed")
        print(f"\n✅ All {len(image_urls)} images processed")
        
        # Replace URLs in HTML (html_content already has translated text)
        translated_html = replace_image_urls(html_content, url_mapping)
        
        # Save translated HTML
        html_output_path = session_dir / f"{offer_id}.html"
        with open(html_output_path, "w", encoding="utf-8") as f:
            f.write(translated_html)
        logger.info(f"[{offer_id}] Saved HTML to {html_output_path}")
        
        # Count successes and failures
        successful = sum(1 for r in results if r and r.success)
        failed = len(results) - successful
        
        logger.info(f"[{offer_id}] Batch complete: {successful}/{len(results)} successful, {failed} failed")
        
        return BatchTranslateResponse(
            offer_id=offer_id,
            message=f"✅ Pipeline complete. {successful}/{len(results)} images translated.",
            total_images=len(results),
            successful_translations=successful,
            failed_translations=failed,
            translated_html=translated_html,
            image_results=results
        )
    
    except Exception as e:
        import traceback
        error_details = f"{type(e).__name__}: {str(e)}"
        logger.error(f"Batch translation error: {error_details}")
        logger.debug(f"Traceback: {traceback.format_exc()}")
        sys.stdout.flush()
        raise HTTPException(status_code=500, detail=f"Batch translation error: {str(e)}")