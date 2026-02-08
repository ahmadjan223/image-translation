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
import os
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
inpainting_semaphore = asyncio.Semaphore(4)


@router.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Image Translation API",
        "endpoints": {
            "/translate-batch": "POST - Batch translate images in HTML content",
            "/health": "GET - Health check",
            "/memory": "GET - Memory usage monitoring"
        }
    }


@router.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


@router.get("/memory")
async def memory_status():
    """Memory usage monitoring endpoint."""
    import psutil
    import gc
    
    process = psutil.Process()
    mem_info = process.memory_info()
    
    # Get garbage collector stats
    gc_stats = {
        f"generation_{i}": gc.get_count()[i] 
        for i in range(3)
    }
    
    return {
        "memory_mb": round(mem_info.rss / 1024 / 1024, 2),
        "memory_percent": round(process.memory_percent(), 2),
        "gc_stats": gc_stats,
        "open_files": len(process.open_files()),
        "threads": process.num_threads()
    }


# ============================================================================
# Phase 1: OCR Extraction (parallel, in-memory)
# ============================================================================

async def extract_chinese_from_image(
    image_url: str,
    image_index: int,
    images_dir: Path,
    offer_id: str,
    ocr: PaddleOCR,
    client: httpx.AsyncClient  # ← Shared HTTP client
) -> tuple[Optional[bytes], List[Dict]]:
    """
    Download image and extract Chinese text via OCR (in-memory, no disk I/O).
    Returns (image_bytes, ch_items)
    """
    request_id = f"{offer_id}-img{image_index}"
    
    logger.info(f"[{request_id}] Starting OCR extraction for image {image_index + 1}")
    
    try:
        # Download to memory
        logger.debug(f"[{request_id}] Downloading image from {image_url[:100]}...")
        response = await client.get(image_url, headers={"User-Agent": USER_AGENT})
        response.raise_for_status()
        img_bytes = response.content
        
        logger.info(f"[{request_id}] Downloaded {len(img_bytes)} bytes")
        print(f"   [{image_index + 1}] ✅ Downloaded")
        
        # OCR from memory
        logger.debug(f"[{request_id}] Waiting for OCR semaphore...")
        async with ocr_semaphore:
            logger.info(f"[{request_id}] Acquired OCR semaphore, starting OCR processing")
            loop = asyncio.get_running_loop()
            
            def run_ocr_in_memory():
                from utils.image import load_image_from_bytes
                import gc
                import glob
                import tempfile
                
                img_bgr = load_image_from_bytes(img_bytes)
                logger.debug(f"[{request_id}] Loaded image to BGR format: {img_bgr.shape}")
                
                # Run OCR
                logger.info(f"[{request_id}] Running PaddleOCR...")
                outputs = ocr.ocr(img_bgr)
                logger.info(f"[{request_id}] PaddleOCR completed")
                
                # Free the image immediately after OCR
                del img_bgr
                gc.collect()
                
                # Save OCR result to JSON (correct way to handle PaddleOCR results)
                with tempfile.TemporaryDirectory() as tmpdir:
                    for res in outputs:
                        if res is not None:
                            res.save_to_json(tmpdir)
                    
                    # Find the JSON file that was just created
                    jfiles = sorted(glob.glob(os.path.join(tmpdir, "*.json")), key=os.path.getmtime)
                    if not jfiles:
                        # No detections - return empty structure
                        logger.info(f"[{request_id}] No OCR detections found")
                        data = {
                            "rec_texts": [],
                            "rec_scores": [],
                            "rec_boxes": [],
                            "rec_polys": []
                        }
                    else:
                        # Load the JSON file
                        with open(jfiles[-1], "r", encoding="utf-8") as f:
                            data = json.load(f)
                        logger.info(f"[{request_id}] Loaded OCR data: {len(data.get('rec_texts', []))} detections")
                
                return data
            
            ocr_data = await loop.run_in_executor(None, run_ocr_in_memory)
            logger.info(f"[{request_id}] OCR executor completed, filtering Chinese items...")
            ch_items = get_chinese_items(ocr_data, conf_thresh=OCR_CONFIDENCE_THRESH)
            
            logger.info(f"[{request_id}] Released OCR semaphore. Found {len(ch_items)} Chinese regions out of {len(ocr_data.get('rec_texts', []))} total detections")
            print(f"   [{image_index + 1}] ✅ OCR: {len(ch_items)} Chinese regions")
        
        logger.info(f"[{request_id}] Successfully completed OCR extraction")
        return img_bytes, ch_items
    
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        logger.error(f"[{request_id}] OCR extraction failed: {error_msg}")
        print(f"   [{image_index + 1}] ❌ {error_msg}")
        return None, []


# ============================================================================
# Phase 2: Inpainting & Overlay (parallel, with pre-translated text)
# ============================================================================

async def inpaint_and_overlay_image(
    image_url: str,
    image_index: int,
    image_bytes: Optional[bytes],
    ch_items: List[Dict],
    en_translations: List[str],
    images_dir: Path,
    offer_id: str,
    results: List,
    url_mapping: Dict[str, str],
    lama: Optional[SimpleLama]
):
    """
    Inpaint and overlay English text on image (in-memory, no disk I/O).
    """
    request_id = f"{offer_id}-img{image_index}"
    
    try:
        if not image_bytes:
            raise ValueError("No image data")
        
        # Get event loop for async operations
        loop = asyncio.get_running_loop()
        
        # Add translations to ch_items
        for it, en in zip(ch_items, en_translations):
            it["en"] = en
        
        # No Chinese text - just convert to WebP
        if not ch_items:
            print(f"   [{image_index + 1}] ⭐️ No Chinese text")
            img_bytes = convert_to_webp(
                Image.open(io.BytesIO(image_bytes)),
                lossless=WEBP_LOSSLESS,
                quality=WEBP_QUALITY,
                method=WEBP_METHOD
            )
            chinese_count = 0
        else:
            # Inpaint and overlay in-memory
            print(f"   [{image_index + 1}] 🎨 Inpainting...")
            
            async with inpainting_semaphore:
                def inpaint_and_overlay():
                    from utils.image import load_image_from_bytes
                    import gc
                    
                    img_bgr = load_image_from_bytes(image_bytes)
                    H, W = get_image_dimensions(img_bgr)
                    mask = create_mask_from_items(ch_items, H, W, pad=MASK_PAD)
                    inpainted_bgr = inpaint_with_lama(img_bgr, mask, lama, request_id)
                    
                    # Free large NumPy arrays immediately
                    del img_bgr
                    del mask
                    
                    final_pil = overlay_english_text(inpainted_bgr, ch_items, FONT_PATH)
                    del inpainted_bgr  # Free ~10-40MB
                    
                    result_bytes = convert_to_webp(
                        final_pil,
                        lossless=WEBP_LOSSLESS,
                        quality=WEBP_QUALITY,
                        method=WEBP_METHOD
                    )
                    del final_pil  # Free ~40MB RGBA image
                    gc.collect()  # Force cleanup of large objects
                    
                    return result_bytes
                
                img_bytes = await loop.run_in_executor(None, inpaint_and_overlay)
            
            chinese_count = len(ch_items)
        
        # Save to disk so user can inspect translated image
        local_filename = f"translated_{image_index}.webp"
        local_path = images_dir / local_filename
        await loop.run_in_executor(
            None,
            lambda: local_path.write_bytes(img_bytes)
        )
        print(f"   [{image_index + 1}] 💾 Saved to {local_path}")
        
        # Upload the EXACT same bytes to GCS (no re-conversion)
        public_url = None
        # if gcp_storage.is_available():
            # blob_path = await loop.run_in_executor(
            #     None,
            #     gcp_storage.upload_image,
            #     img_bytes,
            #     offer_id,
            #     f"image_{image_index}",
            #     "image/webp"
            # )
            # if blob_path:
            #     public_url = await loop.run_in_executor(
            #         None,
            #         gcp_storage.get_public_url,
            #         blob_path,
            #         True  # use_cdn
            #     )
                # # Delete local file after successful upload to save disk space
                # try:
                #     await loop.run_in_executor(None, os.remove, local_path)
                # except Exception as e:
                #     logger.warning(f"[{request_id}] Failed to delete local file: {e}")
        
        # Store results
        results[image_index] = ImageTranslationResult(
            original_url=image_url,
            local_path=str(local_path),
            public_url=public_url,
            chinese_count=chinese_count,
            success=True,
            error=None
        )
        url_mapping[image_url] = public_url or image_url  # Fallback to original if no GCS
        
        # Free img_bytes after all processing is complete
        del img_bytes
        
        print(f"   [{image_index + 1}] ✅ Complete!")
    
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        logger.error(f"[{request_id}] Inpainting failed: {error_msg}")
        print(f"   [{image_index + 1}] ❌ {error_msg}")
        
        results[image_index] = ImageTranslationResult(
            original_url=image_url,
            local_path="",  # No file saved on error
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
        
        # ============ OPTIMIZED WORKFLOW: Collect all Chinese → Single translation ============
        
        # Step 1: Extract Chinese text from HTML
        html_content = original_html
        chinese_items, soup = extract_chinese_text_from_html(html_content, CJK_RE)
        html_chinese_items = [{"text": item["original"], "i": item["index"]} for item in chinese_items]
        
        if chinese_items:
            logger.info(f"[{offer_id}] Found {len(chinese_items)} Chinese text segments in HTML")
            print(f"📝 Found {len(chinese_items)} Chinese text segments in HTML")
            html_content = replace_chinese_with_markers(soup, chinese_items)
        
        # Step 2: Extract Chinese from ALL images in parallel (OCR phase)
        if not image_urls:
            # No images - translate HTML text only if needed
            if chinese_items:
                print(f"🌐 Translating HTML text...")
                loop = asyncio.get_running_loop()
                translations = await loop.run_in_executor(None, translate_items_gemini, html_chinese_items)
                html_content = replace_markers_with_translations(html_content, chinese_items, translations)
            
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
        print(f"🚀 Phase 1: Extracting Chinese from {len(image_urls)} images (parallel OCR)...")
        
        # Create shared HTTP client for all downloads (connection pooling/reuse)
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as shared_client:
            # OCR extraction phase - parallel with shared client
            ocr_tasks = [
                extract_chinese_from_image(url, idx, images_dir, offer_id, OCR_INSTANCE, shared_client)
                for idx, url in enumerate(image_urls)
            ]
            ocr_results = await asyncio.gather(*ocr_tasks, return_exceptions=True)
        
        # Collect all Chinese text from images
        image_chinese_items = {}  # image_index -> list of chinese items
        image_data = {}  # image_index -> image_bytes
        
        for idx, result in enumerate(ocr_results):
            if isinstance(result, Exception):
                logger.error(f"[{offer_id}] Image {idx} OCR failed: {result}")
                continue
            
            img_bytes, ch_items = result
            # Store image bytes if download succeeded
            if img_bytes:
                image_data[idx] = img_bytes
            # Store Chinese items only if they exist
            if ch_items:
                image_chinese_items[str(idx)] = ch_items
        
        total_chinese = len(html_chinese_items) + sum(len(items) for items in image_chinese_items.values())
        logger.info(f"[{offer_id}] Total Chinese text items: {total_chinese} (HTML: {len(html_chinese_items)}, Images: {total_chinese - len(html_chinese_items)})")
        
        # Step 3: Single batch translation for ALL Chinese text
        print(f"\n🌐 Phase 2: Translating ALL {total_chinese} Chinese texts in ONE API call...")
        
        from services.translation import translate_batch_all
        
        loop = asyncio.get_running_loop()
        html_translations, image_translations = await loop.run_in_executor(
            None,
            translate_batch_all,
            html_chinese_items,
            image_chinese_items
        )
        
        logger.info(f"[{offer_id}] ✅ Single batch translation complete!")
        print(f"   ✅ Translation complete (1 API call for all text)")
        
        # Replace HTML markers with translations
        if chinese_items:
            html_content = replace_markers_with_translations(html_content, chinese_items, html_translations)
        
        # Step 4: Process all images with pre-translated text (inpainting phase - parallel)
        print(f"\n🎨 Phase 3: Inpainting {len(image_urls)} images with translated text...")
        
        results = [None] * len(image_urls)
        url_mapping: Dict[str, str] = {}
        
        inpainting_tasks = [
            inpaint_and_overlay_image(
                image_urls[idx],
                idx,
                image_data.get(idx),
                image_chinese_items.get(str(idx), []),
                image_translations.get(str(idx), []),
                images_dir,
                offer_id,
                results,
                url_mapping,
                LAMA_INSTANCE
            )
            for idx in range(len(image_urls))
        ]
        
        await asyncio.gather(*inpainting_tasks, return_exceptions=True)
        
        # CRITICAL: Clear image data from memory immediately after processing
        logger.info(f"[{offer_id}] Clearing image data from memory...")
        image_data.clear()
        del image_data
        del ocr_results
        del image_chinese_items
        
        # Force garbage collection for large objects
        import gc
        gc.collect()
        
        logger.info(f"[{offer_id}] All {len(image_urls)} images processed")
        print(f"\n✅ All {len(image_urls)} images processed")
        
        # Replace URLs in HTML (html_content already has translated text)
        translated_html = replace_image_urls(html_content, url_mapping)
        
        # Save translated HTML
        # html_output_path = session_dir / f"{offer_id}.html"
        # with open(html_output_path, "w", encoding="utf-8") as f:
        #     f.write(translated_html)
        # logger.info(f"[{offer_id}] Saved HTML to {html_output_path}")
        
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
    
    # finally:
    #     # Clean up session directory to prevent disk space exhaustion
    #     import shutil
    #     try:
    #         if session_dir.exists():
    #             await asyncio.get_running_loop().run_in_executor(
    #                 None,
    #                 lambda: shutil.rmtree(session_dir, ignore_errors=True)
    #             )
    #             logger.info(f"[{offer_id}] Cleaned up session directory: {session_dir}")
    #     except Exception as e:
    #         logger.warning(f"[{offer_id}] Failed to clean up session directory: {e}")