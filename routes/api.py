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
    USER_AGENT
)
from models import (
    BatchTranslateRequest,
    BatchTranslateResponse,
    ImageTranslationResult
)
from utils.image import load_image_to_bgr, save_image, get_image_dimensions
from utils.html_parser import extract_image_urls, replace_image_urls
from services.ocr import get_chinese_items
from services.translation import translate_items_gemini
from services.inpainting import create_mask_from_items
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

# sara ocr related stuff is function mein he
def run_ocr_on_image(image_path: str, outdir: str, ocr: PaddleOCR):
    """
    Run OCR on image using provided OCR instance.
    
    Uses ocr.predict() like the working notebook version.
    """
    import os
    import glob
    
    os.makedirs(outdir, exist_ok=True)
    
    img_bgr = load_image_to_bgr(image_path)
    
    # Save the exact bitmap fed to OCR
    fed_path = os.path.join(outdir, "fed_to_ocr.png")
    cv2.imwrite(fed_path, img_bgr)
    
    # Run OCR using predict() which returns OCRResult objects
    outputs = ocr.predict(fed_path)
    
    # Save OCRResult to JSON (like the working notebook code)
    for res in outputs:
        if res is not None:
            res.save_to_json(outdir)
    
    # Find the JSON file that was just created
    jfiles = sorted(glob.glob(os.path.join(outdir, "*.json")), key=os.path.getmtime)
    if not jfiles:
        # No detections - return empty structure
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
    
    return data, fed_path


def inpaint_with_lama(img_bgr, mask, lama: Optional[SimpleLama], request_id: str = "unknown"):
    """
    Inpaint image using SimpleLama with OpenCV fallback.
    
    Args:
        img_bgr: Input BGR image
        mask: Binary mask for inpainting
        lama: SimpleLama instance (can be None if initialization failed)
        request_id: Request identifier for logging
    
    Returns:
        Inpainted BGR image
    """
    # If SimpleLama not available, use OpenCV directly
    if lama is None:
        logger.debug(f"[{request_id}] Using OpenCV inpainting (SimpleLama not available)")
        return cv2.inpaint(img_bgr, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    
    # Try SimpleLama first
    try:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        inpaint_pil = lama(img_rgb, mask)
        logger.debug(f"[{request_id}] SimpleLama inpainting successful")
        return cv2.cvtColor(np.array(inpaint_pil), cv2.COLOR_RGB2BGR)
    except Exception as e:
        logger.error(f"[{request_id}] SimpleLama failed: {type(e).__name__}: {str(e)[:200]}")
        logger.info(f"[{request_id}] Falling back to OpenCV inpainting")
        return cv2.inpaint(img_bgr, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)


def convert_to_webp(image_source, output_path: Optional[Path] = None) -> bytes:
    """
    Convert image to WebP format using configured settings.
    
    Args:
        image_source: Either a file path (str/Path) or PIL Image object
        output_path: Optional path to save the WebP file
        
    Returns:
        WebP image as bytes
    """
    # Load image if path provided, otherwise use PIL Image directly
    if isinstance(image_source, (str, Path)):
        img = Image.open(image_source).convert("RGB")
    else:
        img = image_source.convert("RGB")
    
    # Convert to WebP
    img_buffer = io.BytesIO()
    img.save(img_buffer, format="WEBP", lossless=WEBP_LOSSLESS, quality=WEBP_QUALITY, method=WEBP_METHOD)
    img_bytes = img_buffer.getvalue()
    
    # Optionally save to file
    if output_path:
        with open(output_path, "wb") as f:
            f.write(img_bytes)
    
    return img_bytes


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
    Process single image through the pipeline.
    
    Receives model instances as parameters - clean and explicit.
    No copies created, just passing references.
    """
    request_id = f"{offer_id}-img{image_index}"
    original_path = None
    try:
        # Stage 1: Download
        logger.info(f"[{request_id}] 📥 Downloading...")
        print(f"   [{image_index + 1}] 📥 Downloading...")
        sys.stdout.flush()
        
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            response = await client.get(
                image_url,
                headers={"User-Agent": USER_AGENT}
            )
            response.raise_for_status()
        
        # Determine file extension
        content_type = response.headers.get("content-type", "").lower()
        if "jpeg" in content_type or "jpg" in content_type:
            ext = ".jpg"
        elif "webp" in content_type:
            ext = ".webp"
        elif "png" in content_type:
            ext = ".png"
        else:
            url_path = image_url.split("?")[0]
            ext = Path(url_path).suffix or ".jpg"
        
        original_path = images_dir / f"original_{image_index}{ext}"
        with open(original_path, "wb") as f:
            f.write(response.content)
        
        print(f"   [{image_index + 1}] ✅ Downloaded")
        sys.stdout.flush()
        
        # Stage 2: OCR
        print(f"   [{image_index + 1}] 🔍 Waiting for OCR...")
        sys.stdout.flush()
        
        async with ocr_semaphore:
            print(f"   [{image_index + 1}] 🔍 Running OCR...")
            
            ocr_dir = images_dir / f"ocr_{image_index}"
            ocr_dir.mkdir(exist_ok=True)
            
            loop = asyncio.get_running_loop()
            ocr_data, fed_path = await loop.run_in_executor(
                None,
                run_ocr_on_image,
                str(original_path),
                str(ocr_dir),
                ocr  # ← Pass the OCR instance
            )
            ch_items = get_chinese_items(ocr_data, conf_thresh=OCR_CONFIDENCE_THRESH)
            
            print(f"   [{image_index + 1}] ✅ OCR done: {len(ch_items)} Chinese regions")
            sys.stdout.flush()
        
        # If no Chinese text, skip translation/inpainting
        if not ch_items:
            print(f"   [{image_index + 1}] ⭐️ No Chinese text, converting...")
            sys.stdout.flush()
            
            output_path = images_dir / f"translated_{image_index}.webp"
            img_bytes = convert_to_webp(original_path, output_path)
            
            # Upload to GCS
            public_url = None
            if gcp_storage.is_available():
                print(f"   [{image_index + 1}] ☁️  Uploading...")
                sys.stdout.flush()
                loop = asyncio.get_running_loop()
                public_url = await loop.run_in_executor(
                    None,
                    gcp_storage.upload_from_bytes,
                    img_bytes,
                    f"translated/{offer_id}",
                    f"image_{image_index}"
                )
                print(f"   [{image_index + 1}] ✅ Uploaded")
                sys.stdout.flush()
            
            results[image_index] = ImageTranslationResult(
                original_url=image_url,
                local_path=str(output_path),
                public_url=public_url,
                chinese_count=0,
                success=True,
                error=None
            )
            url_mapping[image_url] = public_url or str(output_path)
            print(f"   [{image_index + 1}] 🎉 Complete!")
            sys.stdout.flush()
            return
        
        # Stage 3: Translation
        print(f"   [{image_index + 1}] 🌐 Translating {len(ch_items)} regions...")
        sys.stdout.flush()
        
        loop = asyncio.get_running_loop()
        en_lines = await loop.run_in_executor(
            None,
            translate_items_gemini,
            ch_items
        )
        for it, en in zip(ch_items, en_lines):
            it["en"] = en
        
        print(f"   [{image_index + 1}] ✅ Translated")
        sys.stdout.flush()
        
        # Stage 4: Inpainting
        print(f"   [{image_index + 1}] 🎨 Waiting for inpainting...")
        sys.stdout.flush()
        
        async with inpainting_semaphore:
            print(f"   [{image_index + 1}] 🎨 Inpainting...")
            sys.stdout.flush()
            
            loop = asyncio.get_running_loop()
            
            def inpaint_and_overlay():
                img_bgr = load_image_to_bgr(str(original_path))
                H, W = get_image_dimensions(img_bgr)
                mask = create_mask_from_items(ch_items, H, W, pad=MASK_PAD)
                inpainted_bgr = inpaint_with_lama(img_bgr, mask, lama, request_id)  # ← Pass lama instance and request_id
                final_pil = overlay_english_text(inpainted_bgr, ch_items, FONT_PATH)
                
                # Convert to WebP
                img_buffer = io.BytesIO()
                final_pil.convert("RGB").save(img_buffer, format="WEBP", lossless=WEBP_LOSSLESS, quality=WEBP_QUALITY, method=WEBP_METHOD)
                return img_buffer.getvalue()
            
            img_bytes = await loop.run_in_executor(None, inpaint_and_overlay)
            
            print(f"   [{image_index + 1}] ✅ Inpainted")
            sys.stdout.flush()
        
        # Save locally
        output_path = images_dir / f"translated_{image_index}.webp"
        with open(output_path, "wb") as f:
            f.write(img_bytes)
        
        # Stage 5: Upload to GCS
        public_url = None
        if gcp_storage.is_available():
            print(f"   [{image_index + 1}] ☁️  Uploading...")
            sys.stdout.flush()
            loop = asyncio.get_running_loop()
            public_url = await loop.run_in_executor(
                None,
                gcp_storage.upload_from_bytes,
                img_bytes,
                f"translated/{offer_id}",
                f"image_{image_index}"
            )
            print(f"   [{image_index + 1}] ✅ Uploaded")
            sys.stdout.flush()
        
        results[image_index] = ImageTranslationResult(
            original_url=image_url,
            local_path=str(output_path),
            public_url=public_url,
            chinese_count=len(ch_items),
            success=True,
            error=None
        )
        url_mapping[image_url] = public_url or str(output_path)
        
        print(f"   [{image_index + 1}] 🎉 Complete!")
        sys.stdout.flush()
    
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        logger.error(f"[{request_id}] ❌ Pipeline failed: {error_msg}")
        print(f"   [{image_index + 1}] ❌ ERROR: {error_msg}")
        sys.stdout.flush()
        
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
    Batch translate images - SIMPLIFIED VERSION.
    
    Simply passes OCR and SimpleLama instances to workers.
    Clean, explicit, and efficient - your intuition was correct!
    """
    try:
        offer_id = request.offer_id or str(uuid.uuid4())
        logger.info(f"[{offer_id}] Starting batch translation request")
        session_dir = DOWNLOADS_DIR / offer_id
        session_dir.mkdir(exist_ok=True)
        
        # Extract image URLs from HTML
        image_urls = extract_image_urls(request.description)
        logger.info(f"[{offer_id}] Found {len(image_urls)} images in HTML")
        
        if not image_urls:
            return BatchTranslateResponse(
                offer_id=offer_id,
                message="⚠️ No images found in HTML content.",
                total_images=0,
                successful_translations=0,
                failed_translations=0,
                translated_html=request.description,
                image_results=[]
            )
        
        images_dir = session_dir / "images"
        images_dir.mkdir(exist_ok=True)
        
        print(f"🚀 Processing {len(image_urls)} images")
        print(f"   Using global OCR and SimpleLama instances")
        print(f"   GPU ops sequential, I/O parallel\n")
        sys.stdout.flush()
        
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
        
        print(f"\n✅ All {len(image_urls)} images processed")
        sys.stdout.flush()
        
        # Replace URLs in HTML
        translated_html = replace_image_urls(request.description, url_mapping)
        
        # Save translated HTML
        html_output_path = session_dir / f"{offer_id}.html"
        with open(html_output_path, "w", encoding="utf-8") as f:
            f.write(translated_html)
        print(f"💾 Saved: {html_output_path}")
        sys.stdout.flush()
        
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
        print(f"❌ Batch translation error: {str(e)}")
        print(f"📋 Traceback: {traceback.format_exc()}")
        sys.stdout.flush()
        raise HTTPException(status_code=500, detail=f"Batch translation error: {str(e)}")