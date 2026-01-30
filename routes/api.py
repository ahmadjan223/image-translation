"""
API routes for the Image Translation API.

Performance optimizations (GPU-optimized streaming pipeline):
- Each image flows: Download → OCR → Translate → Inpaint → Upload immediately
- GPU operations (OCR, inpainting) use semaphores for sequential execution
- I/O operations (download, translate, upload) run in parallel
- No waiting for batch phases - maximum throughput
"""
import sys
import uuid
import json
import asyncio
from pathlib import Path
from typing import Dict, Tuple, List

import httpx
import cv2
from fastapi import APIRouter, HTTPException

from config import DOWNLOADS_DIR, FONT_PATH
from models import (
    ImageDownloadRequest,
    ImageDownloadResponse,
    OCRRequest,
    OCRResponse,
    TranslateRequest,
    TranslateResponse,
    BatchTranslateRequest,
    BatchTranslateResponse,
    ImageTranslationResult
)
from utils.image import load_image_to_bgr, save_image, get_image_dimensions
from utils.html_parser import extract_image_urls, replace_image_urls
from services.ocr import ocr_predict_to_json, get_chinese_items
from services.translation import translate_items_gemini
from services.inpainting import create_mask_from_items, inpaint_image
from services.text_overlay import overlay_english_text
from gcp_storage import gcp_storage

router = APIRouter()

# GPU operation semaphores - ensure sequential execution to prevent OOM
ocr_semaphore = asyncio.Semaphore(1)
inpainting_semaphore = asyncio.Semaphore(1)


@router.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Image Translation API",
        "endpoints": {
            "/download": "POST - Download image from URL",
            "/ocr": "POST - Run OCR on downloaded image",
            "/translate": "POST - Full pipeline: OCR → Translate → Inpaint → Overlay",
            "/translate-batch": "POST - Batch translate images in HTML content",
            "/health": "GET - Health check"
        }
    }


@router.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


@router.post("/download", response_model=ImageDownloadResponse)
async def download_image(request: ImageDownloadRequest):
    """
    Milestone 1: Download image from URL and save it locally.
    
    Args:
        request: Contains the image URL and optional session ID.
        
    Returns:
        Session ID, image path, and image dimensions.
    """
    try:
        offer_id = request.offer_id or str(uuid.uuid4())
        session_dir = DOWNLOADS_DIR / offer_id
        session_dir.mkdir(exist_ok=True)
        
        image_url = str(request.image_url)
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                image_url,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
            )
            response.raise_for_status()
        
        # Determine file extension from content-type or URL
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
        
        # Save downloaded image
        image_path = session_dir / f"original{ext}"
        with open(image_path, "wb") as f:
            f.write(response.content)
        
        # Get image dimensions
        img_bgr = load_image_to_bgr(str(image_path))
        height, width = get_image_dimensions(img_bgr)
        
        return ImageDownloadResponse(
            offer_id=offer_id,
            message="✅ Image downloaded successfully",
            image_path=str(image_path),
            image_size={"width": width, "height": height}
        )
    
    except httpx.RequestError as e:
        raise HTTPException(status_code=400, detail=f"Failed to download image: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@router.post("/ocr", response_model=OCRResponse)
async def run_ocr(request: OCRRequest):
    """
    Milestone 2: Run OCR on downloaded image and extract Chinese text.
    
    Args:
        request: Contains the session ID.
        
    Returns:
        OCR results with detected Chinese text items.
    """
    try:
        offer_id = request.offer_id
        session_dir = DOWNLOADS_DIR / offer_id
        
        if not session_dir.exists():
            raise HTTPException(status_code=404, detail=f"Session not found: {offer_id}")
        
        # Find the original image
        exts = ("*.webp", "*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
        images = []
        for e in exts:
            images += list(session_dir.glob(e))
        
        if not images:
            raise HTTPException(status_code=404, detail="No image found in session directory")
        
        image_path = str(images[0])
        
        # Create OCR output directory
        ocr_dir = session_dir / "ocr_output"
        ocr_dir.mkdir(exist_ok=True)
        
        # Run OCR
        ocr_data, fed_path = ocr_predict_to_json(image_path, str(ocr_dir))
        
        # Extract Chinese items
        ch_items = get_chinese_items(ocr_data, conf_thresh=0.6)
        
        # Save chinese items to JSON
        ch_items_path = ocr_dir / "chinese_items.json"
        with open(ch_items_path, "w", encoding="utf-8") as f:
            json.dump(ch_items, f, ensure_ascii=False, indent=2)
        
        return OCRResponse(
            offer_id=offer_id,
            message=f"✅ OCR completed. Found {len(ch_items)} Chinese text regions.",
            chinese_count=len(ch_items),
            chinese_items=ch_items,
            ocr_json_path=str(ch_items_path),
            fed_image_path=fed_path
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR error: {str(e)}")


@router.post("/translate", response_model=TranslateResponse)
async def translate_image(request: TranslateRequest):
    """
    Full pipeline: OCR → Translate → Inpaint → Overlay English text.
    
    Args:
        request: Contains the session ID.
        
    Returns:
        Translation results with output image paths.
    """
    try:
        offer_id = request.offer_id
        session_dir = DOWNLOADS_DIR / offer_id
        
        if not session_dir.exists():
            raise HTTPException(status_code=404, detail=f"Session not found: {offer_id}")
        
        # Find the original image
        exts = ("*.webp", "*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
        images = []
        for e in exts:
            images += list(session_dir.glob(e))
        
        if not images:
            raise HTTPException(status_code=404, detail="No image found in session directory")
        
        image_path = str(images[0])
        
        # Create output directory
        output_dir = session_dir / "output"
        output_dir.mkdir(exist_ok=True)
        
        # Step 1: Run OCR
        ocr_data, fed_path = ocr_predict_to_json(image_path, str(output_dir))
        ch_items = get_chinese_items(ocr_data, conf_thresh=0.6)
        
        if not ch_items:
            # No Chinese text found, just copy original
            img_bgr = load_image_to_bgr(image_path)
            output_path = output_dir / "translated.png"
            save_image(img_bgr, str(output_path))
            return TranslateResponse(
                offer_id=offer_id,
                message="✅ No Chinese text found. Original image returned.",
                chinese_count=0,
                output_image_path=str(output_path),
                inpainted_image_path=str(output_path),
                translations=[]
            )
        
        # Step 2: Translate Chinese to English
        en_lines = translate_items_gemini(ch_items)
        for it, en in zip(ch_items, en_lines):
            it["en"] = en
        
        # Save translations
        translations_path = output_dir / "translations.json"
        with open(translations_path, "w", encoding="utf-8") as f:
            json.dump(ch_items, f, ensure_ascii=False, indent=2)
        
        # Step 3: Inpaint (remove Chinese text)
        img_bgr = load_image_to_bgr(image_path)
        H, W = get_image_dimensions(img_bgr)
        mask = create_mask_from_items(ch_items, H, W, pad=6)
        inpainted_bgr = inpaint_image(img_bgr, mask)
        
        inpainted_path = output_dir / "inpainted.png"
        save_image(inpainted_bgr, str(inpainted_path))
        
        # Step 4: Overlay English text
        final_pil = overlay_english_text(inpainted_bgr, ch_items, FONT_PATH)
        
        output_path = output_dir / "translated.png"
        final_pil.convert("RGB").save(str(output_path))
        
        return TranslateResponse(
            offer_id=offer_id,
            message=f"✅ Translation completed. {len(ch_items)} Chinese regions translated.",
            chinese_count=len(ch_items),
            output_image_path=str(output_path),
            inpainted_image_path=str(inpainted_path),
            translations=ch_items
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Translation error: {str(e)}")


# ============================================================================
# Streaming pipeline worker function
# ============================================================================

async def process_single_image_pipeline(
    image_url: str,
    image_index: int,
    images_dir: Path,
    offer_id: str,
    results: List,
    url_mapping: Dict[str, str]
):
    """
    Streaming pipeline for a single image:
    Download → OCR → Translate → Inpaint → Upload
    
    Each stage starts immediately after the previous completes.
    GPU operations use semaphores for sequential execution.
    """
    original_path = None
    try:
        # =====================================================================
        # STAGE 1: Download (I/O - parallel)
        # =====================================================================
        print(f"   [{image_index + 1}] 📥 Downloading...")
        sys.stdout.flush()
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.get(
                image_url,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
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
        
        # Save original image
        original_path = images_dir / f"original_{image_index}{ext}"
        with open(original_path, "wb") as f:
            f.write(response.content)
        
        print(f"   [{image_index + 1}] ✅ Downloaded")
        sys.stdout.flush()
        
        # =====================================================================
        # STAGE 2: OCR (GPU - sequential via semaphore)
        # =====================================================================
        print(f"   [{image_index + 1}] 🔍 Waiting for OCR...")
        sys.stdout.flush()
        
        async with ocr_semaphore:  # Only 1 OCR at a time
            print(f"   [{image_index + 1}] 🔍 Running OCR...")
            sys.stdout.flush()
            
            ocr_dir = images_dir / f"ocr_{image_index}"
            ocr_dir.mkdir(exist_ok=True)
            
            # Run OCR in thread pool
            loop = asyncio.get_running_loop()
            ocr_data, fed_path = await loop.run_in_executor(
                None,
                ocr_predict_to_json,
                str(original_path),
                str(ocr_dir)
            )
            ch_items = get_chinese_items(ocr_data, conf_thresh=0.6)
            
            print(f"   [{image_index + 1}] ✅ OCR done: {len(ch_items)} Chinese regions")
            sys.stdout.flush()
        
        # If no Chinese text, skip translation/inpainting
        if not ch_items:
            print(f"   [{image_index + 1}] ⏭️  No Chinese text, converting...")
            sys.stdout.flush()
            
            from PIL import Image
            import io
            img = Image.open(original_path).convert("RGB")
            img_buffer = io.BytesIO()
            img.save(img_buffer, format="WEBP", lossless=True)
            img_bytes = img_buffer.getvalue()
            
            output_path = images_dir / f"translated_{image_index}.webp"
            with open(output_path, "wb") as f:
                f.write(img_bytes)
            
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
        
        # =====================================================================
        # STAGE 3: Translation (I/O - parallel API calls)
        # =====================================================================
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
        
        # =====================================================================
        # STAGE 4: Inpainting (GPU - sequential via semaphore)
        # =====================================================================
        print(f"   [{image_index + 1}] 🎨 Waiting for inpainting...")
        sys.stdout.flush()
        
        async with inpainting_semaphore:  # Only 1 inpainting at a time
            print(f"   [{image_index + 1}] 🎨 Inpainting...")
            sys.stdout.flush()
            
            loop = asyncio.get_running_loop()
            
            def inpaint_and_overlay():
                img_bgr = load_image_to_bgr(str(original_path))
                H, W = get_image_dimensions(img_bgr)
                mask = create_mask_from_items(ch_items, H, W, pad=6)
                inpainted_bgr = inpaint_image(img_bgr, mask)
                final_pil = overlay_english_text(inpainted_bgr, ch_items, FONT_PATH)
                
                import io
                img_buffer = io.BytesIO()
                final_pil.convert("RGB").save(img_buffer, format="WEBP", lossless=True)
                return img_buffer.getvalue()
            
            img_bytes = await loop.run_in_executor(None, inpaint_and_overlay)
            
            print(f"   [{image_index + 1}] ✅ Inpainted")
            sys.stdout.flush()
        
        # Save locally
        output_path = images_dir / f"translated_{image_index}.webp"
        with open(output_path, "wb") as f:
            f.write(img_bytes)
        
        # =====================================================================
        # STAGE 5: Upload to GCS (I/O - parallel)
        # =====================================================================
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
        
        # Store result
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
        import traceback
        error_msg = f"{type(e).__name__}: {str(e)}"
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
    Batch translate images using streaming pipeline architecture.
    
    Each image flows through all stages immediately:
    Download → OCR → Translate → Inpaint → Upload
    
    No waiting for batch phases. GPU operations use semaphores for
    sequential execution while I/O operations run in parallel.
    
    Args:
        request: Contains HTML content (description) with image tags and optional offer ID.
        
    Returns:
        Translated HTML and details about each image translation.
    """
    try:
        offer_id = request.offer_id or str(uuid.uuid4())
        session_dir = DOWNLOADS_DIR / offer_id
        session_dir.mkdir(exist_ok=True)
        
        # Extract image URLs from HTML
        image_urls = extract_image_urls(request.description)
        
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
        
        # Create images directory
        images_dir = session_dir / "images"
        images_dir.mkdir(exist_ok=True)
        
        print(f"🚀 Streaming pipeline: {len(image_urls)} images")
        print(f"   Each flows: Download → OCR → Translate → Inpaint → Upload")
        print(f"   GPU ops sequential, I/O parallel\n")
        sys.stdout.flush()
        
        # Shared state for results
        results = [None] * len(image_urls)
        url_mapping: Dict[str, str] = {}
        
        # Create pipeline tasks - all images start processing immediately
        pipeline_tasks = [
            process_single_image_pipeline(
                url,
                idx,
                images_dir,
                offer_id,
                results,
                url_mapping
            )
            for idx, url in enumerate(image_urls)
        ]
        
        # Execute all pipelines concurrently
        # Each image downloads and flows through stages immediately
        # GPU stages (OCR, inpainting) are controlled by semaphores
        await asyncio.gather(*pipeline_tasks, return_exceptions=True)
        
        print(f"\n✅ All {len(image_urls)} images processed")
        sys.stdout.flush()
        
        # Replace URLs in HTML
        translated_html = replace_image_urls(request.description, url_mapping)
        
        # Save translated HTML to file
        html_output_path = session_dir / f"{offer_id}.html"
        with open(html_output_path, "w", encoding="utf-8") as f:
            f.write(translated_html)
        print(f"💾 Saved: {html_output_path}")
        sys.stdout.flush()
        
        # Count successes and failures
        successful = sum(1 for r in results if r and r.success)
        failed = len(results) - successful
        
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
        print(f"❌ Batch translation error: {str(e)}")
        print(f"📋 Traceback: {traceback.format_exc()}")
        sys.stdout.flush()
        raise HTTPException(status_code=500, detail=f"Batch translation error: {str(e)}")
