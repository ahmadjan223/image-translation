"""
API routes for the Image Translation API.

Performance optimizations (GPU-optimized):
- Phase 1: Parallel image downloads (I/O)
- Phase 2: Sequential OCR processing (GPU constraint)
- Phase 3: Parallel translation API calls (I/O)
- Phase 4: Sequential inpainting (GPU constraint)
- Phase 5: Parallel GCS uploads (I/O)
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
# Phase-based helper functions for GPU-optimized processing
# ============================================================================

async def download_image_helper(
    image_url: str,
    image_dir: Path,
    image_index: int
) -> Tuple[str, int, str]:
    """
    Phase 1: Download a single image (I/O operation).
    
    Returns:
        Tuple of (original_url, image_index, local_path)
    """
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
    original_path = image_dir / f"original_{image_index}{ext}"
    with open(original_path, "wb") as f:
        f.write(response.content)
    
    print(f"   📥 Downloaded image {image_index + 1}: {Path(original_path).name}")
    sys.stdout.flush()
    return image_url, image_index, str(original_path)


def process_ocr_phase(original_path: str, ocr_dir: str) -> Tuple[list, str]:
    """
    Phase 2: Run OCR on a single image (GPU operation - must be sequential).
    
    Returns:
        Tuple of (ch_items, fed_path)
    """
    print(f"   🔍 Running OCR: {Path(original_path).name}")
    sys.stdout.flush()
    
    ocr_data, fed_path = ocr_predict_to_json(original_path, ocr_dir)
    ch_items = get_chinese_items(ocr_data, conf_thresh=0.6)
    
    print(f"   ✅ OCR complete: Found {len(ch_items)} Chinese text regions")
    sys.stdout.flush()
    return ch_items, fed_path


async def process_translation_phase(ch_items: list, image_index: int) -> list:
    """
    Phase 3: Translate Chinese text to English (I/O operation - API call).
    
    Returns:
        List of ch_items with 'en' field added
    """
    if not ch_items:
        return ch_items
    
    print(f"   🌐 Translating {len(ch_items)} text regions for image {image_index + 1}...")
    sys.stdout.flush()
    
    # Run translation in the event loop (it's async-friendly)
    en_lines = translate_items_gemini(ch_items)
    for it, en in zip(ch_items, en_lines):
        it["en"] = en
    
    print(f"   ✅ Translation complete for image {image_index + 1}")
    sys.stdout.flush()
    return ch_items


def process_inpainting_phase(
    original_path: str,
    ch_items: list
) -> bytes:
    """
    Phase 4: Inpaint and overlay text (GPU operation - must be sequential).
    
    Returns:
        Translated image as WebP bytes
    """
    print(f"   🎨 Inpainting and overlaying: {Path(original_path).name}")
    sys.stdout.flush()
    
    # Inpaint (remove Chinese text)
    img_bgr = load_image_to_bgr(original_path)
    H, W = get_image_dimensions(img_bgr)
    mask = create_mask_from_items(ch_items, H, W, pad=6)
    inpainted_bgr = inpaint_image(img_bgr, mask)
    
    # Overlay English text
    final_pil = overlay_english_text(inpainted_bgr, ch_items, FONT_PATH)
    
    # Convert to WebP bytes
    import io
    img_buffer = io.BytesIO()
    final_pil.convert("RGB").save(img_buffer, format="WEBP", lossless=True)
    
    print(f"   ✅ Inpainting complete: {Path(original_path).name}")
    sys.stdout.flush()
    return img_buffer.getvalue()


async def upload_to_gcs_helper(
    image_bytes: bytes,
    offer_id: str,
    image_index: int
) -> str:
    """
    Phase 5: Upload image to GCS (I/O operation).
    
    Returns:
        Public URL of the uploaded image
    """
    if not gcp_storage.is_available():
        return None
    
    public_url = gcp_storage.upload_from_bytes(
        image_bytes,
        folder_name=f"translated/{offer_id}",
        image_name=f"image_{image_index}"
    )
    
    print(f"   ☁️  Uploaded to GCS: image_{image_index}")
    sys.stdout.flush()
    return public_url


@router.post("/translate-batch", response_model=BatchTranslateResponse)
async def translate_batch(request: BatchTranslateRequest):
    """
    Batch translate images in HTML content using GPU-optimized phase-based processing.
    
    Architecture:
    1. Phase 1: Download all images in parallel (I/O)
    2. Phase 2: Process OCR sequentially (GPU constraint)
    3. Phase 3: Translate all texts in parallel (I/O - API calls)
    4. Phase 4: Inpaint all images sequentially (GPU constraint)
    5. Phase 5: Upload all to GCS in parallel (I/O)
    
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
        
        print(f"📷 Processing {len(image_urls)} images with GPU-optimized pipeline...")
        sys.stdout.flush()
        
        # ========================================================================
        # PHASE 1: Download all images in parallel (I/O)
        # ========================================================================
        print(f"\n📥 Phase 1: Downloading {len(image_urls)} images in parallel...")
        sys.stdout.flush()
        
        download_tasks = [
            download_image_helper(url, images_dir, idx)
            for idx, url in enumerate(image_urls)
        ]
        download_results = await asyncio.gather(*download_tasks, return_exceptions=True)
        
        # Build mapping of successful downloads
        downloaded_images = []
        results = []
        
        for idx, result in enumerate(download_results):
            if isinstance(result, Exception):
                error_msg = f"{type(result).__name__}: {str(result)}"
                print(f"❌ Download failed for image {idx + 1}: {error_msg}")
                sys.stdout.flush()
                results.append(ImageTranslationResult(
                    original_url=image_urls[idx],
                    local_path="",
                    public_url=None,
                    chinese_count=0,
                    success=False,
                    error=error_msg[:200]
                ))
            else:
                original_url, image_index, local_path = result
                downloaded_images.append((original_url, image_index, local_path))
                # Placeholder - will update after processing
                results.append(None)
        
        print(f"✅ Phase 1 complete: {len(downloaded_images)}/{len(image_urls)} images downloaded")
        sys.stdout.flush()
        
        if not downloaded_images:
            return BatchTranslateResponse(
                offer_id=offer_id,
                message="❌ All image downloads failed.",
                total_images=len(results),
                successful_translations=0,
                failed_translations=len(results),
                translated_html=request.description,
                image_results=results
            )
        
        # ========================================================================
        # PHASE 2: Process OCR sequentially (GPU constraint)
        # ========================================================================
        print(f"\n🔍 Phase 2: Running OCR sequentially on {len(downloaded_images)} images...")
        sys.stdout.flush()
        
        ocr_results = []
        for original_url, image_index, local_path in downloaded_images:
            try:
                ocr_dir = images_dir / f"ocr_{image_index}"
                ocr_dir.mkdir(exist_ok=True)
                ch_items, fed_path = process_ocr_phase(local_path, str(ocr_dir))
                ocr_results.append((original_url, image_index, local_path, ch_items))
            except Exception as e:
                error_msg = f"{type(e).__name__}: {str(e)}"
                print(f"❌ OCR failed for image {image_index + 1}: {error_msg}")
                sys.stdout.flush()
                results[image_index] = ImageTranslationResult(
                    original_url=original_url,
                    local_path=local_path,
                    public_url=None,
                    chinese_count=0,
                    success=False,
                    error=error_msg[:200]
                )
        
        print(f"✅ Phase 2 complete: OCR processed {len(ocr_results)} images")
        sys.stdout.flush()
        
        # ========================================================================
        # PHASE 3: Translate all texts in parallel (I/O - API calls)
        # ========================================================================
        print(f"\n🌐 Phase 3: Translating texts in parallel...")
        sys.stdout.flush()
        
        translation_tasks = [
            process_translation_phase(ch_items, image_index)
            for original_url, image_index, local_path, ch_items in ocr_results
        ]
        translation_results = await asyncio.gather(*translation_tasks, return_exceptions=True)
        
        # Update ocr_results with translations
        translated_images = []
        for idx, trans_result in enumerate(translation_results):
            original_url, image_index, local_path, ch_items = ocr_results[idx]
            if isinstance(trans_result, Exception):
                error_msg = f"{type(trans_result).__name__}: {str(trans_result)}"
                print(f"❌ Translation failed for image {image_index + 1}: {error_msg}")
                sys.stdout.flush()
                results[image_index] = ImageTranslationResult(
                    original_url=original_url,
                    local_path=local_path,
                    public_url=None,
                    chinese_count=len(ch_items),
                    success=False,
                    error=error_msg[:200]
                )
            else:
                translated_ch_items = trans_result
                translated_images.append((original_url, image_index, local_path, translated_ch_items))
        
        print(f"✅ Phase 3 complete: Translated {len(translated_images)} images")
        sys.stdout.flush()
        
        # ========================================================================
        # PHASE 4: Inpaint sequentially (GPU constraint)
        # ========================================================================
        print(f"\n🎨 Phase 4: Inpainting and overlaying sequentially...")
        sys.stdout.flush()
        
        inpainted_images = []
        for original_url, image_index, local_path, ch_items in translated_images:
            try:
                if not ch_items:
                    # No Chinese text - convert original to WebP
                    from PIL import Image
                    import io
                    img = Image.open(local_path).convert("RGB")
                    img_buffer = io.BytesIO()
                    img.save(img_buffer, format="WEBP", lossless=True)
                    img_bytes = img_buffer.getvalue()
                else:
                    # Inpaint and overlay
                    img_bytes = process_inpainting_phase(local_path, ch_items)
                
                # Save locally
                output_path = images_dir / f"translated_{image_index}.webp"
                with open(output_path, "wb") as f:
                    f.write(img_bytes)
                
                inpainted_images.append((original_url, image_index, str(output_path), img_bytes, len(ch_items)))
            except Exception as e:
                error_msg = f"{type(e).__name__}: {str(e)}"
                print(f"❌ Inpainting failed for image {image_index + 1}: {error_msg}")
                sys.stdout.flush()
                results[image_index] = ImageTranslationResult(
                    original_url=original_url,
                    local_path=local_path,
                    public_url=None,
                    chinese_count=len(ch_items),
                    success=False,
                    error=error_msg[:200]
                )
        
        print(f"✅ Phase 4 complete: Inpainted {len(inpainted_images)} images")
        sys.stdout.flush()
        
        # ========================================================================
        # PHASE 5: Upload to GCS in parallel (I/O)
        # ========================================================================
        print(f"\n☁️  Phase 5: Uploading to GCS in parallel...")
        sys.stdout.flush()
        
        upload_tasks = [
            upload_to_gcs_helper(img_bytes, offer_id, image_index)
            for original_url, image_index, output_path, img_bytes, chinese_count in inpainted_images
        ]
        upload_results = await asyncio.gather(*upload_tasks, return_exceptions=True)
        
        # Build final results
        url_mapping: Dict[str, str] = {}
        
        for idx, upload_result in enumerate(upload_results):
            original_url, image_index, output_path, img_bytes, chinese_count = inpainted_images[idx]
            
            if isinstance(upload_result, Exception):
                public_url = None
                print(f"⚠️  Upload failed for image {image_index + 1}, using local path")
                sys.stdout.flush()
            else:
                public_url = upload_result
            
            results[image_index] = ImageTranslationResult(
                original_url=original_url,
                local_path=output_path,
                public_url=public_url,
                chinese_count=chinese_count,
                success=True,
                error=None
            )
            
            # Use public URL if available, otherwise local path
            if public_url:
                url_mapping[original_url] = public_url
            else:
                url_mapping[original_url] = output_path
        
        print(f"✅ Phase 5 complete: Uploaded {len([r for r in upload_results if not isinstance(r, Exception)])} images")
        sys.stdout.flush()
        
        # Replace URLs in HTML
        translated_html = replace_image_urls(request.description, url_mapping)
        
        # Count successes and failures
        successful = sum(1 for r in results if r and r.success)
        failed = len(results) - successful
        
        print(f"\n🎉 All phases complete: {successful}/{len(results)} images successfully translated")
        sys.stdout.flush()
        
        return BatchTranslateResponse(
            offer_id=offer_id,
            message=f"✅ Batch translation completed. {successful}/{len(results)} images translated.",
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
