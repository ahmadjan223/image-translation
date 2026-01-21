"""
API routes for the Image Translation API.
"""
import uuid
import json
import asyncio
from pathlib import Path
from typing import Dict, Tuple

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
        session_id = request.session_id or str(uuid.uuid4())
        session_dir = DOWNLOADS_DIR / session_id
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
            session_id=session_id,
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
        session_id = request.session_id
        session_dir = DOWNLOADS_DIR / session_id
        
        if not session_dir.exists():
            raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")
        
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
        ch_items = get_chinese_items(ocr_data, conf_thresh=None)
        
        # Save chinese items to JSON
        ch_items_path = ocr_dir / "chinese_items.json"
        with open(ch_items_path, "w", encoding="utf-8") as f:
            json.dump(ch_items, f, ensure_ascii=False, indent=2)
        
        return OCRResponse(
            session_id=session_id,
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
        session_id = request.session_id
        session_dir = DOWNLOADS_DIR / session_id
        
        if not session_dir.exists():
            raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")
        
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
        ch_items = get_chinese_items(ocr_data, conf_thresh=None)
        
        if not ch_items:
            # No Chinese text found, just copy original
            img_bgr = load_image_to_bgr(image_path)
            output_path = output_dir / "translated.png"
            save_image(img_bgr, str(output_path))
            return TranslateResponse(
                session_id=session_id,
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
            session_id=session_id,
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


async def download_and_translate_image(
    image_url: str,
    image_dir: Path,
    image_index: int,
    session_id: str
) -> Tuple[str, ImageTranslationResult]:
    """
    Download a single image, translate it, upload to GCS, and return the result.
    
    Args:
        image_url: URL of the image to download.
        image_dir: Directory to save the image.
        image_index: Index of the image for naming.
        session_id: Session ID for organizing uploads in GCS.
        
    Returns:
        Tuple of (original_url, ImageTranslationResult).
    """
    try:
        # Download image
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
        
        # Run OCR
        ocr_dir = image_dir / f"ocr_{image_index}"
        ocr_dir.mkdir(exist_ok=True)
        ocr_data, fed_path = ocr_predict_to_json(str(original_path), str(ocr_dir))
        ch_items = get_chinese_items(ocr_data, conf_thresh=None)
        
        if not ch_items:
            # No Chinese text found, upload original to GCS
            public_url = None
            if gcp_storage.is_available():
                with open(original_path, "rb") as f:
                    image_bytes = f.read()
                public_url = gcp_storage.upload_from_bytes(
                    image_bytes,
                    folder_name=f"translated/{session_id}",
                    image_name=f"image_{image_index}"
                )
            
            return image_url, ImageTranslationResult(
                original_url=image_url,
                local_path=str(original_path),
                public_url=public_url,
                chinese_count=0,
                success=True,
                error=None
            )
        
        # Translate Chinese to English
        en_lines = translate_items_gemini(ch_items)
        for it, en in zip(ch_items, en_lines):
            it["en"] = en
        
        # Inpaint (remove Chinese text)
        img_bgr = load_image_to_bgr(str(original_path))
        H, W = get_image_dimensions(img_bgr)
        mask = create_mask_from_items(ch_items, H, W, pad=6)
        inpainted_bgr = inpaint_image(img_bgr, mask)
        
        # Overlay English text
        final_pil = overlay_english_text(inpainted_bgr, ch_items, FONT_PATH)
        
        # Save translated image locally
        output_path = image_dir / f"translated_{image_index}.png"
        final_pil.convert("RGB").save(str(output_path))
        
        # Upload to GCS and get public URL
        public_url = None
        if gcp_storage.is_available():
            import io
            img_buffer = io.BytesIO()
            final_pil.convert("RGB").save(img_buffer, format="PNG")
            img_bytes = img_buffer.getvalue()
            public_url = gcp_storage.upload_from_bytes(
                img_bytes,
                folder_name=f"translated/{session_id}",
                image_name=f"image_{image_index}"
            )
            print(f"   ☁️  Uploaded to GCS: {public_url}")
        
        return image_url, ImageTranslationResult(
            original_url=image_url,
            local_path=str(output_path),
            public_url=public_url,
            chinese_count=len(ch_items),
            success=True,
            error=None
        )
    
    except Exception as e:
        return image_url, ImageTranslationResult(
            original_url=image_url,
            local_path="",
            public_url=None,
            chinese_count=0,
            success=False,
            error=str(e)[:200]
        )


@router.post("/translate-batch", response_model=BatchTranslateResponse)
async def translate_batch(request: BatchTranslateRequest):
    """
    Batch translate images in HTML content.
    
    Extracts all image URLs from the HTML, downloads and translates each image,
    then returns the HTML with image src replaced with local translated paths.
    
    Args:
        request: Contains HTML content with image tags and optional session ID.
        
    Returns:
        Translated HTML and details about each image translation.
    """
    try:
        session_id = request.session_id or str(uuid.uuid4())
        session_dir = DOWNLOADS_DIR / session_id
        session_dir.mkdir(exist_ok=True)
        
        # Extract image URLs from HTML


        image_urls = extract_image_urls(request.html_content)
        
        if not image_urls:
            return BatchTranslateResponse(
                session_id=session_id,
                message="⚠️ No images found in HTML content.",
                total_images=0,
                successful_translations=0,
                failed_translations=0,
                translated_html=request.html_content,
                image_results=[]
            )
        
        # Create images directory
        images_dir = session_dir / "images"
        images_dir.mkdir(exist_ok=True)
        
        # Process images sequentially (to avoid overwhelming OCR/translation services)
        results = []
        url_mapping: Dict[str, str] = {}
        
        for idx, url in enumerate(image_urls):
            print(f"📷 Processing image {idx + 1}/{len(image_urls)}: {url[:80]}...")
            original_url, result = await download_and_translate_image(url, images_dir, idx, session_id)
            results.append(result)
            
            # Use public URL if available, otherwise fall back to local path
            if result.success:
                if result.public_url:
                    url_mapping[original_url] = result.public_url
                elif result.local_path:
                    url_mapping[original_url] = result.local_path
        
        # Replace URLs in HTML with GCS public URLs (or local paths as fallback)
        translated_html = replace_image_urls(request.html_content, url_mapping)
        
        # Count successes and failures
        successful = sum(1 for r in results if r.success)
        failed = len(results) - successful
        
        return BatchTranslateResponse(
            session_id=session_id,
            message=f"✅ Batch translation completed. {successful}/{len(results)} images translated.",
            total_images=len(results),
            successful_translations=successful,
            failed_translations=failed,
            translated_html=translated_html,
            image_results=results
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch translation error: {str(e)}")
