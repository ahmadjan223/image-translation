"""
API routes for the Image Translation API.
"""
import uuid
import json
from pathlib import Path

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
    TranslateResponse
)
from utils.image import load_image_to_bgr, save_image, get_image_dimensions
from services.ocr import ocr_predict_to_json, get_chinese_items
from services.translation import translate_items_gemini
from services.inpainting import create_mask_from_items, inpaint_image
from services.text_overlay import overlay_english_text

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
