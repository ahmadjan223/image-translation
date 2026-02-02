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
from pathlib import Path
from typing import Dict, List

import httpx
import cv2
import numpy as np
from fastapi import APIRouter, HTTPException
from paddleocr import PaddleOCR
from simple_lama_inpainting import SimpleLama

from config import DOWNLOADS_DIR, FONT_PATH, OCR_CONFIG
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
from services.ocr import get_chinese_items
from services.translation import translate_items_gemini
from services.inpainting import create_mask_from_items
from services.text_overlay import overlay_english_text
from gcp_storage import gcp_storage

router = APIRouter()

# ============================================================================
# Pre-initialize models at module load - SIMPLE AND CLEAR
# ============================================================================

print("🔧 Initializing models at startup...")

# Create OCR instance once - will be passed to all workers
OCR_INSTANCE = PaddleOCR(**OCR_CONFIG)
print("✅ OCR model loaded")

# Create SimpleLama instance once - will be passed to all workers
LAMA_INSTANCE = SimpleLama()
print("✅ SimpleLama model loaded")

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
    """Download image from URL and save it locally."""
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


def inpaint_with_lama(img_bgr, mask, lama: SimpleLama):
    """
    Inpaint image using provided SimpleLama instance.
    """
    try:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        inpaint_pil = lama(img_rgb, mask)
        return cv2.cvtColor(np.array(inpaint_pil), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"⚠️ SimpleLaMa failed, falling back to OpenCV: {str(e)[:100]}")
        return cv2.inpaint(img_bgr, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)


@router.post("/ocr", response_model=OCRResponse)
async def run_ocr(request: OCRRequest):
    """Run OCR on downloaded image and extract Chinese text."""
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
        
        # Run OCR using global instance
        ocr_data, fed_path = run_ocr_on_image(image_path, str(ocr_dir), OCR_INSTANCE)
        
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
    """Full pipeline: OCR → Translate → Inpaint → Overlay English text."""
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
        
        # Step 1: Run OCR using global instance
        ocr_data, fed_path = run_ocr_on_image(image_path, str(output_dir), OCR_INSTANCE)
        ch_items = get_chinese_items(ocr_data, conf_thresh=0.6)
        
        if not ch_items:
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
        
        # Step 2: Translate
        en_lines = translate_items_gemini(ch_items)
        for it, en in zip(ch_items, en_lines):
            it["en"] = en
        
        # Save translations
        translations_path = output_dir / "translations.json"
        with open(translations_path, "w", encoding="utf-8") as f:
            json.dump(ch_items, f, ensure_ascii=False, indent=2)
        
        # Step 3: Inpaint using global instance
        img_bgr = load_image_to_bgr(image_path)
        H, W = get_image_dimensions(img_bgr)
        mask = create_mask_from_items(ch_items, H, W, pad=6)
        inpainted_bgr = inpaint_with_lama(img_bgr, mask, LAMA_INSTANCE)
        
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
# Streaming pipeline worker - receives model instances as parameters
# ============================================================================

async def process_single_image_pipeline(
    image_url: str,
    image_index: int,
    images_dir: Path,
    offer_id: str,
    results: List,
    url_mapping: Dict[str, str],
    ocr: PaddleOCR,          # ← Receive OCR instance
    lama: SimpleLama         # ← Receive SimpleLama instance
):
    """
    Process single image through the pipeline.
    
    Receives model instances as parameters - clean and explicit.
    No copies created, just passing references.
    """
    original_path = None
    try:
        # Stage 1: Download
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
            ch_items = get_chinese_items(ocr_data, conf_thresh=0.6)
            
            print(f"   [{image_index + 1}] ✅ OCR done: {len(ch_items)} Chinese regions")
            sys.stdout.flush()
        
        # If no Chinese text, skip translation/inpainting
        if not ch_items:
            print(f"   [{image_index + 1}] ⭐️ No Chinese text, converting...")
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
                mask = create_mask_from_items(ch_items, H, W, pad=6)
                inpainted_bgr = inpaint_with_lama(img_bgr, mask, lama)  # ← Pass lama instance
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