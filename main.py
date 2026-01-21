import os
import re
import uuid
import json
import glob
import httpx
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from typing import Optional, List, Dict, Any
import cv2
import numpy as np
from PIL import Image

# Fix for threading issues with inpainting/OCR
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from paddleocr import PaddleOCR

# --- Initialize PaddleOCR (global instance) ---
ocr = PaddleOCR(
    lang="ch",
    use_doc_unwarping=False,
    use_doc_orientation_classify=False,
    text_det_limit_type="max",
    text_det_limit_side_len=4000,
    text_det_thresh=0.2,
    text_det_box_thresh=0.3,
    text_det_unclip_ratio=1.8,
    text_rec_score_thresh=0.0
)

# Regex for Chinese characters
CJK_RE = re.compile(r"[\u4e00-\u9fff]")

app = FastAPI(
    title="Image Translation API",
    description="OCR Chinese text, translate to English, and overlay on images",
    version="0.1.0"
)

ROOT_DIR = Path(__file__).parent
DOWNLOADS_DIR = ROOT_DIR / "downloads"
OUTPUTS_DIR = ROOT_DIR / "outputs"
DOWNLOADS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

class ImageDownloadRequest(BaseModel):
    image_url: HttpUrl
    session_id: Optional[str] = None

class ImageDownloadResponse(BaseModel):
    session_id: str
    message: str
    image_path: str
    image_size: dict

class OCRRequest(BaseModel):
    session_id: str

class ChineseItem(BaseModel):
    text: str
    conf: float
    poly: Optional[List[List[float]]] = None
    box: Optional[List[float]] = None

class OCRResponse(BaseModel):
    session_id: str
    message: str
    chinese_count: int
    chinese_items: List[Dict[str, Any]]
    ocr_json_path: str
    fed_image_path: str

def load_image_to_bgr(path: str):
    """Load ANY common image format into BGR (OpenCV style)"""
    ext = os.path.splitext(path.lower())[1]
    if ext in (".jpg", ".jpeg", ".webp"):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise RuntimeError(f"OpenCV failed to read image: {path}")
        if img.ndim == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img
    img = Image.open(path).convert("RGB")
    arr = np.array(img)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def ocr_predict_to_json(image_path: str, outdir: str) -> tuple:
    """Run OCR on image and save results to JSON."""
    os.makedirs(outdir, exist_ok=True)
    
    img_bgr = load_image_to_bgr(image_path)
    
    # Save the exact bitmap fed to OCR
    fed_path = os.path.join(outdir, "fed_to_ocr.png")
    cv2.imwrite(fed_path, img_bgr)
    
    # Run OCR
    outputs = ocr.predict(fed_path)
    for res in outputs:
        res.save_to_json(outdir)
    
    # Find the latest JSON file
    jfiles = sorted(glob.glob(os.path.join(outdir, "*.json")), key=os.path.getmtime)
    if not jfiles:
        raise RuntimeError("No JSON produced by OCR.")
    
    with open(jfiles[-1], "r", encoding="utf-8") as f:
        data = json.load(f)
    
    return data, fed_path


def get_chinese_items(ocr_json: dict, conf_thresh: float = None) -> List[Dict]:
    """Extract Chinese text items from OCR results."""
    if not ocr_json:
        return []
    
    rec_texts = ocr_json.get("rec_texts", []) or []
    rec_scores = ocr_json.get("rec_scores", []) or []
    rec_polys = ocr_json.get("rec_polys", None)
    rec_boxes = ocr_json.get("rec_boxes", None)
    
    found = []
    for i, txt in enumerate(rec_texts):
        if not CJK_RE.search(txt or ""):
            continue
        
        score = float(rec_scores[i]) if i < len(rec_scores) else 0.0
        if conf_thresh is not None and score < conf_thresh:
            continue
        
        item = {"text": txt, "conf": score}
        if rec_polys is not None and i < len(rec_polys):
            item["poly"] = rec_polys[i]
        if rec_boxes is not None and i < len(rec_boxes):
            item["box"] = rec_boxes[i]
        found.append(item)
    
    return found

@app.get("/")
async def root():
    return {
        "message": "Image Translation API",
        "endpoints": {
            "/download": "POST - Download image from URL",
            "/ocr": "POST - Run OCR on downloaded image",
            "/health": "GET - Health check"
        }
    }

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/download", response_model=ImageDownloadResponse)
async def download_image(request: ImageDownloadRequest):
    """Milestone 1: Download image from URL and save it locally."""
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
        height, width = img_bgr.shape[:2]
        
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


@app.post("/ocr", response_model=OCRResponse)
async def run_ocr(request: OCRRequest):
    """Milestone 2: Run OCR on downloaded image and extract Chinese text."""
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
