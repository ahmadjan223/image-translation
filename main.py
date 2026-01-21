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
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# Fix for threading issues with inpainting/OCR
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from paddleocr import PaddleOCR
from google import genai
from simple_lama_inpainting import SimpleLama

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

# --- Initialize SimpleLama for inpainting ---
simple_lama = SimpleLama()

# --- Initialize Gemini client ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyAe33GrwIicD5N4JIwxYSO6Nb7b35s2fH4")
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# Regex for Chinese characters
CJK_RE = re.compile(r"[\u4e00-\u9fff]")

# Translation system prompt
TRANSLATION_SYSTEM_PROMPT = """
Translate Chinese OCR lines to English for product images.

Rules:
- ONE translation per line (no options, no numbering, no explanations).
- Keep it SHORT to fit the original box: en length <= max_chars.
- Keep repeated terms consistent.
- Use simple, short size synonyms to occupy less space for product images.

Output JSON ONLY:
[{ "i": <index>, "en": "<translation>" }]
""".strip()

# Font path (Linux default)
FONT_PATH = "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"

# Text overlay parameters
PAD_IN = 2
MIN_BOX_W = 20
MIN_BOX_H = 14
MIN_FONT_SIZE = 12
MAX_FONT_SIZE = 120
FONT_SIZE_STEP = 2
SPLIT_THRESHOLD = 12
MAX_LINES_FALLBACK = 2
LINE_SPACING = 1.025
SHADOW_BLUR = 1
SHADOW_OFFSET = (1, 1)

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

class TranslateRequest(BaseModel):
    session_id: str

class TranslateResponse(BaseModel):
    session_id: str
    message: str
    chinese_count: int
    output_image_path: str
    inpainted_image_path: str
    translations: List[Dict[str, Any]]

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


# ==================== TRANSLATION FUNCTIONS ====================

def translate_items_gemini(ch_items: List[Dict], model: str = "models/gemini-flash-lite-latest") -> List[str]:
    """Translate Chinese items to English using Gemini."""
    if not ch_items:
        return []
    
    items = []
    for i, it in enumerate(ch_items):
        cn = (it.get("text") or "").strip()
        max_chars = max(6, int(len(cn) * 1.35))
        items.append({"i": i, "cn": cn, "max_chars": max_chars})
    
    payload = {"items": items}
    
    resp = gemini_client.models.generate_content(
        model=model,
        contents=json.dumps(payload, ensure_ascii=False),
        config={
            "system_instruction": TRANSLATION_SYSTEM_PROMPT,
            "temperature": 0.2,
            "response_mime_type": "application/json",
        },
    )
    
    arr = json.loads((resp.text or "").strip())
    mp = {int(o["i"]): (o.get("en") or "").strip() for o in arr}
    return [mp.get(i, "") for i in range(len(items))]


# ==================== INPAINTING FUNCTIONS ====================

def create_mask_from_items(ch_items: List[Dict], H: int, W: int, pad: int = 6) -> np.ndarray:
    """Create inpainting mask from Chinese text detections."""
    mask = np.zeros((H, W), dtype=np.uint8)
    
    for it in ch_items:
        if it.get("poly") is not None:
            pts = np.array(it["poly"], dtype=np.int32).reshape(-1, 2)
            cv2.fillPoly(mask, [pts], 255)
        elif it.get("box") is not None:
            x1, y1, x2, y2 = map(int, it["box"])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W - 1, x2), min(H - 1, y2)
            if x2 > x1 and y2 > y1:
                cv2.rectangle(mask, (x1, y1), (x2, y2), 255, thickness=-1)
    
    # Expand mask to cover strokes better
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*pad + 1, 2*pad + 1))
    mask = cv2.dilate(mask, k, iterations=1)
    
    return mask


def inpaint_image(img_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Inpaint image using SimpleLama (fallback to OpenCV)."""
    try:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        inpaint_pil = simple_lama(img_rgb, mask)
        return cv2.cvtColor(np.array(inpaint_pil), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"⚠️ SimpleLaMa failed, falling back to OpenCV: {str(e)[:100]}")
        return cv2.inpaint(img_bgr, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)


# ==================== TEXT OVERLAY FUNCTIONS ====================

def clamp_box(x1, y1, x2, y2, W, H):
    x1 = int(max(0, min(W - 1, x1)))
    y1 = int(max(0, min(H - 1, y1)))
    x2 = int(max(0, min(W - 1, x2)))
    y2 = int(max(0, min(H - 1, y2)))
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1
    return x1, y1, x2, y2


def text_size(draw, text, font):
    bb = draw.textbbox((0, 0), text, font=font)
    return bb[2] - bb[0], bb[3] - bb[1]


def wrap_text(draw, text, font, max_w, max_lines=2):
    """Greedy wrap by spaces; fall back to char wrap."""
    text = (text or "").strip()
    if not text:
        return []
    if text_size(draw, text, font)[0] <= max_w:
        return [text]
    
    words = text.split()
    if len(words) == 1:
        lines, cur = [], ""
        for ch in text:
            test = cur + ch
            if text_size(draw, test, font)[0] <= max_w or not cur:
                cur = test
            else:
                lines.append(cur)
                cur = ch
                if len(lines) >= max_lines:
                    break
        if len(lines) < max_lines and cur:
            lines.append(cur)
        return lines[:max_lines]
    
    lines, cur = [], ""
    for w in words:
        test = (cur + " " + w).strip()
        if text_size(draw, test, font)[0] <= max_w or not cur:
            cur = test
        else:
            lines.append(cur)
            cur = w
            if len(lines) >= max_lines:
                break
    if len(lines) < max_lines and cur:
        lines.append(cur)
    return lines[:max_lines]


def truncate_line_to_width(draw, s, font, max_w):
    s = (s or "").strip()
    if not s:
        return ""
    if text_size(draw, s, font)[0] <= max_w:
        return s
    ell = "…"
    if text_size(draw, ell, font)[0] > max_w:
        return ""
    while s and text_size(draw, s + ell, font)[0] > max_w:
        s = s[:-1]
    return (s + ell) if s else ell


def fit_font_single_line(draw, text, target_w, target_h, font_path,
                         min_size=10, max_size=200, step=2):
    text = (text or "").strip()
    if not text:
        return None
    for sz in range(max_size, min_size - 1, -step):
        f = ImageFont.truetype(font_path, sz)
        tw, th = text_size(draw, text, f)
        if tw <= target_w and th <= target_h:
            return f, [text], sz
    return None


def fit_font_multi_line(draw, text, target_w, target_h, font_path,
                        max_lines=2, min_size=10, max_size=200, step=2, line_spacing=1.1):
    text = (text or "").strip()
    if not text:
        return None
    best = None
    for sz in range(max_size, min_size - 1, -step):
        f = ImageFont.truetype(font_path, sz)
        lines = wrap_text(draw, text, f, target_w, max_lines=max_lines)
        if not lines:
            continue
        lines = [truncate_line_to_width(draw, li, f, target_w) for li in lines]
        line_h = text_size(draw, "Ag", f)[1]
        total_h = int(line_h * len(lines) * line_spacing)
        max_line_w = max(text_size(draw, li, f)[0] for li in lines) if lines else 0
        if max_line_w <= target_w and total_h <= target_h:
            return f, lines, sz
        best = (f, lines, sz)
    return best


def sample_bg_luma(bgr_img, x1, y1, x2, y2, pad=6):
    H, W = bgr_img.shape[:2]
    x1p, y1p = max(0, x1 - pad), max(0, y1 - pad)
    x2p, y2p = min(W, x2 + pad), min(H, y2 + pad)
    roi = bgr_img[y1p:y2p, x1p:x2p]
    if roi.size == 0:
        return 255
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    return float(np.median(gray))


def draw_text_with_shadow(pil_rgba, xy, lines, font, fill_rgba, shadow_rgba,
                          shadow_blur=2, shadow_offset=(1, 1), align="center", line_spacing=1.1):
    base = pil_rgba
    x, y = xy
    tmp = Image.new("RGBA", base.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(tmp)
    
    line_h = d.textbbox((0, 0), "Ag", font=font)[3]
    widths = [d.textbbox((0, 0), li, font=font)[2] for li in lines]
    block_w = max(widths) if widths else 0
    step_y = int(line_h * line_spacing)
    
    yy = y
    for li, w in zip(lines, widths):
        if align == "center":
            xx = x + (block_w - w) // 2
        elif align == "left":
            xx = x
        else:
            xx = x + (block_w - w)
        d.text((xx + shadow_offset[0], yy + shadow_offset[1]), li, font=font, fill=shadow_rgba)
        yy += step_y
    
    tmp = tmp.filter(ImageFilter.GaussianBlur(radius=shadow_blur))
    
    d = ImageDraw.Draw(tmp)
    yy = y
    for li, w in zip(lines, widths):
        if align == "center":
            xx = x + (block_w - w) // 2
        elif align == "left":
            xx = x
        else:
            xx = x + (block_w - w)
        d.text((xx, yy), li, font=font, fill=fill_rgba)
        yy += step_y
    
    block_h = int(line_h * len(lines) * line_spacing)
    return Image.alpha_composite(base, tmp), (block_w, block_h)


def overlay_english_text(inpainted_bgr: np.ndarray, ch_items: List[Dict], font_path: str) -> Image.Image:
    """Overlay translated English text on inpainted image."""
    H, W = inpainted_bgr.shape[:2]
    rgb = cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb).convert("RGBA")
    draw = ImageDraw.Draw(pil_img)
    
    for it in ch_items:
        en = (it.get("en") or "").strip()
        if not en or it.get("box") is None:
            continue
        
        x1, y1, x2, y2 = map(int, it["box"])
        x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, W, H)
        bw, bh = (x2 - x1), (y2 - y1)
        
        if bw < MIN_BOX_W or bh < MIN_BOX_H:
            continue
        
        target_w = max(1, bw - 2 * PAD_IN)
        target_h = max(1, bh - 2 * PAD_IN)
        
        result = fit_font_single_line(
            draw, en, target_w, target_h, font_path,
            min_size=MIN_FONT_SIZE, max_size=MAX_FONT_SIZE, step=FONT_SIZE_STEP
        )
        
        if result:
            font, lines, font_sz = result
            if font_sz <= SPLIT_THRESHOLD:
                multi = fit_font_multi_line(
                    draw, en, target_w, target_h, font_path,
                    max_lines=MAX_LINES_FALLBACK,
                    min_size=MIN_FONT_SIZE, max_size=MAX_FONT_SIZE, step=FONT_SIZE_STEP,
                    line_spacing=LINE_SPACING
                )
                if multi and multi[2] > font_sz:
                    font, lines, font_sz = multi
        else:
            multi = fit_font_multi_line(
                draw, en, target_w, target_h, font_path,
                max_lines=MAX_LINES_FALLBACK,
                min_size=MIN_FONT_SIZE, max_size=MAX_FONT_SIZE, step=FONT_SIZE_STEP,
                line_spacing=LINE_SPACING
            )
            if multi:
                font, lines, font_sz = multi
            else:
                continue
        
        if not lines:
            continue
        
        luma = sample_bg_luma(inpainted_bgr, x1, y1, x2, y2, pad=8)
        if luma > 160:
            fill = (15, 15, 15, 255)
            shadow = (255, 255, 255, 140)
        else:
            fill = (245, 245, 245, 255)
            shadow = (0, 0, 0, 160)
        
        widths = [draw.textbbox((0, 0), li, font=font)[2] for li in lines]
        line_h = draw.textbbox((0, 0), "Ag", font=font)[3]
        block_w = max(widths) if widths else 0
        block_h = int(line_h * len(lines) * LINE_SPACING)
        
        tx = x1 + (bw - block_w) // 2
        ty = y1 + (bh - block_h) // 2
        
        pil_img, _ = draw_text_with_shadow(
            pil_img, (tx, ty), lines, font,
            fill_rgba=fill, shadow_rgba=shadow,
            shadow_blur=SHADOW_BLUR, shadow_offset=SHADOW_OFFSET,
            align="center", line_spacing=LINE_SPACING,
        )
    
    return pil_img

@app.get("/")
async def root():
    return {
        "message": "Image Translation API",
        "endpoints": {
            "/download": "POST - Download image from URL",
            "/ocr": "POST - Run OCR on downloaded image",
            "/translate": "POST - Full pipeline: OCR → Translate → Inpaint → Overlay",
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


@app.post("/translate", response_model=TranslateResponse)
async def translate_image(request: TranslateRequest):
    """Full pipeline: OCR → Translate → Inpaint → Overlay English text."""
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
            cv2.imwrite(str(output_path), img_bgr)
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
        H, W = img_bgr.shape[:2]
        mask = create_mask_from_items(ch_items, H, W, pad=6)
        inpainted_bgr = inpaint_image(img_bgr, mask)
        
        inpainted_path = output_dir / "inpainted.png"
        cv2.imwrite(str(inpainted_path), inpainted_bgr)
        
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
