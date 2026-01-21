import os
import uuid
import httpx
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from typing import Optional
import cv2
import numpy as np
from PIL import Image

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

@app.get("/")
async def root():
    return {
        "message": "Image Translation API",
        "endpoints": {
            "/download": "POST",
            "/health": "GET"
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
