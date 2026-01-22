"""
Image Translation API - Main Application Entry Point

A FastAPI application that performs OCR on Chinese text in images,
translates to English, and overlays the translations using inpainting.
"""
import os
from fastapi import FastAPI

from routes import router

# Create FastAPI application
app = FastAPI(
    title="Image Translation API",
    description="OCR Chinese text, translate to English, and overlay on images",
    version="0.1.0"
)

# Include API routes
app.include_router(router)


@app.on_event("startup")
async def startup_event():
    """Warm up models on application startup."""
    print("🚀 Starting Image Translation API...")
    print("🔥 Warming up PaddleOCR models...")
    
    # Import and initialize OCR to warm up models
    from services.ocr import get_ocr_instance
    try:
        ocr = get_ocr_instance()
        print("✅ PaddleOCR models loaded successfully!")
    except Exception as e:
        print(f"⚠️  Warning: Could not pre-load OCR models: {e}")
        print("   Models will be loaded on first request.")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))