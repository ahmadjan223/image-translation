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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))