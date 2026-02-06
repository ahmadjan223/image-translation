"""
Image Translation API - Main Application Entry Point

A FastAPI application that performs OCR on Chinese text in images,
translates to English, and overlays the translations using inpainting.
"""
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI

from routes import router

# Create ThreadPoolExecutor with limited workers to prevent unbounded thread growth
# This prevents memory exhaustion from unlimited thread creation
MAX_WORKERS = 4  # Adjust based on CPU cores
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

# Set the default executor for asyncio event loop
def set_default_executor():
    """Set custom executor for all run_in_executor calls."""
    loop = asyncio.get_running_loop()
    loop.set_default_executor(executor)

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
    print("✅ Models are pre-initialized in routes/api.py (OCR_INSTANCE, LAMA_INSTANCE)")
    print(f"✅ ThreadPoolExecutor configured with {MAX_WORKERS} workers")
    print("   Ready to process requests!")
    
    # Set default executor for all async operations
    set_default_executor()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))