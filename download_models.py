"""
Pre-download PaddleOCR models for Docker build.
This script should be run during Docker image build to cache models.
"""
import os
from paddleocr import PaddleOCR
from config import OCR_CONFIG

print("🔧 Initializing PaddleOCR to download models...")
print(f"📦 Models will be cached in: {os.path.expanduser('~/.paddlex/')}")

# Initialize PaddleOCR - this will download models to ~/.paddlex/
ocr = PaddleOCR(**OCR_CONFIG)

print("✅ PaddleOCR models downloaded successfully!")
print("📁 Model cache location:", os.path.expanduser('~/.paddlex/'))
