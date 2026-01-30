"""
Configuration and constants for the Image Translation API.
"""
import os
import re
import sys
from pathlib import Path
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # GCP Settings
    GCP_PROJECT_ID: Optional[str] = None
    GCP_BUCKET_NAME: Optional[str] = None
    GCP_CDN_URL: Optional[str] = None
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


# Create settings instance
settings = Settings()


# --- Add CUDA libraries from venv to LD_LIBRARY_PATH ---
# PaddlePaddle-GPU bundles CUDA libs in venv, but they need to be in LD_LIBRARY_PATH
if sys.prefix:  # Check if we're in a venv
    nvidia_libs = Path(sys.prefix) / "lib/python3.11/site-packages/nvidia"
    if nvidia_libs.exists():
        cuda_paths = []
        for subdir in ["cublas/lib", "cudnn/lib", "cuda_runtime/lib"]:
            lib_path = nvidia_libs / subdir
            if lib_path.exists():
                cuda_paths.append(str(lib_path))
        
        if cuda_paths:
            existing_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
            new_ld_path = ":".join(cuda_paths + ([existing_ld_path] if existing_ld_path else []))
            os.environ["LD_LIBRARY_PATH"] = new_ld_path
            print(f"✅ Added CUDA libraries to LD_LIBRARY_PATH: {':'.join(cuda_paths)}")


# --- Environment setup for threading issues ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OPENBLAS_CORETYPE"] = "generic"

# --- Disable oneDNN/MKL-DNN to avoid compatibility issues in Docker ---
os.environ["FLAGS_use_mkldnn"] = "false"
os.environ["PADDLE_USE_MKLDNN"] = "0"
os.environ["FLAGS_use_onednn"] = "false"

# --- Directory paths ---
ROOT_DIR = Path(__file__).parent
DOWNLOADS_DIR = ROOT_DIR / "downloads"
OUTPUTS_DIR = ROOT_DIR / "outputs"

# Create directories if they don't exist
DOWNLOADS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

# --- API Keys ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyAe33GrwIicD5N4JIwxYSO6Nb7b35s2fH4")

# --- Regex patterns ---
CJK_RE = re.compile(r"[\u4e00-\u9fff]")

# --- WebP conversion settings ---
WEBP_LOSSLESS = True  # Set to True for lossless WebP (best quality, larger files)
WEBP_QUALITY = 100    # Only used when WEBP_LOSSLESS=False
WEBP_METHOD = 6       # 0-6, higher = better compression (slower)

# --- Font settings ---
FONT_PATH = "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"

# --- Text overlay parameters ---
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
True,  # GPU enabled - CUDA libraries are in venv
# --- Translation prompt ---
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

# --- OCR settings ---
OCR_CONFIG = {
    "lang": "ch",
    "use_gpu": False,  # Force CPU - LD_LIBRARY_PATH must be set BEFORE process starts
    "use_doc_unwarping": False,
    "use_doc_orientation_classify": False,
    "text_det_limit_type": "max",
    "text_det_limit_side_len": 4000,
    "text_det_thresh": 0.4,
    "text_det_box_thresh": 0.5,
    "text_det_unclip_ratio": 1.8,
    "text_rec_score_thresh": 0.6
}
