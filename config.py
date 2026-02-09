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
    GCP_CDN_URL: str = "https://media.public.markaz.app/"
    GOOGLE_APPLICATION_CREDENTIALS: Optional[str] = None
    GEMINI_API_KEY: Optional[str] = None
    
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

# --- Set GCP credentials path for Cloud Storage client ---
if settings.GOOGLE_APPLICATION_CREDENTIALS:
    creds_path = Path(settings.GOOGLE_APPLICATION_CREDENTIALS)
    if not creds_path.is_absolute():
        creds_path = Path(__file__).parent / creds_path
    if creds_path.exists():
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(creds_path)

# --- Directory paths ---
ROOT_DIR = Path(__file__).parent
DOWNLOADS_DIR = ROOT_DIR / "downloads"
OUTPUTS_DIR = ROOT_DIR / "outputs"

# Create directories if they don't exist
DOWNLOADS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

# --- API Keys & Credentials ---
GEMINI_API_KEY = settings.GEMINI_API_KEY
GOOGLE_APPLICATION_CREDENTIALS = settings.GOOGLE_APPLICATION_CREDENTIALS

# --- Regex patterns ---
CJK_RE = re.compile(r"[\u4e00-\u9fff]")

# --- WebP conversion settings ---
WEBP_LOSSLESS = False  # False = use quality mode (70% smaller, no visible loss)
WEBP_QUALITY = 85      # 85 = excellent quality with great compression
WEBP_METHOD = 4        # 0-6, 4 = good balance of speed and compression

# --- Font settings ---
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

# --- Text overlay parameters ---
PAD_IN = 2
MIN_BOX_W = 20
MIN_BOX_H = 14
MIN_FONT_SIZE = 11
MAX_FONT_SIZE = 120
FONT_SIZE_STEP = 2

# --- Image processing parameters ---
OCR_CONFIDENCE_THRESH = 0.8  # Minimum confidence threshold for OCR text detection
MASK_PAD = 6                  # Padding around detected text for inpainting mask

# --- HTTP settings ---
HTTP_TIMEOUT = 60.0  # Timeout in seconds for image downloads
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
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
     "use_gpu": False, 
    "use_doc_unwarping": False,
    "use_doc_orientation_classify": False,
    "text_det_limit_type": "max",
    "text_det_limit_side_len": 4000,
    "text_det_thresh": 0.4,
    "text_det_box_thresh": 0.5,
    "text_det_unclip_ratio": 1.8,
    "text_rec_score_thresh": 0.8
}
