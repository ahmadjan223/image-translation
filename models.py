"""
Pydantic models for API requests and responses.
"""
from pydantic import BaseModel, HttpUrl
from typing import Optional, List, Dict, Any


# --- Request Models ---

class ImageDownloadRequest(BaseModel):
    """Request model for downloading an image from URL."""
    image_url: HttpUrl
    session_id: Optional[str] = None


class OCRRequest(BaseModel):
    """Request model for running OCR on a session."""
    session_id: str


class TranslateRequest(BaseModel):
    """Request model for translating an image."""
    session_id: str


# --- Response Models ---

class ImageDownloadResponse(BaseModel):
    """Response model after downloading an image."""
    session_id: str
    message: str
    image_path: str
    image_size: dict


class ChineseItem(BaseModel):
    """Model for a detected Chinese text item."""
    text: str
    conf: float
    poly: Optional[List[List[float]]] = None
    box: Optional[List[float]] = None


class OCRResponse(BaseModel):
    """Response model after running OCR."""
    session_id: str
    message: str
    chinese_count: int
    chinese_items: List[Dict[str, Any]]
    ocr_json_path: str
    fed_image_path: str


class TranslateResponse(BaseModel):
    """Response model after full translation pipeline."""
    session_id: str
    message: str
    chinese_count: int
    output_image_path: str
    inpainted_image_path: str
    translations: List[Dict[str, Any]]


# --- Batch Translation Models ---

class BatchTranslateRequest(BaseModel):
    """Request model for batch translating images in HTML content."""
    html_content: str
    session_id: Optional[str] = None


class ImageTranslationResult(BaseModel):
    """Result of translating a single image."""
    original_url: str
    local_path: str
    chinese_count: int
    success: bool
    error: Optional[str] = None


class BatchTranslateResponse(BaseModel):
    """Response model after batch translation."""
    session_id: str
    message: str
    total_images: int
    successful_translations: int
    failed_translations: int
    translated_html: str
    image_results: List[ImageTranslationResult]
