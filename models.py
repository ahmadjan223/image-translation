"""
Pydantic models for API requests and responses.
"""
from pydantic import BaseModel, HttpUrl
from typing import Optional, List, Dict, Any


# --- Request Models ---


class TranslateRequest(BaseModel):
    """Request model for translating an image."""
    offer_id: str


# --- Response Models ---

class ChineseItem(BaseModel):
    """Model for a detected Chinese text item."""
    text: str
    conf: float
    poly: Optional[List[List[float]]] = None
    box: Optional[List[float]] = None

# --- Batch Translation Models ---

class BatchTranslateRequest(BaseModel):
    """Request model for batch translating images in HTML content."""
    description: str
    offer_id: Optional[str] = None


class ImageTranslationResult(BaseModel):
    """Result of translating a single image."""
    original_url: str
    local_path: str
    public_url: Optional[str] = None  # GCS public URL
    chinese_count: int
    success: bool
    error: Optional[str] = None


class BatchTranslateResponse(BaseModel):
    """Response model after batch translation."""
    offer_id: str
    message: str
    total_images: int
    successful_translations: int
    failed_translations: int
    translated_html: str
    image_results: List[ImageTranslationResult]
