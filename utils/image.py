"""
Image loading and manipulation utilities.
"""
import os
import cv2
import numpy as np
from PIL import Image


def load_image_to_bgr(path: str) -> np.ndarray:
    """
    Load ANY common image format into BGR (OpenCV style).
    
    Args:
        path: Path to the image file.
        
    Returns:
        BGR numpy array of the image.
        
    Raises:
        RuntimeError: If the image cannot be loaded.
    """
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
    
    # Fallback to PIL for other formats
    img = Image.open(path).convert("RGB")
    arr = np.array(img)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def bgr_to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    """Convert BGR image to RGB."""
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(img_rgb: np.ndarray) -> np.ndarray:
    """Convert RGB image to BGR."""
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)


def save_image(img_bgr: np.ndarray, path: str) -> None:
    """Save a BGR image to file."""
    cv2.imwrite(path, img_bgr)


def get_image_dimensions(img: np.ndarray) -> tuple:
    """
    Get image height and width.
    
    Returns:
        Tuple of (height, width).
    """
    return img.shape[:2]


def convert_to_webp(
    image_source,
    output_path: Optional[Path] = None,
    lossless: bool = False,
    quality: int = 85,
    method: int = 4
) -> bytes:
    """
    Convert image to WebP format.
    
    Args:
        image_source: Either a file path (str/Path) or PIL Image object
        output_path: Optional path to save the WebP file
        lossless: Use lossless compression (larger files)
        quality: Quality setting (0-100, only used if not lossless)
        method: Compression method (0-6, higher = better compression but slower)
        
    Returns:
        WebP image as bytes
    """
    # Load image if path provided, otherwise use PIL Image directly
    if isinstance(image_source, (str, Path)):
        img = Image.open(image_source).convert("RGB")
    else:
        img = image_source.convert("RGB")
    
    # Convert to WebP
    img_buffer = io.BytesIO()
    img.save(img_buffer, format="WEBP", lossless=lossless, quality=quality, method=method)
    img_bytes = img_buffer.getvalue()
    
    # Optionally save to file
    if output_path:
        with open(output_path, "wb") as f:
            f.write(img_bytes)
    
    return img_bytes


def get_file_extension(image_url: str, content_type: str) -> str:
    """
    Determine file extension from content type or URL.
    
    Args:
        image_url: URL of the image
        content_type: HTTP Content-Type header value
        
    Returns:
        File extension (with leading dot)
    """
    content_type = content_type.lower()
    if "jpeg" in content_type or "jpg" in content_type:
        return ".jpg"
    elif "webp" in content_type:
        return ".webp"
    elif "png" in content_type:
        return ".png"
    else:
        url_path = image_url.split("?")[0]
        return Path(url_path).suffix or ".jpg"
