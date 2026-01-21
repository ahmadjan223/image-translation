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
