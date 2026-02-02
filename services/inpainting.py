"""
Inpainting service using SimpleLama.
"""
import cv2
import numpy as np
from typing import List, Dict

from simple_lama_inpainting import SimpleLama


def create_mask_from_items(
    ch_items: List[Dict],
    H: int,
    W: int,
    pad: int = 6
) -> np.ndarray:
    """
    Create inpainting mask from Chinese text detections.
    
    Args:
        ch_items: List of detected Chinese text items with poly/box info.
        H: Image height.
        W: Image width.
        pad: Padding to expand the mask around detected regions.
        
    Returns:
        Binary mask as numpy array (255 for regions to inpaint).
    """
    mask = np.zeros((H, W), dtype=np.uint8)
    
    for it in ch_items:
        if it.get("poly") is not None:
            pts = np.array(it["poly"], dtype=np.int32).reshape(-1, 2)
            cv2.fillPoly(mask, [pts], 255)
        elif it.get("box") is not None:
            x1, y1, x2, y2 = map(int, it["box"])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W - 1, x2), min(H - 1, y2)
            if x2 > x1 and y2 > y1:
                cv2.rectangle(mask, (x1, y1), (x2, y2), 255, thickness=-1)
    
    # Expand mask to cover strokes better
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*pad + 1, 2*pad + 1))
    mask = cv2.dilate(mask, k, iterations=1)
    
    return mask
