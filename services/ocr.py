"""
OCR service using PaddleOCR.
"""
import os
import json
import glob
import cv2
from typing import List, Dict, Any, Optional, Tuple

from paddleocr import PaddleOCR

from config import CJK_RE, OCR_CONFIG
from utils.image import load_image_to_bgr

# Note: OCR execution is now handled in routes/api.py by run_ocr_on_image()
# This service only contains the helper function to filter Chinese items


def get_chinese_items(ocr_json: Dict, conf_thresh: Optional[float] = None) -> List[Dict[str, Any]]:
    """
    Extract Chinese text items from OCR results.
    
    Args:
        ocr_json: OCR output dictionary.
        conf_thresh: Optional confidence threshold to filter results.
        
    Returns:
        List of dictionaries containing Chinese text items with their properties.
    """
    if not ocr_json:
        return []
    
    rec_texts = ocr_json.get("rec_texts", []) or []
    rec_scores = ocr_json.get("rec_scores", []) or []
    rec_polys = ocr_json.get("rec_polys", None)
    rec_boxes = ocr_json.get("rec_boxes", None)
    
    found = []
    for i, txt in enumerate(rec_texts):
        if not CJK_RE.search(txt or ""):
            continue
        
        score = float(rec_scores[i]) if i < len(rec_scores) else 0.0
        if conf_thresh is not None and score < conf_thresh:
            continue
        
        item = {"text": txt, "conf": score}
        if rec_polys is not None and i < len(rec_polys):
            item["poly"] = rec_polys[i]
        if rec_boxes is not None and i < len(rec_boxes):
            item["box"] = rec_boxes[i]
        found.append(item)
    
    return found


