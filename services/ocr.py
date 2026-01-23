"""
OCR service using PaddleOCR.
"""
import os
import json
import glob
import cv2
from typing import List, Dict, Any, Optional, Tuple

from paddleocr import PaddleOCR
import threading

from config import CJK_RE, OCR_CONFIG
from utils.image import load_image_to_bgr

# Thread-local storage for PaddleOCR instances (thread-safe)
_thread_local = threading.local()


def get_ocr_instance() -> PaddleOCR:
    """Get or create a thread-local PaddleOCR instance for thread safety."""
    if not hasattr(_thread_local, 'ocr'):
        _thread_local.ocr = PaddleOCR(**OCR_CONFIG)
    return _thread_local.ocr


def ocr_predict_to_json(image_path: str, outdir: str) -> Tuple[Dict, str]:
    """
    Run OCR on image and save results to JSON.
    
    Args:
        image_path: Path to the input image.
        outdir: Directory to save OCR outputs.
        
    Returns:
        Tuple of (ocr_data_dict, fed_image_path).
        
    Raises:
        RuntimeError: If no JSON is produced by OCR.
    """
    os.makedirs(outdir, exist_ok=True)
    
    img_bgr = load_image_to_bgr(image_path)

    # Save the exact bitmap fed to OCR
    fed_path = os.path.join(outdir, "fed_to_ocr.png")
    cv2.imwrite(fed_path, img_bgr)
    
    # Run OCR
    ocr = get_ocr_instance()
    outputs = ocr.predict(fed_path)
    for res in outputs:
        res.save_to_json(outdir)
    
    # Find the latest JSON file
    jfiles = sorted(glob.glob(os.path.join(outdir, "*.json")), key=os.path.getmtime)
    if not jfiles:
        raise RuntimeError("No JSON produced by OCR.")
    
    with open(jfiles[-1], "r", encoding="utf-8") as f:
        data = json.load(f)
    
    return data, fed_path


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


