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
        RuntimeError: If OCR fails to produce results.
    """
    os.makedirs(outdir, exist_ok=True)
    
    img_bgr = load_image_to_bgr(image_path)

    # Save the exact bitmap fed to OCR
    fed_path = os.path.join(outdir, "fed_to_ocr.png")
    cv2.imwrite(fed_path, img_bgr)
    
    # Run OCR
    ocr = get_ocr_instance()
    results = ocr.ocr(fed_path)
    
    # Convert PaddleOCR results to our JSON format
    # results is a list with one element per image
    if not results or results[0] is None:
        # No text detected
        data = {
            "rec_texts": [],
            "rec_scores": [],
            "rec_boxes": [],
            "rec_polys": []
        }
    else:
        rec_texts = []
        rec_scores = []
        rec_boxes = []
        rec_polys = []
        
        for line in results[0]:
            if line is None:
                continue
            # Each line is: [box_coords, (text, confidence)]
            box_coords = line[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            text_info = line[1]   # (text, confidence)
            
            rec_texts.append(text_info[0])
            rec_scores.append(float(text_info[1]))
            
            # Convert polygon to bounding box [x1, y1, x2, y2]
            xs = [pt[0] for pt in box_coords]
            ys = [pt[1] for pt in box_coords]
            x1, x2 = min(xs), max(xs)
            y1, y2 = min(ys), max(ys)
            rec_boxes.append([x1, y1, x2, y2])
            
            # Keep original polygon coordinates
            rec_polys.append(box_coords)
        
        data = {
            "rec_texts": rec_texts,
            "rec_scores": rec_scores,
            "rec_boxes": rec_boxes,
            "rec_polys": rec_polys
        }
    
    # Save to JSON for debugging
    json_path = os.path.join(outdir, "ocr_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
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


