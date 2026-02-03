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


# def run_ocr_on_image(image_path: str, outdir: str, ocr: PaddleOCR) -> Tuple[Dict, str]:
#     """
#     Run OCR on image using provided OCR instance.
    
#     Args:
#         image_path: Path to the input image
#         outdir: Directory to save OCR outputs
#         ocr: PaddleOCR instance to use
        
#     Returns:
#         Tuple of (ocr_data_dict, fed_image_path)
#     """
#     os.makedirs(outdir, exist_ok=True)
    
#     img_bgr = load_image_to_bgr(image_path)
    
#     # Save the exact bitmap fed to OCR
#     fed_path = os.path.join(outdir, "fed_to_ocr.png")
#     cv2.imwrite(fed_path, img_bgr)
    
#     # Run OCR using predict() which returns OCRResult objects
#     outputs = ocr.predict(fed_path)
    
#     # Save OCRResult to JSON
#     for res in outputs:
#         if res is not None:
#             res.save_to_json(outdir)
    
#     # Find the JSON file that was just created
#     jfiles = sorted(glob.glob(os.path.join(outdir, "*.json")), key=os.path.getmtime)
#     if not jfiles:
#         # No detections - return empty structure
#         data = {
#             "rec_texts": [],
#             "rec_scores": [],
#             "rec_boxes": [],
#             "rec_polys": []
#         }
#     else:
#         # Load the JSON file
#         with open(jfiles[-1], "r", encoding="utf-8") as f:
#             data = json.load(f)
    
#     return data, fed_path

def run_ocr_on_image(image_path: str, outdir: str, ocr: PaddleOCR):
    """
    Run OCR on image using provided OCR instance.
    
    This is a simplified version of ocr_predict_to_json that receives
    the OCR instance as a parameter instead of calling get_ocr_instance().
    """
    import os
    
    os.makedirs(outdir, exist_ok=True)
    
    img_bgr = load_image_to_bgr(image_path)
    
    # Save the exact bitmap fed to OCR
    fed_path = os.path.join(outdir, "fed_to_ocr.png")
    cv2.imwrite(fed_path, img_bgr)
    
    # Run OCR using the provided instance
    results = ocr.ocr(fed_path)
    
    # Convert PaddleOCR results to our JSON format
    if not results or results[0] is None:
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
            box_coords = line[0]
            text_info = line[1]
            
            rec_texts.append(text_info[0])
            rec_scores.append(float(text_info[1]))
            
            # Convert polygon to bounding box
            xs = [pt[0] for pt in box_coords]
            ys = [pt[1] for pt in box_coords]
            x1, x2 = min(xs), max(xs)
            y1, y2 = min(ys), max(ys)
            rec_boxes.append([x1, y1, x2, y2])
            rec_polys.append(box_coords)
        
        data = {
            "rec_texts": rec_texts,
            "rec_scores": rec_scores,
            "rec_boxes": rec_boxes,
            "rec_polys": rec_polys
        }
    
    # Save to JSON
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


