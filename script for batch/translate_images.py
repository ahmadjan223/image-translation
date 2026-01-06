#!/usr/bin/env python3
"""
Chinese to English Image Translation Script

This script processes images containing Chinese text by:
1. Detecting Chinese text using PaddleOCR
2. Translating the text to English using Google Gemini API
3. Removing the original text using AI inpainting
4. Overlaying the English translation with optimized font fitting

Usage:
    python translate_images.py input_folder output_folder [--gemini-key YOUR_API_KEY]

Requirements:
    - paddleocr, paddlepaddle
    - opencv-python-headless
    - pillow
    - simple-lama-inpainting
    - google-genai
    - numpy
    - matplotlib (for debugging)
"""

import os

# Fix threading conflicts that cause crashes during OCR prediction
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import sys
import json
import glob
import argparse
import math
import re
import shutil
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# OCR imports
from paddleocr import PaddleOCR

# Inpainting import
from simple_lama_inpainting import SimpleLama

# Translation import
from google import genai

# Constants and default parameters
DEFAULT_PARAMS = {
    'pad_in': 2,                    # inner padding inside each box (px)
    'min_box_w': 20,               # boxes narrower than this are skipped
    'min_box_h': 14,               # boxes shorter than this are skipped
    'min_font_size': 12,           # below this, try multi-line
    'max_font_size': 120,          # upper bound for font sizing
    'font_size_step': 2,           # decrement step when searching
    'split_threshold': 12,         # if single-line font <= this, try 2 lines
    'max_lines_fallback': 2,       # max lines when splitting
    'line_spacing': 1.025,         # line height multiplier
    'shadow_blur': 1,              # Gaussian blur radius for shadow
    'shadow_offset': (1, 1),       # pixel offset for shadow
}

# Chinese character regex
CJK_RE = re.compile(r"[\u4e00-\u9fff]")

# Translation system prompt
SYSTEM_PROMPT = """
Translate Chinese OCR lines to English for product images.

Rules:
- ONE translation per line (no options, no numbering, no explanations).
- Keep it SHORT to fit the original box: en length <= max_chars.
- Keep repeated terms consistent.
- Use simple, short size synonyms to occupy less space for product images.

Output JSON ONLY:
[{ "i": <index>, "en": "<translation>" }]
""".strip()


class ImageTranslator:
    """Main class for Chinese to English image translation"""
    
    def __init__(self, gemini_api_key: str, params: Dict = None):
        """Initialize the translator with API key and parameters"""
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        
        # Initialize OCR
        self.ocr = PaddleOCR(
            lang="ch",
            use_doc_unwarping=False,
            use_doc_orientation_classify=False,
            text_det_limit_type="max",
            text_det_limit_side_len=4000,
            text_det_thresh=0.2,
            text_det_box_thresh=0.3,
            text_det_unclip_ratio=1.8,
            text_rec_score_thresh=0.0
        )
        
        # Initialize inpainting
        self.simple_lama = SimpleLama()
        
        # Initialize translation client
        self.gemini_client = genai.Client(api_key=gemini_api_key)
        
        # Setup font paths
        self.font_candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        ]
        
        self.font_path = self._find_font()
        
        print(f"✅ ImageTranslator initialized with font: {os.path.basename(self.font_path)}")
    
    def _find_font(self) -> str:
        """Find the first available font from candidates"""
        for candidate in self.font_candidates:
            if os.path.exists(candidate):
                return candidate
        
        raise RuntimeError(f"No suitable font found! Tried: {self.font_candidates}")
    
    def load_image_to_bgr(self, path: str) -> np.ndarray:
        """Load image in any format to BGR (OpenCV style)"""
        ext = os.path.splitext(path.lower())[1]
        
        # Use OpenCV for JPEG + WEBP (avoids Pillow WebP bug)
        if ext in (".jpg", ".jpeg", ".webp"):
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise RuntimeError(f"OpenCV failed to read image: {path}")
            
            # If WebP has alpha (BGRA), convert to BGR
            if img.ndim == 3 and img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            
            # If grayscale, convert to BGR
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            
            return img
        
        # Pillow for everything else
        img = Image.open(path).convert("RGB")
        arr = np.array(img)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    
    def detect_chinese_text(self, image_path: str) -> Tuple[List[Dict], str]:
        """Detect Chinese text in image and return OCR results"""
        print(f"   🔍 Step 1: Loading image and running OCR...")
        img_bgr = self.load_image_to_bgr(image_path)
        
        # Save temp image for OCR
        temp_dir = os.path.dirname(image_path)
        fed_path = os.path.join(temp_dir, "temp_ocr_input.png")
        cv2.imwrite(fed_path, img_bgr)
        
        # Run OCR
        print(f"   📖 Running OCR on {os.path.basename(image_path)}...")
        outputs = self.ocr.predict(fed_path)
        
        # Process OCR results like in the notebook
        if not outputs:
            print(f"   ⚠️ OCR returned no results")
            return [], fed_path
        
        # Save outputs to JSON and reload (matches notebook approach)
        temp_json_dir = os.path.join(temp_dir, "temp_ocr_json")
        os.makedirs(temp_json_dir, exist_ok=True)
        
        # Save each result to JSON
        for res in outputs:
            res.save_to_json(temp_json_dir)
        
        # Load the most recent JSON file
        jfiles = sorted(glob.glob(os.path.join(temp_json_dir, "*.json")), key=os.path.getmtime)
        if not jfiles:
            print(f"   ⚠️ No JSON files generated")
            return [], fed_path
        
        with open(jfiles[-1], "r", encoding="utf-8") as f:
            ocr_data = json.load(f)
        
        # Debug: show all OCR results
        rec_texts = ocr_data.get("rec_texts", []) or []
        rec_scores = ocr_data.get("rec_scores", []) or []
        print(f"   📊 OCR found {len(rec_texts)} text regions total")
        for i, (txt, score) in enumerate(zip(rec_texts[:5], rec_scores[:5])):
            print(f"      {i+1}. '{txt}' (confidence: {score:.2f})")
        if len(rec_texts) > 5:
            print(f"      ... and {len(rec_texts)-5} more")
        
        # Extract Chinese items
        chinese_items = self._get_chinese_items(ocr_data)
        print(f"   🈯 Found {len(chinese_items)} regions with Chinese text")
        
        # Clean up temp files
        if os.path.exists(fed_path):
            os.remove(fed_path)
        import shutil
        if os.path.exists(temp_json_dir):
            shutil.rmtree(temp_json_dir)
        
        return chinese_items, image_path
    
    def _get_chinese_items(self, ocr_json: Dict, conf_thresh: float = 0.1) -> List[Dict]:
        """Filter OCR results for Chinese text only"""
        if not ocr_json:
            return []
        
        rec_texts = ocr_json.get("rec_texts", []) or []
        rec_scores = ocr_json.get("rec_scores", []) or []
        rec_polys = ocr_json.get("rec_polys", None)
        rec_boxes = ocr_json.get("rec_boxes", None)
        
        found = []
        for i, txt in enumerate(rec_texts):
            txt = txt or ""
            # Check for Chinese characters (broader detection)
            has_chinese = bool(CJK_RE.search(txt))
            # Also check for other Asian scripts that might be Chinese
            has_asian = any(ord(c) > 0x3000 for c in txt)
            
            print(f"      Checking: '{txt[:20]}...' - Chinese: {has_chinese}, Asian: {has_asian}")
            
            if not (has_chinese or has_asian):
                continue
            
            score = float(rec_scores[i]) if i < len(rec_scores) else 0.0
            if score < conf_thresh:
                print(f"        Skipping low confidence: {score:.2f} < {conf_thresh}")
                continue
            
            item = {"text": txt, "conf": score}
            if rec_polys is not None and i < len(rec_polys):
                item["poly"] = rec_polys[i]
            if rec_boxes is not None and i < len(rec_boxes):
                item["box"] = rec_boxes[i]
            found.append(item)
            print(f"        ✅ Added Chinese text: '{txt[:30]}...' (conf: {score:.2f})")
        
        return found
    
    def translate_chinese_items(self, chinese_items: List[Dict], model: str = "models/gemini-flash-lite-latest") -> List[str]:
        """Translate Chinese text items to English"""
        if not chinese_items:
            return []
        
        items = []
        for i, item in enumerate(chinese_items):
            cn = (item.get("text") or "").strip()
            max_chars = max(6, int(len(cn) * 1.35))
            items.append({"i": i, "cn": cn, "max_chars": max_chars})
        
        payload = {"items": items}
        
        try:
            resp = self.gemini_client.models.generate_content(
                model=model,
                contents=json.dumps(payload, ensure_ascii=False),
                config={
                    "system_instruction": SYSTEM_PROMPT,
                    "temperature": 0.2,
                    "response_mime_type": "application/json",
                },
            )
            
            arr = json.loads((resp.text or "").strip())
            mp = {int(o["i"]): (o.get("en") or "").strip() for o in arr}
            return [mp.get(i, "") for i in range(len(items))]
        
        except Exception as e:
            print(f"❌ Translation failed: {e}")
            return ["" for _ in items]
    
    def create_inpaint_mask(self, chinese_items: List[Dict], img_shape: Tuple[int, int, int]) -> np.ndarray:
        """Create mask for inpainting from Chinese text detections"""
        H, W = img_shape[:2]
        mask = np.zeros((H, W), dtype=np.uint8)
        
        for item in chinese_items:
            if item.get("poly") is not None:
                pts = np.array(item["poly"], dtype=np.int32).reshape(-1, 2)
                cv2.fillPoly(mask, [pts], 255)
            elif item.get("box") is not None:
                x1, y1, x2, y2 = map(int, item["box"])
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(W - 1, x2), min(H - 1, y2)
                if x2 > x1 and y2 > y1:
                    cv2.rectangle(mask, (x1, y1), (x2, y2), 255, thickness=-1)
        
        # Expand mask a bit (covers strokes better)
        pad = 6
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*pad + 1, 2*pad + 1))
        mask = cv2.dilate(mask, k, iterations=1)
        
        return mask
    
    def inpaint_image(self, img_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Remove Chinese text using AI inpainting"""
        try:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            inpaint_pil = self.simple_lama(img_rgb, mask)
            out = cv2.cvtColor(np.array(inpaint_pil), cv2.COLOR_RGB2BGR)
            print("✅ SimpleLaMa inpainting done.")
            return out
        except Exception as e:
            print(f"⚠️ SimpleLaMa failed, falling back to OpenCV inpaint. Error: {str(e)[:100]}")
            out = cv2.inpaint(img_bgr, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
            print("✅ OpenCV inpaint done.")
            return out
    
    # Text rendering helper functions
    def clamp_box(self, x1: int, y1: int, x2: int, y2: int, W: int, H: int) -> Tuple[int, int, int, int]:
        """Clamp bounding box to image dimensions"""
        x1 = int(max(0, min(W - 1, x1)))
        y1 = int(max(0, min(H - 1, y1)))
        x2 = int(max(0, min(W - 1, x2)))
        y2 = int(max(0, min(H - 1, y2)))
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        return x1, y1, x2, y2
    
    def text_size(self, draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont) -> Tuple[int, int]:
        """Get text dimensions"""
        bb = draw.textbbox((0, 0), text, font=font)
        return bb[2] - bb[0], bb[3] - bb[1]
    
    def wrap_text(self, draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, 
                  max_w: int, max_lines: int = 2) -> List[str]:
        """Wrap text to fit width constraints"""
        text = (text or "").strip()
        if not text:
            return []
        
        # Fits on one line?
        if self.text_size(draw, text, font)[0] <= max_w:
            return [text]
        
        words = text.split()
        if len(words) == 1:
            # char wrap for single long word
            lines = []
            cur = ""
            for ch in text:
                test = cur + ch
                if self.text_size(draw, test, font)[0] <= max_w or not cur:
                    cur = test
                else:
                    lines.append(cur)
                    cur = ch
                    if len(lines) >= max_lines:
                        break
            if len(lines) < max_lines and cur:
                lines.append(cur)
            return lines[:max_lines]
        
        # word wrap
        lines = []
        cur = ""
        for w in words:
            test = (cur + " " + w).strip()
            if self.text_size(draw, test, font)[0] <= max_w or not cur:
                cur = test
            else:
                lines.append(cur)
                cur = w
                if len(lines) >= max_lines:
                    break
        if len(lines) < max_lines and cur:
            lines.append(cur)
        
        return lines[:max_lines]
    
    def truncate_line_to_width(self, draw: ImageDraw.ImageDraw, s: str, 
                              font: ImageFont.FreeTypeFont, max_w: int) -> str:
        """Truncate line with ellipsis to fit width"""
        s = (s or "").strip()
        if not s:
            return ""
        if self.text_size(draw, s, font)[0] <= max_w:
            return s
        ell = "…"
        if self.text_size(draw, ell, font)[0] > max_w:
            return ""
        while s and self.text_size(draw, s + ell, font)[0] > max_w:
            s = s[:-1]
        return (s + ell) if s else ell
    
    def fit_font_single_line(self, draw: ImageDraw.ImageDraw, text: str, target_w: int, target_h: int) -> Optional[Tuple]:
        """Try to fit text on ONE line. Returns (font, [line], font_size) or None."""
        text = (text or "").strip()
        if not text:
            return None
        
        for sz in range(self.params['max_font_size'], self.params['min_font_size'] - 1, 
                       -self.params['font_size_step']):
            f = ImageFont.truetype(self.font_path, sz)
            tw, th = self.text_size(draw, text, f)
            if tw <= target_w and th <= target_h:
                return f, [text], sz
        return None
    
    def fit_font_multi_line(self, draw: ImageDraw.ImageDraw, text: str, target_w: int, target_h: int) -> Optional[Tuple]:
        """Try to fit text wrapped to max_lines. Returns (font, lines, font_size)."""
        text = (text or "").strip()
        if not text:
            return None
        
        best = None
        for sz in range(self.params['max_font_size'], self.params['min_font_size'] - 1, 
                       -self.params['font_size_step']):
            f = ImageFont.truetype(self.font_path, sz)
            lines = self.wrap_text(draw, text, f, target_w, max_lines=self.params['max_lines_fallback'])
            if not lines:
                continue
            
            # truncate any overflowing line
            lines = [self.truncate_line_to_width(draw, li, f, target_w) for li in lines]
            
            line_h = self.text_size(draw, "Ag", f)[1]
            total_h = int(line_h * len(lines) * self.params['line_spacing'])
            max_line_w = max(self.text_size(draw, li, f)[0] for li in lines) if lines else 0
            
            if max_line_w <= target_w and total_h <= target_h:
                return f, lines, sz
            
            best = (f, lines, sz)
        
        return best
    
    def sample_bg_luma(self, bgr_img: np.ndarray, x1: int, y1: int, x2: int, y2: int, pad: int = 6) -> float:
        """Sample border region around the box to guess background brightness."""
        H, W = bgr_img.shape[:2]
        x1p = max(0, x1 - pad)
        y1p = max(0, y1 - pad)
        x2p = min(W, x2 + pad)
        y2p = min(H, y2 + pad)
        roi = bgr_img[y1p:y2p, x1p:x2p]
        if roi.size == 0:
            return 255
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        return float(np.median(gray))
    
    def draw_text_with_shadow(self, pil_rgba: Image.Image, xy: Tuple[int, int], lines: List[str], 
                             font: ImageFont.FreeTypeFont, fill_rgba: Tuple, shadow_rgba: Tuple,
                             align: str = "center") -> Tuple[Image.Image, Tuple[int, int]]:
        """Draw multi-line text with a soft shadow layer."""
        base = pil_rgba
        x, y = xy
        tmp = Image.new("RGBA", base.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(tmp)
        
        line_h = d.textbbox((0, 0), "Ag", font=font)[3]
        widths = [d.textbbox((0, 0), li, font=font)[2] for li in lines]
        block_w = max(widths) if widths else 0
        step_y = int(line_h * self.params['line_spacing'])
        
        # Draw shadow
        yy = y
        for li, w in zip(lines, widths):
            if align == "center":
                xx = x + (block_w - w) // 2
            elif align == "left":
                xx = x
            else:
                xx = x + (block_w - w)
            d.text((xx + self.params['shadow_offset'][0], yy + self.params['shadow_offset'][1]), 
                   li, font=font, fill=shadow_rgba)
            yy += step_y
        
        if self.params['shadow_blur'] > 0:
            tmp = tmp.filter(ImageFilter.GaussianBlur(radius=self.params['shadow_blur']))
        
        # Draw main text
        d = ImageDraw.Draw(tmp)
        yy = y
        for li, w in zip(lines, widths):
            if align == "center":
                xx = x + (block_w - w) // 2
            elif align == "left":
                xx = x
            else:
                xx = x + (block_w - w)
            d.text((xx, yy), li, font=font, fill=fill_rgba)
            yy += step_y
        
        block_h = int(line_h * len(lines) * self.params['line_spacing'])
        return Image.alpha_composite(base, tmp), (block_w, block_h)
    
    def overlay_english_text(self, img_bgr: np.ndarray, chinese_items: List[Dict], 
                            english_translations: List[str]) -> np.ndarray:
        """Overlay English translations on the inpainted image"""
        H, W = img_bgr.shape[:2]
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb).convert("RGBA")
        draw = ImageDraw.Draw(pil_img)
        
        for item, en in zip(chinese_items, english_translations):
            en = (en or "").strip()
            if not en or item.get("box") is None:
                continue
            
            x1, y1, x2, y2 = map(int, item["box"])
            x1, y1, x2, y2 = self.clamp_box(x1, y1, x2, y2, W, H)
            bw, bh = (x2 - x1), (y2 - y1)
            
            if bw < self.params['min_box_w'] or bh < self.params['min_box_h']:
                continue
            
            target_w = max(1, bw - 2 * self.params['pad_in'])
            target_h = max(1, bh - 2 * self.params['pad_in'])
            
            # Strategy: try single-line first
            result = self.fit_font_single_line(draw, en, target_w, target_h)
            
            if result:
                font, lines, font_sz = result
                # If font size is too small, try multi-line for better visibility
                if font_sz <= self.params['split_threshold']:
                    multi = self.fit_font_multi_line(draw, en, target_w, target_h)
                    if multi and multi[2] > font_sz:
                        font, lines, font_sz = multi
            else:
                # Single-line failed entirely, try multi-line
                multi = self.fit_font_multi_line(draw, en, target_w, target_h)
                if multi:
                    font, lines, font_sz = multi
                else:
                    continue
            
            if not lines:
                continue
            
            # Pick text color based on local background brightness
            luma = self.sample_bg_luma(img_bgr, x1, y1, x2, y2, pad=8)
            if luma > 160:
                fill = (15, 15, 15, 255)
                shadow = (255, 255, 255, 140)
            else:
                fill = (245, 245, 245, 255)
                shadow = (0, 0, 0, 160)
            
            # Compute block size and center it
            widths = [draw.textbbox((0, 0), li, font=font)[2] for li in lines]
            line_h = draw.textbbox((0, 0), "Ag", font=font)[3]
            block_w = max(widths) if widths else 0
            block_h = int(line_h * len(lines) * self.params['line_spacing'])
            
            tx = x1 + (bw - block_w) // 2
            ty = y1 + (bh - block_h) // 2
            
            pil_img, _ = self.draw_text_with_shadow(
                pil_img, (tx, ty), lines, font,
                fill_rgba=fill, shadow_rgba=shadow, align="center"
            )
        
        # Convert back to BGR
        result_rgb = pil_img.convert("RGB")
        return cv2.cvtColor(np.array(result_rgb), cv2.COLOR_RGB2BGR)
    
    def process_image(self, input_path: str, output_path: str) -> bool:
        """Process a single image: OCR -> Translate -> Inpaint -> Overlay"""
        try:
            print(f"\n🔄 Processing: {os.path.basename(input_path)}")
            print(f"   📁 Input: {input_path}")
            print(f"   📁 Output: {output_path}")
            
            # Step 1: Load and detect Chinese text
            print(f"\n📖 STEP 1: OCR Detection")
            chinese_items, _ = self.detect_chinese_text(input_path)
            if not chinese_items:
                print(f"❌ No Chinese text found in {input_path}")
                print(f"   💡 Tip: Check if image contains Chinese text and OCR settings")
                return False
            
            print(f"   ✅ Step 1 Complete: Found {len(chinese_items)} Chinese text regions")
            
            # Step 2: Translate Chinese to English
            print(f"\n🌐 STEP 2: Translation")
            print(f"   🔄 Sending {len(chinese_items)} items to Gemini API...")
            english_translations = self.translate_chinese_items(chinese_items)
            
            # Update items with translations
            for item, en in zip(chinese_items, english_translations):
                item["en"] = en
                print(f"      '{item['text'][:20]}...' → '{en[:30]}...'")
            
            successful_translations = len([t for t in english_translations if t])
            print(f"   ✅ Step 2 Complete: Translated {successful_translations}/{len(chinese_items)} items")
            
            # Step 3: Load original image and create inpaint mask
            print(f"\n🎨 STEP 3: Inpainting Preparation")
            img_bgr = self.load_image_to_bgr(input_path)
            print(f"   📏 Image dimensions: {img_bgr.shape}")
            mask = self.create_inpaint_mask(chinese_items, img_bgr.shape)
            mask_area = np.sum(mask > 0)
            print(f"   🖼️ Inpaint mask created: {mask_area} pixels to inpaint")
            print(f"   ✅ Step 3 Complete: Mask ready")
            
            # Step 4: Inpaint to remove Chinese text
            print(f"\n🔧 STEP 4: Inpainting (Removing Chinese text)")
            inpainted_img = self.inpaint_image(img_bgr, mask)
            print(f"   ✅ Step 4 Complete: Chinese text removed")
            
            # Step 5: Overlay English text
            print(f"\n✍️ STEP 5: English Text Overlay")
            final_img = self.overlay_english_text(inpainted_img, chinese_items, english_translations)
            print(f"   ✅ Step 5 Complete: English text overlaid")
            
            # Step 6: Save result
            print(f"\n💾 STEP 6: Saving Result")
            cv2.imwrite(output_path, final_img)
            print(f"   ✅ Step 6 Complete: Saved to {output_path}")
            
            print(f"\n🎉 SUCCESS: Image processing complete!")
            return True
            
        except Exception as e:
            print(f"❌ Error processing {input_path}: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Main function with command-line interface"""
    parser = argparse.ArgumentParser(
        description="Chinese to English Image Translation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python translate_images.py input_folder output_folder
    python translate_images.py input_folder output_folder --gemini-key YOUR_API_KEY
    python translate_images.py input_folder output_folder --model gemini-flash-lite-latest
        """
    )
    
    parser.add_argument("input_folder", help="Input folder containing images with Chinese text")
    parser.add_argument("output_folder", help="Output folder for translated images")
    parser.add_argument("--gemini-key", help="Google Gemini API key (or set GEMINI_API_KEY env var)")
    parser.add_argument("--model", default="gemini-2.0-flash-exp", 
                       help="Gemini model to use for translation")
    parser.add_argument("--extensions", default="jpg,jpeg,png,webp,bmp,tif,tiff",
                       help="Image file extensions to process (comma-separated)")
    
    args = parser.parse_args()
    
    # Get API key from argument or environment
    gemini_key = args.gemini_key or os.environ.get("GEMINI_API_KEY")
    if not gemini_key:
        print("❌ Error: Gemini API key required. Provide --gemini-key or set GEMINI_API_KEY env var")
        sys.exit(1)
    
    # Validate input folder
    input_folder = Path(args.input_folder)
    if not input_folder.exists():
        print(f"❌ Error: Input folder does not exist: {input_folder}")
        sys.exit(1)
    
    # Create output folder
    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Find images to process
    extensions = args.extensions.split(',')
    image_files = []
    for ext in extensions:
        pattern = f"*.{ext.strip()}"
        image_files.extend(list(input_folder.glob(pattern)))
        image_files.extend(list(input_folder.glob(pattern.upper())))
    
    image_files = sorted(list(set(image_files)))  # Remove duplicates and sort
    
    if not image_files:
        print(f"❌ Error: No images found in {input_folder} with extensions: {extensions}")
        sys.exit(1)
    
    print(f"📁 Found {len(image_files)} images to process")
    
    # Initialize translator
    try:
        translator = ImageTranslator(gemini_key)
    except Exception as e:
        print(f"❌ Error initializing translator: {e}")
        sys.exit(1)
    
    # Process images
    success_count = 0
    for image_file in image_files:
        output_file = output_folder / f"{image_file.stem}_translated{image_file.suffix}"
        
        if translator.process_image(str(image_file), str(output_file)):
            success_count += 1
    
    print(f"\n🎉 Completed! Successfully processed {success_count}/{len(image_files)} images")
    print(f"📁 Results saved to: {output_folder}")


if __name__ == "__main__":
    main()