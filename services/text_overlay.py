"""
Text overlay service for rendering translated text on images.
"""
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from typing import List, Dict, Tuple, Optional

from config import (
    FONT_PATH,
    PAD_IN,
    MIN_BOX_W,
    MIN_BOX_H,
    MIN_FONT_SIZE,
    MAX_FONT_SIZE,
    FONT_SIZE_STEP,
    SPLIT_THRESHOLD,
    MAX_LINES_FALLBACK,
    LINE_SPACING,
    SHADOW_BLUR,
    SHADOW_OFFSET
)


def clamp_box(x1: int, y1: int, x2: int, y2: int, W: int, H: int) -> Tuple[int, int, int, int]:
    """Clamp box coordinates to image boundaries."""
    x1 = int(max(0, min(W - 1, x1)))
    y1 = int(max(0, min(H - 1, y1)))
    x2 = int(max(0, min(W - 1, x2)))
    y2 = int(max(0, min(H - 1, y2)))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont) -> Tuple[int, int]:
    """Get the size of text when rendered with given font."""
    bb = draw.textbbox((0, 0), text, font=font)
    return bb[2] - bb[0], bb[3] - bb[1]


def wrap_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont,
    max_w: int,
    max_lines: int = 2
) -> List[str]:
    """
    Wrap text to fit within a maximum width.
    Uses greedy wrap by spaces; falls back to char wrap.
    """
    text = (text or "").strip()
    if not text:
        return []
    if text_size(draw, text, font)[0] <= max_w:
        return [text]
    
    words = text.split()
    if len(words) == 1:
        lines, cur = [], ""
        for ch in text:
            test = cur + ch
            if text_size(draw, test, font)[0] <= max_w or not cur:
                cur = test
            else:
                lines.append(cur)
                cur = ch
                if len(lines) >= max_lines:
                    break
        if len(lines) < max_lines and cur:
            lines.append(cur)
        return lines[:max_lines]
    
    lines, cur = [], ""
    for w in words:
        test = (cur + " " + w).strip()
        if text_size(draw, test, font)[0] <= max_w or not cur:
            cur = test
        else:
            lines.append(cur)
            cur = w
            if len(lines) >= max_lines:
                break
    if len(lines) < max_lines and cur:
        lines.append(cur)
    return lines[:max_lines]


def truncate_line_to_width(
    draw: ImageDraw.ImageDraw,
    s: str,
    font: ImageFont.FreeTypeFont,
    max_w: int
) -> str:
    """Truncate a line to fit within max width, adding ellipsis if needed."""
    s = (s or "").strip()
    if not s:
        return ""
    if text_size(draw, s, font)[0] <= max_w:
        return s
    ell = "…"
    if text_size(draw, ell, font)[0] > max_w:
        return ""
    while s and text_size(draw, s + ell, font)[0] > max_w:
        s = s[:-1]
    return (s + ell) if s else ell


def fit_font_single_line(
    draw: ImageDraw.ImageDraw,
    text: str,
    target_w: int,
    target_h: int,
    font_path: str,
    min_size: int = 10,
    max_size: int = 200,
    step: int = 2
) -> Optional[Tuple[ImageFont.FreeTypeFont, List[str], int]]:
    """
    Find the largest font size that fits text in a single line.
    
    Returns:
        Tuple of (font, lines, font_size) or None if it doesn't fit.
    """
    text = (text or "").strip()
    if not text:
        return None
    for sz in range(max_size, min_size - 1, -step):
        f = ImageFont.truetype(font_path, sz)
        tw, th = text_size(draw, text, f)
        if tw <= target_w and th <= target_h:
            return f, [text], sz
    return None


def fit_font_multi_line(
    draw: ImageDraw.ImageDraw,
    text: str,
    target_w: int,
    target_h: int,
    font_path: str,
    max_lines: int = 2,
    min_size: int = 10,
    max_size: int = 200,
    step: int = 2,
    line_spacing: float = 1.1
) -> Optional[Tuple[ImageFont.FreeTypeFont, List[str], int]]:
    """
    Find the largest font size that fits text in multiple lines.
    
    Returns:
        Tuple of (font, lines, font_size) or None if it doesn't fit.
    """
    text = (text or "").strip()
    if not text:
        return None
    best = None
    for sz in range(max_size, min_size - 1, -step):
        f = ImageFont.truetype(font_path, sz)
        lines = wrap_text(draw, text, f, target_w, max_lines=max_lines)
        if not lines:
            continue
        lines = [truncate_line_to_width(draw, li, f, target_w) for li in lines]
        line_h = text_size(draw, "Ag", f)[1]
        total_h = int(line_h * len(lines) * line_spacing)
        max_line_w = max(text_size(draw, li, f)[0] for li in lines) if lines else 0
        if max_line_w <= target_w and total_h <= target_h:
            return f, lines, sz
        best = (f, lines, sz)
    return best


def sample_bg_luma(bgr_img: np.ndarray, x1: int, y1: int, x2: int, y2: int, pad: int = 6) -> float:
    """Sample the median luminance of the background around a region."""
    H, W = bgr_img.shape[:2]
    x1p, y1p = max(0, x1 - pad), max(0, y1 - pad)
    x2p, y2p = min(W, x2 + pad), min(H, y2 + pad)
    roi = bgr_img[y1p:y2p, x1p:x2p]
    if roi.size == 0:
        return 255
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    return float(np.median(gray))


def draw_text_with_shadow(
    pil_rgba: Image.Image,
    xy: Tuple[int, int],
    lines: List[str],
    font: ImageFont.FreeTypeFont,
    fill_rgba: Tuple[int, int, int, int],
    shadow_rgba: Tuple[int, int, int, int],
    shadow_blur: int = 2,
    shadow_offset: Tuple[int, int] = (1, 1),
    align: str = "center",
    line_spacing: float = 1.1
) -> Tuple[Image.Image, Tuple[int, int]]:
    """
    Draw text with shadow effect on a PIL RGBA image.
    
    Returns:
        Tuple of (composited_image, (block_width, block_height)).
    """
    base = pil_rgba
    x, y = xy
    tmp = Image.new("RGBA", base.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(tmp)
    
    line_h = d.textbbox((0, 0), "Ag", font=font)[3]
    widths = [d.textbbox((0, 0), li, font=font)[2] for li in lines]
    block_w = max(widths) if widths else 0
    step_y = int(line_h * line_spacing)
    
    yy = y
    for li, w in zip(lines, widths):
        if align == "center":
            xx = x + (block_w - w) // 2
        elif align == "left":
            xx = x
        else:
            xx = x + (block_w - w)
        d.text((xx + shadow_offset[0], yy + shadow_offset[1]), li, font=font, fill=shadow_rgba)
        yy += step_y
    
    tmp = tmp.filter(ImageFilter.GaussianBlur(radius=shadow_blur))
    
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
    
    block_h = int(line_h * len(lines) * line_spacing)
    return Image.alpha_composite(base, tmp), (block_w, block_h)


def overlay_english_text(
    inpainted_bgr: np.ndarray,
    ch_items: List[Dict],
    font_path: str = FONT_PATH
) -> Image.Image:
    """
    Overlay translated English text on inpainted image.
    
    Args:
        inpainted_bgr: Inpainted BGR image.
        ch_items: List of Chinese items with 'en' translations.
        font_path: Path to the font file.
        
    Returns:
        PIL RGBA image with overlaid text.
    """
    H, W = inpainted_bgr.shape[:2]
    rgb = cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb).convert("RGBA")
    draw = ImageDraw.Draw(pil_img)
    
    for it in ch_items:
        en = (it.get("en") or "").strip()
        if not en or it.get("box") is None:
            continue
        
        x1, y1, x2, y2 = map(int, it["box"])
        x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, W, H)
        bw, bh = (x2 - x1), (y2 - y1)
        
        if bw < MIN_BOX_W or bh < MIN_BOX_H:
            continue
        
        target_w = max(1, bw - 2 * PAD_IN)
        target_h = max(1, bh - 2 * PAD_IN)
        
        result = fit_font_single_line(
            draw, en, target_w, target_h, font_path,
            min_size=MIN_FONT_SIZE, max_size=MAX_FONT_SIZE, step=FONT_SIZE_STEP
        )
        
        if result:
            font, lines, font_sz = result
            if font_sz <= SPLIT_THRESHOLD:
                multi = fit_font_multi_line(
                    draw, en, target_w, target_h, font_path,
                    max_lines=MAX_LINES_FALLBACK,
                    min_size=MIN_FONT_SIZE, max_size=MAX_FONT_SIZE, step=FONT_SIZE_STEP,
                    line_spacing=LINE_SPACING
                )
                if multi and multi[2] > font_sz:
                    font, lines, font_sz = multi
        else:
            multi = fit_font_multi_line(
                draw, en, target_w, target_h, font_path,
                max_lines=MAX_LINES_FALLBACK,
                min_size=MIN_FONT_SIZE, max_size=MAX_FONT_SIZE, step=FONT_SIZE_STEP,
                line_spacing=LINE_SPACING
            )
            if multi:
                font, lines, font_sz = multi
            else:
                continue
        
        if not lines:
            continue
        
        luma = sample_bg_luma(inpainted_bgr, x1, y1, x2, y2, pad=8)
        if luma > 160:
            fill = (15, 15, 15, 255)
            shadow = (255, 255, 255, 140)
        else:
            fill = (245, 245, 245, 255)
            shadow = (0, 0, 0, 160)
        
        widths = [draw.textbbox((0, 0), li, font=font)[2] for li in lines]
        line_h = draw.textbbox((0, 0), "Ag", font=font)[3]
        block_w = max(widths) if widths else 0
        block_h = int(line_h * len(lines) * LINE_SPACING)
        
        tx = x1 + (bw - block_w) // 2
        ty = y1 + (bh - block_h) // 2
        
        pil_img, _ = draw_text_with_shadow(
            pil_img, (tx, ty), lines, font,
            fill_rgba=fill, shadow_rgba=shadow,
            shadow_blur=SHADOW_BLUR, shadow_offset=SHADOW_OFFSET,
            align="center", line_spacing=LINE_SPACING,
        )
    
    return pil_img
