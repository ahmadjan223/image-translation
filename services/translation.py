"""
Translation service using Google Gemini.
"""
import json
from typing import List, Dict

from google import genai

from config import GEMINI_API_KEY, TRANSLATION_SYSTEM_PROMPT

# Initialize Gemini client
_gemini_client = None


def get_gemini_client() -> genai.Client:
    """Get or create the Gemini client."""
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    return _gemini_client


def translate_items_gemini(
    ch_items: List[Dict],
    model: str = "models/gemini-2.5-pro"
) -> List[str]:
    """
    Translate Chinese items to English using Gemini.
    
    Args:
        ch_items: List of dictionaries containing Chinese text items.
        model: Gemini model to use for translation.
        
    Returns:
        List of English translations corresponding to each Chinese item.
    """
    if not ch_items:
        return []
    
    items = []
    for i, it in enumerate(ch_items):
        cn = (it.get("text") or "").strip()
        max_chars = max(6, int(len(cn) * 1.35))
        items.append({"i": i, "cn": cn, "max_chars": max_chars})
    
    payload = {"items": items}
    
    client = get_gemini_client()
    resp = client.models.generate_content(
        model=model,
        contents=json.dumps(payload, ensure_ascii=False),
        config={
            "system_instruction": TRANSLATION_SYSTEM_PROMPT,
            "temperature": 0.2,
            "response_mime_type": "application/json",
        },
    )
    
    arr = json.loads((resp.text or "").strip())
    mp = {int(o["i"]): (o.get("en") or "").strip() for o in arr}
    return [mp.get(i, "") for i in range(len(items))]


def translate_batch_all(
    html_items: List[Dict],
    image_items: Dict[str, List[Dict]],
    model: str = "models/gemini-2.5-pro"
) -> tuple[List[str], Dict[str, List[str]]]:
    """
    Translate ALL Chinese text (HTML + all images) in a SINGLE API call.
    
    This ensures translation consistency across the entire product description:
    - Same Chinese word → same English translation
    - LLM sees full product context for better understanding
    - Reduces API calls from N+1 to 1 (where N = number of images)
    
    Args:
        html_items: List of Chinese text items from HTML
        image_items: Dict mapping image_index -> list of Chinese text items
        model: Gemini model to use for translation
        
    Returns:
        Tuple of (html_translations, image_translations_dict)
        - html_translations: List of English translations for HTML items
        - image_translations_dict: Dict mapping image_index -> list of translations
    """
    # Prepare structured payload with all Chinese text
    payload = {
        "html_text": [],
        "images": {}
    }
    
    # Add HTML text items
    for i, it in enumerate(html_items):
        cn = (it.get("text") or "").strip()
        max_chars = max(6, int(len(cn) * 1.35))
        payload["html_text"].append({"i": i, "cn": cn, "max_chars": max_chars})
    
    # Add image text items
    for img_idx, items in image_items.items():
        payload["images"][img_idx] = []
        for i, it in enumerate(items):
            cn = (it.get("text") or "").strip()
            max_chars = max(6, int(len(cn) * 1.35))
            payload["images"][img_idx].append({"i": i, "cn": cn, "max_chars": max_chars})
    
    # Enhanced system prompt for consistency
    consistency_prompt = """You are translating product descriptions from Chinese to English.

CRITICAL REQUIREMENTS:
1. CONSISTENCY: If the same Chinese word/phrase appears multiple times (in HTML text or images), translate it THE SAME WAY every time
2. CONTEXT: Examine ALL Chinese text to understand the product before translating
3. BREVITY: Keep translations concise (en length <= max_chars)
   - Use SHORT synonyms and compact phrases (e.g., "Slim Fit" not "Slender Body Type")
   - Prefer single words over phrases when possible (e.g., "Size" not "Product Size")
   - Use abbreviations for common terms (e.g., "L" not "Large", "XL" not "Extra Large")
4. NATURAL: Use natural English that fits in UI elements
5. TECHNICAL: Preserve technical terms, model numbers, specifications

You will receive:
- html_text: Array of Chinese text from HTML description
- images: Object with image indices, each containing array of Chinese text from that image

Return JSON with same structure:
{
  "html_text": [{"i": index, "en": "translation"}, ...],
  "images": {
    "0": [{"i": index, "en": "translation"}, ...],
    "1": [{"i": index, "en": "translation"}, ...],
    ...
  }
}

Remember: Same Chinese → Same English throughout! Keep it SHORT!"""
    
    client = get_gemini_client()
    resp = client.models.generate_content(
        model=model,
        contents=json.dumps(payload, ensure_ascii=False),
        config={
            "system_instruction": consistency_prompt,
            "temperature": 0.2,
            "response_mime_type": "application/json",
        },
    )
    
    result = json.loads((resp.text or "").strip())
    
    # Parse HTML translations
    html_translations = []
    if result.get("html_text"):
        html_map = {int(o["i"]): (o.get("en") or "").strip() for o in result["html_text"]}
        html_translations = [html_map.get(i, "") for i in range(len(html_items))]
    
    # Parse image translations
    image_translations = {}
    if result.get("images"):
        for img_idx, items in result["images"].items():
            item_map = {int(o["i"]): (o.get("en") or "").strip() for o in items}
            image_translations[img_idx] = [item_map.get(i, "") for i in range(len(image_items.get(img_idx, [])))]
    
    return html_translations, image_translations



