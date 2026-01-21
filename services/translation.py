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
    model: str = "models/gemini-flash-lite-latest"
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
