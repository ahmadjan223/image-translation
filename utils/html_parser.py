"""
HTML parsing utilities for extracting and replacing image URLs and Chinese text.
"""
import re
from typing import List, Dict, Tuple, Any
from bs4 import BeautifulSoup, NavigableString, formatter


def extract_image_urls(html_content: str) -> List[str]:
    """
    Extract all image URLs from HTML content.
    
    Args:
        html_content: HTML string containing img tags.
        
    Returns:
        List of unique image URLs found in the HTML.
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    img_tags = soup.find_all('img')
    
    urls = []
    seen = set()
    
    for img in img_tags:
        src = img.get('src')
        if src and src not in seen:
            seen.add(src)
            urls.append(src)
    
    return urls


def replace_image_urls(html_content: str, url_mapping: Dict[str, str]) -> str:
    """
    Replace image URLs in HTML with new paths using simple string replacement.
    Preserves the original HTML structure and formatting.
    
    Args:
        html_content: Original HTML string.
        url_mapping: Dictionary mapping original URLs to new local paths.
        
    Returns:
        HTML string with replaced image URLs.
    """
    result = html_content
    for original_url, new_path in url_mapping.items():
        result = result.replace(original_url, new_path)
    return result


def get_image_urls_with_positions(html_content: str) -> List[Tuple[str, int]]:
    """
    Extract image URLs with their positions in HTML.
    
    Args:
        html_content: HTML string containing img tags.
        
    Returns:
        List of tuples (url, index) for each image.
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    img_tags = soup.find_all('img')
    
    results = []
    for idx, img in enumerate(img_tags):
        src = img.get('src')
        if src:
            results.append((src, idx))
    
    return results


def extract_chinese_text_from_html(html_content: str, cjk_pattern) -> Tuple[List[Dict[str, Any]], BeautifulSoup]:
    """
    Extract Chinese text from HTML, preserving position for replacement.
    
    Args:
        html_content: HTML string to parse
        cjk_pattern: Compiled regex pattern for detecting Chinese characters (CJK_RE from config)
        
    Returns:
        Tuple of (chinese_items_list, BeautifulSoup_object)
        chinese_items: [{"original": "购买须知", "marker": "{{TRANS_0}}", "index": 0, "element": NavigableString}, ...]
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    chinese_items = []
    
    # Skip these tags - don't translate content inside
    skip_tags = {'script', 'style', 'img', 'map', 'area', 'code', 'pre'}
    
    # Find all text nodes (NavigableString objects)
    text_index = 0
    for element in soup.find_all(string=True):
        # Skip if parent tag is in skip list
        if element.parent and element.parent.name in skip_tags:
            continue
        
        text = str(element).strip()
        
        # Skip empty strings or strings without Chinese
        if not text or not cjk_pattern.search(text):
            continue
        
        # Create unique marker for this text segment
        marker = f"{{{{TRANS_{text_index}}}}}"
        
        chinese_items.append({
            "original": text,
            "marker": marker,
            "index": text_index,
            "element": element  # Keep reference for in-place replacement
        })
        
        text_index += 1
    
    return chinese_items, soup


def replace_chinese_with_markers(soup: BeautifulSoup, chinese_items: List[Dict[str, Any]]) -> str:
    """
    Replace Chinese text with markers in BeautifulSoup object.
    
    Args:
        soup: BeautifulSoup object
        chinese_items: List of Chinese text items with element references
        
    Returns:
        HTML string with Chinese text replaced by markers
    """
    for item in chinese_items:
        # Replace the NavigableString element with the marker
        item["element"].replace_with(item["marker"])
    
    return str(soup)


def replace_markers_with_translations(html: str, chinese_items: List[Dict[str, Any]], translations: List[str]) -> str:
    """
    Replace markers with English translations.
    
    Args:
        html: HTML string with markers
        chinese_items: Original Chinese text items with markers
        translations: List of English translations (same order as chinese_items)
        
    Returns:
        HTML string with markers replaced by translations
    """
    result = html
    for item, translation in zip(chinese_items, translations):
        result = result.replace(item["marker"], translation)
    
    return result
