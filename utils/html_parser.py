"""
HTML parsing utilities for extracting and replacing image URLs.
"""
import re
from typing import List, Dict, Tuple
from bs4 import BeautifulSoup, formatter


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
