"""
Test script for HTML Chinese text extraction and translation.
"""
from utils.html_parser import (
    extract_chinese_text_from_html,
    replace_chinese_with_markers,
    replace_markers_with_translations
)
from config import CJK_RE

# Sample HTML with Chinese text (from your example)
test_html = """
<div id="offer-template-0"></div>
<p><span style="font-size: 12.0pt;color: #ff00ff;">购买须知：</span></p>
<p><span style="font-size: 12.0pt;color: #ff00ff;">1.本店铺所有产品透明胶袋包装，标价均不配表盒，包装盒一整套直接网站里面拍</span></p>
<p><span style="font-size: 12.0pt;color: #ff00ff;">2.每个款式都可以游泳30-50米防水！但不接受热水洗澡！</span></p>
<p><span style="font-size: 12.0pt;color: #ff00ff;">3.产品质量问题，需买家配合累积10只以上寄回对换-同款产品处理。</span></p>
<img src="https://example.com/image.jpg" alt="undefined"/>
"""

print("=" * 80)
print("Testing HTML Chinese Text Extraction")
print("=" * 80)

# Step 1: Extract Chinese text
chinese_items, soup = extract_chinese_text_from_html(test_html, CJK_RE)

print(f"\n✅ Found {len(chinese_items)} Chinese text segments:\n")
for item in chinese_items:
    print(f"  [{item['index']}] {item['marker']}: {item['original'][:50]}...")

# Step 2: Replace with markers
html_with_markers = replace_chinese_with_markers(soup, chinese_items)
print(f"\n📝 HTML with markers:\n{html_with_markers[:300]}...\n")

# Step 3: Simulate translations (in real code, this comes from Gemini)
mock_translations = [
    "Purchase Notice:",
    "1. All products in this store are packaged in transparent bags. Prices do not include watch boxes. Complete packaging can be purchased directly from the website.",
    "2. Each style is waterproof for swimming at 30-50 meters! But hot water bathing is not accepted!",
    "3. For product quality issues, buyers need to cooperate to accumulate more than 10 pieces and send them back for exchange - same product processing."
]

# Step 4: Replace markers with translations
translated_html = replace_markers_with_translations(html_with_markers, chinese_items, mock_translations)

print(f"🌍 Translated HTML:\n{translated_html}\n")

print("=" * 80)
print("✅ Test Complete!")
print("=" * 80)
