
# https://api.markaz.app/products/v2/details/679766833879
# Method : GET 
# Other Ids
# 557398500350
# 842802486656
# 718989351272
# 670660926576
# 683446453137

from flask import Flask, render_template_string, request, Response, abort
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, quote
import re

API_URL = "https://api.markaz.app/products/v2/details/683446453137"

app = Flask(__name__)


def is_absolute_url(u: str) -> bool:
    try:
        p = urlparse(u)
        return p.scheme in ("http", "https") and bool(p.netloc)
    except Exception:
        return False


def basename(url: str) -> str:
    url = (url or "").split("?")[0].split("#")[0]
    return url.rstrip("/").split("/")[-1].strip()


def collect_urls(obj, out: list):
    if isinstance(obj, dict):
        for v in obj.values():
            collect_urls(v, out)
    elif isinstance(obj, list):
        for v in obj:
            collect_urls(v, out)
    elif isinstance(obj, str):
        s = obj.strip()
        if s.startswith("//"):
            out.append("https:" + s)
        elif s.startswith("http://") or s.startswith("https://"):
            out.append(s)


def build_image_map(json_data: dict) -> dict:
    urls = []
    collect_urls(json_data, urls)
    img_map = {}
    for u in urls:
        b = basename(u)
        if b:
            img_map.setdefault(b, u)
    return img_map


def normalize_src(src: str, img_map: dict) -> str:
    if not src:
        return src

    s = src.strip()
    if s.startswith("//"):
        return "https:" + s

    if is_absolute_url(s):
        return s

    b = basename(s)
    if b in img_map:
        return img_map[b]

    return s


def proxify_url(real_url: str) -> str:
    return f"/img?u={quote(real_url, safe='')}"


def enhance_html(raw_html: str, img_map: dict) -> str:
    soup = BeautifulSoup(raw_html or "", "html.parser")

    for img in soup.find_all("img"):
        src = img.get("src") or img.get("data-src") or img.get("data-original") or ""
        real_src = normalize_src(src, img_map)

        title = img.get("title") or img.get("alt")
        if not title:
            title = basename(real_src) or real_src

        img["title"] = title
        if not img.get("alt"):
            img["alt"] = title

        # load image through our proxy
        img["src"] = proxify_url(real_src)

        # click opens the proxied image too
        link_href = proxify_url(real_src)

        if img.parent and img.parent.name == "a":
            img.parent["href"] = link_href
            img.parent["target"] = "_blank"
            img.parent["rel"] = "noopener"
            if not img.parent.get("title"):
                img.parent["title"] = title
        else:
            a = soup.new_tag("a", href=link_href, target="_blank", rel="noopener", title=title)
            img.wrap(a)

    return str(soup)


def referers_for(url: str) -> list[str]:
    host = urlparse(url).netloc.lower()

    if host.endswith("alicdn.com") or "1688" in host or "alibaba" in host:
        return [
            "https://detail.1688.com/",
            "https://www.1688.com/",
            "https://m.1688.com/",
            "https://www.alibaba.com/",
            "https://cbu01.alicdn.com/",
        ]

    return ["https://api.markaz.app/"]


@app.route("/")
def product_description():
    r = requests.get(API_URL, timeout=20)
    r.raise_for_status()
    data = r.json()

    raw_html = data.get("description", "")
    img_map = build_image_map(data)
    safe_html = enhance_html(raw_html, img_map)

    template = """
    <!doctype html>
    <html>
      <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>Product Description</title>
        <style>
          body { font-family: system-ui, Arial, sans-serif; margin: 0; padding: 16px; }
          .container { max-width: 980px; margin: 0 auto; }
          img { max-width: 100%; height: auto; display: block; }
          a { text-decoration: none; }
          a:hover img { outline: 2px solid rgba(0,0,0,0.15); outline-offset: 2px; }
          .hint { font-size: 13px; opacity: 0.7; margin-bottom: 12px; }
        </style>
      </head>
      <body>
        <div class="container">
          <div class="hint">Tip: Hover images to see titles. Click an image to open it.</div>
          {{ html | safe }}
        </div>
      </body>
    </html>
    """
    return render_template_string(template, html=safe_html)


@app.route("/img")
def img_proxy():
    u = request.args.get("u", "")
    if not u:
        abort(400, "missing u")

    p = urlparse(u)
    if p.scheme not in ("http", "https") or not p.netloc:
        abort(400, "invalid url")

    ua = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )

    base_headers = {
        "User-Agent": ua,
        "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }

    last = None
    for ref in referers_for(u):
        headers = dict(base_headers)
        headers["Referer"] = ref
        headers["Origin"] = f"{urlparse(ref).scheme}://{urlparse(ref).netloc}"

        try:
            resp = requests.get(u, headers=headers, stream=True, timeout=30)
            last = resp
            if resp.status_code == 200:
                ct = resp.headers.get("Content-Type", "image/jpeg")
                return Response(resp.iter_content(64 * 1024), content_type=ct)
        except requests.RequestException:
            continue

    if last is not None:
        return Response(
            f"Upstream blocked (status={last.status_code}). URL: {u}\n",
            status=last.status_code,
            content_type="text/plain",
        )

    return Response("Upstream request failed.", status=502, content_type="text/plain")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True, use_reloader=False)
