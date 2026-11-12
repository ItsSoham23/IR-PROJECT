"""Central source routing and validated PDF/HTML extraction."""

import re
import urllib.parse
import urllib.request

import pymupdf
from bs4 import BeautifulSoup

from src.preprocess import clean_bitscan_pages

USER_AGENT = "Mozilla/5.0 (Academic IR corpus ingestion)"
MIN_WORDS = 50
LONG_DOCUMENT_WORDS = 5000
EXTRACTION_VERSION = "corpus-v2.1"


class ExtractionError(ValueError):
    def __init__(self, message, word_count=0):
        super().__init__(message)
        self.word_count = word_count


def source_kind(url, content_type="", payload=b""):
    """URL hints plus response inspection; PDF detection precedes HTML decoding."""
    parsed = urllib.parse.urlparse(url)
    host = (parsed.hostname or "").lower()
    if host in {"drive.google.com", "docs.google.com"}:
        return "drive"
    if (parsed.path.lower().endswith(".pdf")
            or "application/pdf" in content_type.lower()
            or b"%PDF-" in payload[:1024]):
        return "pdf"
    if host == "wikinews.org" or host.endswith(".wikinews.org"):
        return "wikinews"
    return "html"


def download(url):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=45) as response:
        return response.read(), response.headers.get("Content-Type", ""), response.geturl()


def validate_extraction(text, kind="html", min_words=MIN_WORDS):
    words = text.split()
    count = len(words)
    reason = None
    if not count:
        reason = "empty extraction"
    elif count < min_words:
        reason = f"extremely short extraction ({count} words; minimum {min_words})"
    elif (text.lstrip().startswith("%PDF-")
          or sum(any(c.isalpha() for c in word) for word in words) / count < 0.6
          or (kind == "pdf"
              and sum(sum(c.isalpha() for c in word) >= 2 for word in words) / count < 0.3)
          or text.count("\ufffd") / max(len(text), 1) > 0.01):
        reason = "probable failed extraction (binary or garbled text)"
    if reason:
        if kind == "pdf":
            reason += "; probable failed PDF extraction: inspect text layer or run OCR"
        raise ExtractionError(reason, count)
    return {"word_count": count, "warnings": (
        ["unusually long extraction"] if count > LONG_DOCUMENT_WORDS else []
    )}


def extract_pdf_text(payload, bitscan=False):
    if b"%PDF-" not in payload[:1024]:
        raise ExtractionError("Expected PDF bytes, received a non-PDF response (login/error/viewer page)")
    try:
        with pymupdf.open(stream=payload, filetype="pdf") as document:
            if document.needs_pass:
                raise ExtractionError("Encrypted PDF: provide an unlocked source")
            # Keep each text block intact: line-wise sorting can interleave
            # adjacent magazine columns and destroy sentence coherence.
            pages = ["\n\n".join(block[4] for block in page.get_text("blocks", sort=True)
                                if block[6] == 0) for page in document]
    except ExtractionError:
        raise
    except Exception as exc:
        raise ExtractionError(f"Cannot parse PDF: {exc}") from exc
    # A sparse text layer over mostly scanned pages needs review, not acceptance.
    usable_pages = sum(len(page.split()) >= 20 for page in pages)
    if not pages or usable_pages < len(pages) / 2:
        raise ExtractionError(
            "Probable failed PDF extraction: most pages have no usable text; run OCR",
            sum(len(page.split()) for page in pages),
        )
    text = clean_bitscan_pages(pages) if bitscan else "\n\n".join(p.strip() for p in pages).strip()
    validate_extraction(text, "pdf")
    return text


def fetch_pdf_text(url, bitscan=False):
    if source_kind(url) == "drive":
        match = re.search(r"/file/d/([A-Za-z0-9_-]+)", url)
        file_id = match.group(1) if match else urllib.parse.parse_qs(
            urllib.parse.urlparse(url).query).get("id", [None])[0]
        if not file_id:
            raise ExtractionError("Google Drive URL has no file ID; supply a public PDF download URL")
        url = "https://drive.google.com/uc?export=download&id=" + file_id
    try:
        payload, _, _ = download(url)
        return extract_pdf_text(payload, bitscan=bitscan)
    except Exception as exc:
        if "drive.google.com" in url:
            raise ExtractionError(
                f"Google Drive PDF download failed: {exc}. Enable public download "
                "or replace the viewer URL with a direct PDF URL."
            ) from exc
        raise


def extract_html_text(payload, wikinews=False):
    # Defense in depth for direct callers as well as the central router.
    if b"%PDF-" in payload[:1024]:
        raise ExtractionError("PDF bytes cannot be decoded as HTML")
    soup = BeautifulSoup(payload, "html.parser")
    if wikinews:
        # Observed dynamic related-story boxes, distinct from article bullet lists.
        for tag in soup.select(".infobox.noprint"):
            tag.decompose()
    for tag in soup(["script", "style", "nav", "header", "footer", "iframe", "noscript", "table"]):
        tag.decompose()
    for tag in list(soup.find_all(True)):
        if wikinews or tag.attrs is None or tag.name not in {"div", "aside", "section"}:
            continue
        identity = f"{tag.get('id', '')} {' '.join(tag.get('class', []))}".lower()
        if any(term in identity for term in ("menu", "navbar", "sidebar", "breadcrumb", "footer", "header")):
            tag.decompose()
    node = soup.find("div", id="mw-content-text") if wikinews else (
        soup.find("article") or soup.find("div", class_="entry-content")
        or soup.find("div", id="content") or soup.find("div", class_="content") or soup.find("main"))
    node = node or soup.find("body") or soup
    tags = ["h2", "h3", "h4", "p", "li"] if wikinews else ["h1", "h2", "h3", "h4", "p", "li"]
    lines = []
    for element in node.find_all(tags):
        line = element.get_text(" ", strip=True)
        if wikinews and element.name in {"h2", "h3", "h4"}:
            marks = "=" * int(element.name[1])
            lines.append(f"{marks} {line.removesuffix('[edit]').strip()} {marks}")
        elif len(line) > 10:
            lines.append(line)
    text = "\n\n".join(dict.fromkeys(lines))
    if "Upload Screenshot" in text:
        text = text.split("Upload Screenshot", 1)[1]
    if "Latest news" in text:
        text = text.split("Latest news", 1)[0]
    return text.strip()


def fetch_source_text(url, bitscan=False):
    kind = source_kind(url)
    if kind in {"pdf", "drive"}:
        return fetch_pdf_text(url, bitscan=bitscan), "pdf"
    if kind == "wikinews":
        # Keep the MediaWiki/API path and its article HTML fallback.
        from src.ingest_corpus import fetch_wikinews_text
        return fetch_wikinews_text(url), "wikinews"
    payload, content_type, final_url = download(url)
    kind = source_kind(final_url, content_type, payload)
    if kind == "drive":
        return fetch_pdf_text(final_url, bitscan=bitscan), "pdf"
    if kind == "pdf":
        return extract_pdf_text(payload, bitscan=bitscan), "pdf"
    return extract_html_text(payload), "html"
