"""
Corpus Ingestion and Web Scraping Script.

Downloads metadata from the group's Google Spreadsheet:
https://docs.google.com/spreadsheets/d/1D2YwcC_XE9SZRnGOFqxx1-l5SPYDg5UfdqJ2rno8eeU/edit?gid=0

Scrapes:
- Wikinews articles via MediaWiki API (clean extracts, CC BY compliant)
- BITS Pilani news & announcements via BeautifulSoup HTML extraction

Saves:
- Raw text files to `data/raw/<category>/<document_id>.txt` and `data/raw/<document_id>.txt`
- Full structured corpus database to `data/corpus.sqlite`
- Updated metadata tracking table to `data/metadata.csv`
"""

import os
import sys
import time
import csv
import io
import json
import sqlite3
import urllib.request
import urllib.parse
from typing import Dict, List, Optional, Tuple
from bs4 import BeautifulSoup


SHEET_ID = "1D2YwcC_XE9SZRnGOFqxx1-l5SPYDg5UfdqJ2rno8eeU"
EXPORT_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid=0"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 (Academic IR Research; contact: sohamdeshmukh23@gmail.com)"


def fetch_sheet_metadata(export_url: str = EXPORT_URL) -> List[Dict[str, str]]:
    """Fetch the latest metadata sheet as a list of dicts."""
    req = urllib.request.Request(export_url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=15) as resp:
        content = resp.read().decode("utf-8")
    reader = csv.DictReader(io.StringIO(content))
    return list(reader)


def fetch_wikinews_text(url: str) -> str:
    """
    Fetch clean article text from Wikinews using MediaWiki action=query&prop=extracts.
    Fallback to HTML parsing if API does not return content.
    """
    try:
        # Extract title from URL
        parsed = urllib.parse.urlparse(url)
        path = parsed.path
        if "/wiki/" in path:
            title_part = path.split("/wiki/")[-1]
            title_unquoted = urllib.parse.unquote(title_part)
        else:
            title_unquoted = path.strip("/")

        api_url = (
            f"https://en.wikinews.org/w/api.php?action=query&prop=extracts"
            f"&explaintext=true&titles={urllib.parse.quote(title_unquoted)}&format=json"
        )
        req = urllib.request.Request(api_url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=12) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            pages = data.get("query", {}).get("pages", {})
            for pid, pdata in pages.items():
                if pid != "-1" and "extract" in pdata:
                    extract = pdata["extract"].strip()
                    if len(extract) > 100:
                        return extract
    except Exception as e:
        pass

    # Fallback: scrape directly from article webpage
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=12) as resp:
        html = resp.read().decode("utf-8", errors="ignore")

    soup = BeautifulSoup(html, "html.parser")
    content_div = soup.find("div", id="mw-content-text")
    if not content_div:
        content_div = soup.find("body")

    for tag in content_div(["script", "style", "table", "nav", "header", "footer"]):
        tag.decompose()

    paragraphs = [p.get_text().strip() for p in content_div.find_all("p")]
    return "\n\n".join([p for p in paragraphs if len(p) > 20])


def fetch_bits_text(url: str) -> str:
    """
    Fetch and clean announcement text from BITS Pilani website.
    Handles unverified SSL and 404 fallback to Wayback Machine.
    """
    import ssl
    ctx = ssl._create_unverified_context()
    html = None

    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, context=ctx, timeout=15) as resp:
            html = resp.read().decode("utf-8", errors="ignore")
    except urllib.error.HTTPError as e:
        if e.code == 404:
            # Query Wayback Machine for archived snapshot
            try:
                archive_api = f"https://archive.org/wayback/available?url={urllib.parse.quote(url)}"
                with urllib.request.urlopen(archive_api, timeout=10) as a_resp:
                    res = json.loads(a_resp.read().decode("utf-8"))
                    snap_url = res.get("archived_snapshots", {}).get("closest", {}).get("url")
                    if snap_url:
                        a_req = urllib.request.Request(snap_url, headers={"User-Agent": USER_AGENT})
                        with urllib.request.urlopen(a_req, context=ctx, timeout=15) as snap_resp:
                            html = snap_resp.read().decode("utf-8", errors="ignore")
            except Exception:
                pass
        if not html:
            raise

    soup = BeautifulSoup(html, "html.parser")

    # Remove script, style, navigation, footer elements
    for tag in soup(["script", "style", "nav", "header", "footer", "iframe", "noscript"]):
        tag.decompose()

    # Look for main content container
    main_node = (
        soup.find("main")
        or soup.find("article")
        or soup.find("div", class_="entry-content")
        or soup.find("div", class_="content")
        or soup.find("div", id="content")
    )
    if not main_node:
        main_node = soup.find("body")

    lines = []
    for elem in main_node.find_all(["h1", "h2", "h3", "h4", "p", "li"]):
        txt = elem.get_text().strip()
        if len(txt) > 10:
            lines.append(txt)

    full_text = "\n\n".join(lines)
    if not full_text.strip():
        full_text = main_node.get_text(separator="\n", strip=True)

    return full_text


def setup_database(db_path: str = "data/corpus.sqlite"):
    """Initialize SQLite database for corpus storage."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            document_id TEXT PRIMARY KEY,
            category TEXT,
            title TEXT,
            source TEXT,
            url TEXT,
            publication_date TEXT,
            word_count INTEGER,
            file_path TEXT,
            raw_text TEXT
        )
        """
    )
    conn.commit()
    conn.close()


def save_to_database(db_path: str, doc_data: Dict):
    """Insert or replace document record in SQLite."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        INSERT OR REPLACE INTO documents 
        (document_id, category, title, source, url, publication_date, word_count, file_path, raw_text)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            doc_data["document_id"],
            doc_data["category"],
            doc_data["title"],
            doc_data["source"],
            doc_data["url"],
            doc_data.get("publication_date", ""),
            doc_data["word_count"],
            doc_data["file_path"],
            doc_data["raw_text"],
        ),
    )
    conn.commit()
    conn.close()


def ingest_corpus(raw_dir: str = "data/raw", db_path: str = "data/corpus.sqlite"):
    """Main ingestion runner."""
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    setup_database(db_path)

    print("Fetching metadata sheet from Google Docs...")
    metadata_rows = fetch_sheet_metadata()
    print(f"Loaded metadata for {len(metadata_rows)} documents.")

    # Save a copy of the metadata sheet
    meta_csv_path = "data/metadata.csv"
    with open(meta_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=metadata_rows[0].keys())
        writer.writeheader()
        writer.writerows(metadata_rows)
    print(f"Saved metadata snapshot to {meta_csv_path}")

    success_count = 0
    fail_count = 0

    for idx, row in enumerate(metadata_rows, start=1):
        doc_id = row.get("document_id", "").strip()
        cat = row.get("category", "").strip()
        title = row.get("title", "").strip()
        source = row.get("source", "").strip()
        url = row.get("url", "").strip()

        if not doc_id or not url:
            continue

        cat_dir = os.path.join(raw_dir, cat)
        os.makedirs(cat_dir, exist_ok=True)

        flat_path = os.path.join(raw_dir, f"{doc_id}.txt")
        cat_path = os.path.join(cat_dir, f"{doc_id}.txt")

        # Skip if already downloaded and non-empty
        if os.path.exists(flat_path) and os.path.getsize(flat_path) > 200:
            print(f"[{idx}/{len(metadata_rows)}] {doc_id} already exists on disk. Skipping fetch.")
            success_count += 1
            continue

        print(f"[{idx}/{len(metadata_rows)}] Fetching {doc_id} ({cat}): {title[:45]}...")

        try:
            if "wikinews.org" in url.lower():
                article_text = fetch_wikinews_text(url)
            else:
                article_text = fetch_bits_text(url)

            word_count = len(article_text.split())

            if word_count < 30:
                print(f"  WARNING: Short text ({word_count} words) for {doc_id}")

            # Compose formatted raw file content
            header = [
                f"# DOCUMENT_ID: {doc_id}",
                f"# CATEGORY: {cat}",
                f"# TITLE: {title}",
                f"# SOURCE: {source}",
                f"# URL: {url}",
                f"# WORD_COUNT: {word_count}",
                "=" * 60,
                "",
            ]
            full_content = "\n".join(header) + article_text + "\n"

            # Write to both flat raw/ and category subfolder
            with open(flat_path, "w", encoding="utf-8") as f:
                f.write(full_content)
            with open(cat_path, "w", encoding="utf-8") as f:
                f.write(full_content)

            # Insert into database
            doc_record = {
                "document_id": doc_id,
                "category": cat,
                "title": title,
                "source": source,
                "url": url,
                "publication_date": row.get("publication_date", ""),
                "word_count": word_count,
                "file_path": flat_path,
                "raw_text": article_text,
            }
            save_to_database(db_path, doc_record)

            success_count += 1
            # Respectful rate limiting
            time.sleep(0.3)

        except Exception as err:
            print(f"  ERROR fetching {doc_id} ({url}): {err}")
            fail_count += 1

    print("\n" + "=" * 60)
    print(f"INGESTION COMPLETE: {success_count} succeeded, {fail_count} failed.")
    print(f"Documents written to: {raw_dir}")
    print(f"Database populated at: {db_path}")
    print("=" * 60)


if __name__ == "__main__":
    ingest_corpus()
