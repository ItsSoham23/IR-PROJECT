"""Ingest versioned source text; never reuse a file based on its document ID alone."""

import argparse
import csv
import hashlib
import io
import json
import re
import sqlite3
import sys
import time
import urllib.request
import urllib.parse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.source_extract import (
    EXTRACTION_VERSION, USER_AGENT, ExtractionError, download,
    extract_html_text, fetch_pdf_text, fetch_source_text, source_kind,
    validate_extraction,
)
from src.preprocess import clean_wikinews_text

SHEET_ID = "1D2YwcC_XE9SZRnGOFqxx1-l5SPYDg5UfdqJ2rno8eeU"
EXPORT_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid=0"
METADATA_FIELDS = ("document_id", "category", "title", "source", "url", "publication_date")


def fetch_sheet_metadata(export_url=EXPORT_URL):
    payload, _, _ = download(export_url)
    return list(csv.DictReader(io.StringIO(payload.decode("utf-8-sig"))))


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
                        return clean_wikinews_text(extract)
    except Exception as e:
        pass

    payload, content_type, final_url = download(url)
    if source_kind(final_url, content_type, payload) in {"pdf", "drive"}:
        raise ExtractionError("Wikinews fallback returned a PDF or Drive viewer")
    return clean_wikinews_text(extract_html_text(payload, wikinews=True))


def fetch_bits_text(url):
    """Compatibility entry point; response routing also handles extensionless PDFs."""
    return fetch_source_text(url, bitscan=True)[0]


def text_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def metadata_hash(row):
    return text_hash(json.dumps(
        {key: row.get(key, "").strip() for key in METADATA_FIELDS}, sort_keys=True))


def validate_metadata(rows):
    if not rows:
        raise ValueError("Metadata is empty")
    seen = set()
    for row in rows:
        for key in METADATA_FIELDS:
            row[key] = (row.get(key) or "").strip()
        doc_id = row["document_id"]
        if not re.fullmatch(r"[A-Za-z0-9_-]+", doc_id) or doc_id in seen:
            raise ValueError(f"Invalid or duplicate document_id: {doc_id!r}")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", row["category"]):
            raise ValueError(f"Invalid category for {doc_id}")
        if urllib.parse.urlparse(row["url"]).scheme not in {"http", "https"}:
            raise ValueError(f"Missing/invalid source URL for {doc_id}")
        seen.add(doc_id)
    return rows


def setup_database(db_path="data/corpus.sqlite"):
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                document_id TEXT PRIMARY KEY, category TEXT, title TEXT,
                source TEXT, url TEXT, publication_date TEXT, word_count INTEGER,
                file_path TEXT, raw_text TEXT
            )
        """)
        cols = {row[1] for row in conn.execute("PRAGMA table_info(documents)")}
        if "extraction_version" not in cols and conn.execute(
                "SELECT count(*) FROM documents").fetchone()[0]:
            backup = Path(str(db_path) + ".legacy.bak")
            if not backup.exists():
                with sqlite3.connect(backup) as destination:
                    conn.backup(destination)
        for column in ("extraction_version", "metadata_hash", "text_hash", "source_type"):
            if column not in cols:
                conn.execute(f"ALTER TABLE documents ADD COLUMN {column} TEXT")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ingestion_state (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                metadata_json TEXT NOT NULL, complete INTEGER NOT NULL
            )
        """)


def save_to_database(db_path, doc_data):
    columns = (*METADATA_FIELDS, "word_count", "file_path", "raw_text",
               "extraction_version", "metadata_hash", "text_hash", "source_type")
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            f"INSERT OR REPLACE INTO documents ({','.join(columns)}) "
            f"VALUES ({','.join('?' for _ in columns)})",
            [doc_data.get(key, "") for key in columns],
        )


def cached_source_is_current(record, row):
    return bool(
        record and record.get("extraction_version") == EXTRACTION_VERSION
        and record.get("metadata_hash") == metadata_hash(row)
        and record.get("url") == row["url"]
        and record.get("text_hash") == text_hash(record.get("raw_text") or "")
    )


def write_raw_files(raw_dir, record):
    header = "\n".join(f"# {key.upper()}: {record[key]}" for key in
                       ("document_id", "category", "title", "source", "url", "word_count"))
    content = header + "\n" + "=" * 60 + "\n\n" + record["raw_text"] + "\n"
    for path in (Path(raw_dir) / f"{record['document_id']}.txt",
                 Path(raw_dir) / record["category"] / f"{record['document_id']}.txt"):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".txt.tmp")
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(path)


def word_count_summary(records):
    import numpy as np
    return {
        category: dict(zip(("n", "min", "median", "p90", "max"), (
            len(counts), int(min(counts)), float(np.median(counts)),
            float(np.percentile(counts, 90)), int(max(counts)))))
        for category in sorted({r["category"] for r in records})
        if (counts := [r["word_count"] for r in records if r["category"] == category])
    }


def write_source_audit(report, path):
    """Export reviewable per-source evidence without storing article text."""
    fields = ("document_id", "category", "url", "source_type", "ingestion_status",
              "extracted_word_count", "warning", "failure_reason", "extraction_version",
              "metadata_hash", "text_hash")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for doc in report["documents"]:
            writer.writerow({
                "document_id": doc["document_id"], "category": doc["category"],
                "url": doc["url"], "source_type": doc["source_type"],
                "ingestion_status": doc["status"], "extracted_word_count": doc["word_count"],
                "warning": "; ".join(doc.get("warnings", [])),
                "failure_reason": doc.get("error", ""),
                "extraction_version": report["extraction_version"],
                "metadata_hash": doc["metadata_hash"], "text_hash": doc.get("text_hash", ""),
            })


def ingest_corpus(raw_dir="data/raw", db_path="data/corpus.sqlite", *,
                  metadata_rows=None, force=False, delay=0.3, audit_path=None):
    """Resume verified sources; invalidate stale rows before any network operation.

    An incomplete run is unusable for model preparation. Raw files are evidence
    only; the database, source hashes and completed-run manifest are authoritative.
    """
    rows = validate_metadata(
        metadata_rows if metadata_rows is not None else fetch_sheet_metadata())
    setup_database(db_path)
    expected = {r["document_id"]: metadata_hash(r) for r in rows}
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cached = {r["document_id"]: dict(r) for r in conn.execute("SELECT * FROM documents")}
        conn.execute("INSERT OR REPLACE INTO ingestion_state VALUES (1, ?, 0)",
                     (json.dumps(expected, sort_keys=True),))
        # Invalidating all obsolete rows first makes interruption/fetch failure safe.
        for doc_id, record in cached.items():
            if (force or doc_id not in expected
                    or record["metadata_hash"] != expected[doc_id]
                    or record["extraction_version"] != EXTRACTION_VERSION
                    or record["text_hash"] != text_hash(record["raw_text"] or "")):
                conn.execute("DELETE FROM documents WHERE document_id = ?", (doc_id,))
    snapshot = Path(db_path).parent / "metadata.csv"
    with snapshot.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {"extraction_version": EXTRACTION_VERSION, "metadata_rows": len(rows), "documents": []}
    accepted = []
    for index, row in enumerate(rows, 1):
        doc_id = row["document_id"]
        record = cached.get(doc_id)
        kind = source_kind(row["url"])
        result = {"document_id": doc_id, "category": row["category"], "url": row["url"],
                  "source_type": "pdf" if kind == "drive" else kind,
                  "metadata_hash": metadata_hash(row)}
        try:
            reused = not force and cached_source_is_current(record, row)
            if reused:
                text, kind = record["raw_text"], record["source_type"]
            else:
                text, kind = fetch_source_text(row["url"], bitscan=row["category"] == "Campus")
                if delay:
                    time.sleep(delay)
            quality = validate_extraction(text, kind)
            record = {**row, **quality, "raw_text": text, "source_type": kind,
                      "file_path": str(Path(raw_dir) / f"{doc_id}.txt"),
                      "extraction_version": EXTRACTION_VERSION,
                      "metadata_hash": metadata_hash(row), "text_hash": text_hash(text)}
            write_raw_files(raw_dir, record)
            save_to_database(db_path, record)
            accepted.append(record)
            result.update(quality, status="cached" if reused else "accepted", source_type=kind,
                          text_hash=record["text_hash"])
        except Exception as exc:
            with sqlite3.connect(db_path) as conn:
                conn.execute("DELETE FROM documents WHERE document_id = ?", (doc_id,))
            result.update(status="failed", error=str(exc),
                          word_count=getattr(exc, "word_count", None))
        report["documents"].append(result)
        print(f"[{index}/{len(rows)}] {doc_id}: {result['status']}; "
              f"words={result['word_count']} {result.get('error', '')}", flush=True)
    report["word_counts"] = word_count_summary(accepted)
    report["accepted"] = len(accepted)
    report["failed"] = len(rows) - len(accepted)
    report["categories"] = {
        cat: {"accepted": sum(d["category"] == cat and d["status"] != "failed"
                             for d in report["documents"]),
              "failed": sum(d["category"] == cat and d["status"] == "failed"
                           for d in report["documents"])}
        for cat in sorted({r["category"] for r in rows})
    }
    report_path = Path(db_path).parent / "ingestion_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if audit_path:
        write_source_audit(report, audit_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute("UPDATE ingestion_state SET complete = ? WHERE id = 1",
                     (int(report["failed"] == 0),))
    print(json.dumps({k: v for k, v in report.items() if k != "documents"}, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Refetch even verified sources")
    parser.add_argument("--metadata", type=Path, help="Use a reviewed local CSV snapshot")
    parser.add_argument("--db-path", default="data/corpus.sqlite")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--audit-path", type=Path, help="Write a text-free per-source audit CSV")
    args = parser.parse_args()
    rows = None
    if args.metadata:
        with args.metadata.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    report = ingest_corpus(args.raw_dir, args.db_path, metadata_rows=rows, force=args.force,
                           audit_path=args.audit_path)
    sys.exit(1 if report["failed"] else 0)
