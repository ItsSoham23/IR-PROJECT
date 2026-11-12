import csv
import json
import sqlite3
from pathlib import Path

import pymupdf
import pytest

from src import ingest_corpus as ingest
from src import source_extract as extract
from src.dataset import load_corpus_from_sqlite
from src.preprocess import TextPreprocessor, clean_bitscan_pages


def article(word="students"):
    return (f"BITS Pilani {word} and faculty in Hyderabad and Goa study science. " * 10).strip()


def row(url="https://example.org/issue.pdf"):
    return {"document_id": "CAMPUS_001", "category": "Campus", "title": "Issue",
            "source": "BITScan", "url": url, "publication_date": ""}


def pdf_bytes(text=None, pages=2):
    with pymupdf.open() as doc:
        for number in range(pages):
            page = doc.new_page()
            if text:
                page.insert_textbox(pymupdf.Rect(40, 40, 550, 780),
                                    f"BITScan\n{text}\nPage {number + 1}", fontsize=10)
        return doc.tobytes()


@pytest.mark.parametrize("url,mime,payload,expected", [
    ("https://example.org/ISSUE.PDF?download=1", "", b"", "pdf"),
    ("https://example.org/download/4", "application/pdf", b"", "pdf"),
    ("https://example.org/download/4", "", b"%PDF-1.7", "pdf"),
    ("https://drive.google.com/file/d/abc/view", "", b"", "drive"),
    ("https://en.wikinews.org/wiki/Story", "", b"", "wikinews"),
    ("https://wikinews.org.evil.example/page", "", b"", "html"),
    ("https://example.org/news", "", b"", "html"),
])
def test_source_routing(url, mime, payload, expected):
    assert extract.source_kind(url, mime, payload) == expected


def test_pdf_extraction_and_extensionless_response(monkeypatch):
    payload = pdf_bytes(article())
    monkeypatch.setattr(extract, "download", lambda url: (payload, "application/pdf", url))
    text, kind = extract.fetch_source_text("https://example.org/download", bitscan=True)
    assert kind == "pdf"
    assert "students and faculty" in text
    assert "BITScan" not in text and "Page 1" not in text
    assert text == extract.fetch_pdf_text("https://example.org/issue.pdf", bitscan=True)


@pytest.mark.parametrize("payload,reason", [
    (b"<html>Login required</html>", "non-PDF"),
    (b"%PDF-1.7\nbroken", "Cannot parse PDF"),
    (pdf_bytes(), "run OCR"),
    (pdf_bytes("Hello"), "run OCR"),
])
def test_bad_pdf_rejected(payload, reason):
    with pytest.raises(extract.ExtractionError, match=reason):
        extract.extract_pdf_text(payload)


def test_drive_download_does_not_scrape_viewer(monkeypatch):
    urls = []
    def download(url):
        urls.append(url)
        return b"<html>Sign in</html>", "text/html", url
    monkeypatch.setattr(extract, "download", download)
    with pytest.raises(extract.ExtractionError, match="Enable public download"):
        extract.fetch_source_text("https://drive.google.com/file/d/abc/view", bitscan=True)
    assert urls == ["https://drive.google.com/uc?export=download&id=abc"]


def test_html_fallback_preserves_root_with_header_css():
    html = (f'<html class="vector-feature-language-in-header-enabled"><body>'
            f'<div id="mw-content-text"><p>{article()}</p></div></body></html>').encode()
    assert "students" in extract.extract_html_text(html, wikinews=True)
    with pytest.raises(extract.ExtractionError, match="cannot be decoded"):
        extract.extract_html_text(b"%PDF-1.7 binary")


def test_conservative_bitscan_cleaning():
    pages = [
        f"BITScan\nPilani Campus\nIssue 22\nSemester I 2024-25\n"
        f"Research .... 4\n{article()}\nUnique event number {i}\n"
        f"Contact us: editor@example.org\n{i}"
        for i in (1, 2, 3)
    ]
    text = clean_bitscan_pages(pages)
    assert "BITScan" not in text and "Issue 22" not in text
    assert "Contact us" not in text and "Research ...." not in text
    for word in ("BITS", "Pilani", "Hyderabad", "Goa", "students", "faculty"):
        assert word in text
        assert word.lower() in TextPreprocessor().tokenize(text)
    assert TextPreprocessor.clean_text("Pilani\nHyderabad\nGoa") == "Pilani\nHyderabad\nGoa"
    repeated_prose = "Students and faculty celebrate in Pilani."
    assert repeated_prose in clean_bitscan_pages([repeated_prose] * 3)


@pytest.mark.parametrize("text,reason", [
    ("", "empty extraction"), ("few words", "extremely short"),
    ("12345 " * 60, "garbled"), ("\ufffd" * 200 + " words" * 60, "garbled"),
])
def test_quality_rejection_reports_count(text, reason):
    with pytest.raises(extract.ExtractionError, match=reason) as err:
        extract.validate_extraction(text, "pdf")
    assert err.value.word_count == len(text.split())
    assert "PDF" in str(err.value)


def test_long_extraction_is_reported():
    quality = extract.validate_extraction("campus " * 6000)
    assert quality == {"word_count": 6000, "warnings": ["unusually long extraction"]}


def test_sports_initials_are_valid_but_fragmented_pdf_text_is_not():
    text = "A C Gilchrist M L Hayden R T Ponting scored runs. " * 8
    assert extract.validate_extraction(text, "wikinews")["word_count"] > 50
    with pytest.raises(extract.ExtractionError, match="failed PDF"):
        extract.validate_extraction("a b c d e " * 20, "pdf")


def test_stale_url_force_and_missing_file_recovery(tmp_path, monkeypatch):
    db, raw = tmp_path / "corpus.sqlite", tmp_path / "raw"
    calls = []
    def fetch(url, **kwargs):
        calls.append(url)
        return article("newsource" if "new" in url else "oldsource"), "pdf"
    monkeypatch.setattr(ingest, "fetch_source_text", fetch)
    # An existing ID-only file is never trusted.
    raw.mkdir()
    (raw / "CAMPUS_001.txt").write_text(article("stale"))
    ingest.ingest_corpus(raw, db, metadata_rows=[row()], delay=0)
    assert len(calls) == 1
    (raw / "CAMPUS_001.txt").unlink()
    ingest.ingest_corpus(raw, db, metadata_rows=[row()], delay=0)
    assert len(calls) == 1 and (raw / "CAMPUS_001.txt").exists()
    updated = row("https://example.org/new.pdf")
    ingest.ingest_corpus(raw, db, metadata_rows=[updated], delay=0)
    assert "newsource" in load_corpus_from_sqlite(db)[0]["raw_text"]
    assert "oldsource" not in (raw / "CAMPUS_001.txt").read_text()
    ingest.ingest_corpus(raw, db, metadata_rows=[updated], force=True, delay=0)
    assert len(calls) == 3


def test_failed_refresh_cannot_leave_old_sqlite_row(tmp_path, monkeypatch):
    db = tmp_path / "corpus.sqlite"
    monkeypatch.setattr(ingest, "fetch_source_text", lambda *a, **k: (article(), "pdf"))
    ingest.ingest_corpus(tmp_path / "raw", db, metadata_rows=[row()], delay=0)
    def fail(*args, **kwargs):
        raise extract.ExtractionError("PDF needs OCR")
    monkeypatch.setattr(ingest, "fetch_source_text", fail)
    report = ingest.ingest_corpus(tmp_path / "raw", db,
                                 metadata_rows=[row("https://example.org/new.pdf")], delay=0)
    assert report["failed"] == 1
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT count(*) FROM documents").fetchone()[0] == 0
    with pytest.raises(ValueError, match="incomplete"):
        load_corpus_from_sqlite(db)


def test_changed_snapshot_is_rejected(tmp_path, monkeypatch):
    db = tmp_path / "corpus.sqlite"
    monkeypatch.setattr(ingest, "fetch_source_text", lambda *a, **k: (article(), "pdf"))
    ingest.ingest_corpus(tmp_path / "raw", db, metadata_rows=[row()], delay=0)
    path = tmp_path / "metadata.csv"
    path.write_text(path.read_text().replace("issue.pdf", "other.pdf"))
    with pytest.raises(ValueError, match="Metadata changed"):
        load_corpus_from_sqlite(db)


def test_legacy_database_requires_rebuild(tmp_path):
    db = tmp_path / "corpus.sqlite"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE documents (document_id TEXT)")
    with pytest.raises(ValueError, match="Rebuild"):
        load_corpus_from_sqlite(db)


def test_version_change_and_text_corruption_are_not_cached(tmp_path, monkeypatch):
    db = tmp_path / "corpus.sqlite"
    calls = []
    def fetch(*args, **kwargs):
        calls.append(True)
        return article(), "pdf"
    monkeypatch.setattr(ingest, "fetch_source_text", fetch)
    ingest.ingest_corpus(tmp_path / "raw", db, metadata_rows=[row()], delay=0)
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE documents SET extraction_version='legacy'")
    with pytest.raises(ValueError, match="Stale"):
        load_corpus_from_sqlite(db)
    ingest.ingest_corpus(tmp_path / "raw", db, metadata_rows=[row()], delay=0)
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE documents SET raw_text='corrupted'")
    ingest.ingest_corpus(tmp_path / "raw", db, metadata_rows=[row()], delay=0)
    assert len(calls) == 3


def test_invalid_metadata_fails_before_touching_database(tmp_path):
    db = tmp_path / "corpus.sqlite"
    with pytest.raises(ValueError, match="duplicate"):
        ingest.ingest_corpus(tmp_path / "raw", db, metadata_rows=[row(), row()], delay=0)
    assert not db.exists()


def test_legacy_migration_backs_up_and_refetches(tmp_path, monkeypatch):
    db = tmp_path / "corpus.sqlite"
    with sqlite3.connect(db) as conn:
        conn.execute("""CREATE TABLE documents (
            document_id TEXT PRIMARY KEY, category TEXT, title TEXT, source TEXT,
            url TEXT, publication_date TEXT, word_count INTEGER, file_path TEXT, raw_text TEXT)""")
        conn.execute("INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                     ("CAMPUS_001", "Campus", "Old campus", "BITS", row()["url"],
                      "", 100, "old.txt", "old campus contents"))
    monkeypatch.setattr(ingest, "fetch_source_text", lambda *a, **k: (article("fresh"), "pdf"))
    ingest.ingest_corpus(tmp_path / "raw", db, metadata_rows=[row()], delay=0)
    assert "fresh" in load_corpus_from_sqlite(db)[0]["raw_text"]
    with sqlite3.connect(str(db) + ".legacy.bak") as conn:
        assert conn.execute("SELECT raw_text FROM documents").fetchone()[0] == "old campus contents"
