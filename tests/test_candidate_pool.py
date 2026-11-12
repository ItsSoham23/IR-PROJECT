import csv
import json
import sqlite3
from collections import Counter

import pytest

from src import ingest_corpus as ingest
from src.corpus_policy import DEFAULT_CATEGORIES
from src.dataset import load_corpus_from_sqlite, prepare_document_corpus
from src.source_extract import ExtractionError


FAILED_NUMBERS = {9, 12, 26, 29, 30, 31, 32, 34, 35, 36, 37, 38, 40, 42, 62, 64}


@pytest.fixture
def candidate_pool(tmp_path, monkeypatch):
    rows = [{"document_id": f"{cat.upper()}_{i:03}", "category": cat,
             "title": f"Story {i}", "source": "fixture",
             "url": f"https://example.org/{cat}/{i}.pdf", "publication_date": ""}
            for cat in DEFAULT_CATEGORIES for i in range(1, 81)]
    failing = {r["url"] for r in rows if r["category"] == "Campus"
               and int(r["document_id"].split("_")[1]) in FAILED_NUMBERS}
    calls = []

    def fetch(url, **kwargs):
        calls.append(url)
        if url in failing:
            raise ExtractionError("PDF needs OCR")
        cat, filename = url.split("/")[-2:]
        number = int(filename.split(".")[0])
        # Paired lengths deliberately exercise the document-ID tie-breaker.
        return (f"{cat} researchers report meaningful events for local readers. "
                * (20 + number // 2)).strip(), "pdf" if cat == "Campus" else "wikinews"

    monkeypatch.setattr(ingest, "fetch_source_text", fetch)
    db, raw, audit = tmp_path / "corpus.sqlite", tmp_path / "raw", tmp_path / "audit.csv"

    def run():
        return ingest.ingest_corpus(raw, db, metadata_rows=rows, delay=0, audit_path=audit)

    report = run()
    return {"db": db, "raw": raw, "audit": audit, "rows": rows, "report": report,
            "run": run, "calls": calls, "failing": failing, "output": tmp_path / "processed"}


def prepare(pool, seed=42):
    result = prepare_document_corpus(db_path=pool["db"], output_dir=pool["output"], seed=seed)
    return result, json.loads((pool["output"] / "splits.json").read_text())


def test_rejected_candidates_allow_balanced_model_ready_pool(candidate_pool):
    pool = candidate_pool
    assert pool["report"]["audit_finished"] and pool["report"]["model_ready"]
    assert pool["report"]["accepted"] == 304 and pool["report"]["failed"] == 16
    result, manifest = prepare(pool)
    assert result[0].shape[0] == 204 and result[1].shape[0] == 52
    train, test = set(manifest["train_document_ids"]), set(manifest["test_document_ids"])
    assert not train & test and len(train | test) == 256
    failed = {f"CAMPUS_{i:03}" for i in FAILED_NUMBERS}
    assert set(manifest["failed_candidate_ids"]) == failed
    assert not failed & (train | test)
    assert len(manifest["valid_unused_document_ids"]) == 48
    for cat, counts in manifest["counts_by_category"].items():
        assert counts == {"candidate": 80, "accepted_valid": 64 if cat == "Campus" else 80,
                          "failed": 16 if cat == "Campus" else 0, "selected": 64,
                          "valid_unused": 0 if cat == "Campus" else 16, "train": 51, "test": 13}
    with sqlite3.connect(pool["db"]) as conn:
        conn.row_factory = sqlite3.Row
        state = dict(conn.execute("SELECT * FROM ingestion_state").fetchone())
        usable = {r[0] for r in conn.execute("SELECT document_id FROM documents")}
    assert state["audit_finished"] == state["model_ready"] == 1
    assert set(json.loads(state["failed_ids_json"])) == failed
    assert set(json.loads(state["accepted_ids_json"])) == usable
    assert not failed & usable and len(usable) == 304
    with pool["audit"].open(newline="") as handle:
        audit = list(csv.DictReader(handle))
    assert Counter(d["selection_status"] for d in audit) == {
        "selected": 256, "valid_unused": 48, "rejected": 16}
    assert {d["document_id"] for d in audit if d["selection_status"] == "selected"} == train | test


def test_selection_precedes_seeded_split_and_uses_centered_lengths(candidate_pool):
    pool = candidate_pool
    _, first = prepare(pool, 42)
    _, second = prepare(pool, 17)
    assert first["selected_document_ids"] == second["selected_document_ids"]
    assert first["valid_unused_document_ids"] == second["valid_unused_document_ids"]
    assert first["corpus_fingerprint"] == second["corpus_fingerprint"]
    assert first["train_document_ids"] != second["train_document_ids"]
    docs = load_corpus_from_sqlite(pool["db"])
    for cat in DEFAULT_CATEGORIES:
        ranked = sorted((d for d in docs if d["category"] == cat),
                        key=lambda d: (d["word_count"], d["document_id"]))
        offset = (len(ranked) - 64) // 2
        assert {d["document_id"] for d in ranked[offset:offset + 64]} == {
            i for i in first["selected_document_ids"] if i.startswith(cat.upper() + "_")}


def test_63_valid_rejected_and_previously_accepted_row_removed(candidate_pool):
    pool = candidate_pool
    changed = pool["rows"][0]
    changed["url"] += "?broken"
    pool["failing"].add(changed["url"])
    report = pool["run"]()
    assert report["audit_finished"] and not report["model_ready"]
    assert report["valid_counts"]["Campus"] == 63
    with sqlite3.connect(pool["db"]) as conn:
        assert conn.execute("SELECT count(*) FROM documents WHERE document_id='CAMPUS_001'").fetchone()[0] == 0
    with pytest.raises(ValueError, match="Campus: need 64 valid documents, found 63"):
        prepare(pool)


def test_failed_candidate_repaired_url_is_fetched_and_selection_recomputed(candidate_pool):
    pool = candidate_pool
    changed = pool["rows"][8]
    old_hash = ingest.metadata_hash(changed)
    changed["url"] += "?repaired"
    (pool["raw"] / "CAMPUS_009.txt").write_text("stale raw file must never be used")
    pool["calls"].clear()
    report = pool["run"]()
    assert changed["url"] in pool["calls"]
    assert ingest.metadata_hash(changed) != old_hash
    assert report["valid_counts"]["Campus"] == 65
    assert "CAMPUS_009" not in report["failed_document_ids"]
    _, manifest = prepare(pool)
    assert "CAMPUS_009" in manifest["selected_document_ids"]
    assert manifest["counts_by_category"]["Campus"]["valid_unused"] == 1
    docs = load_corpus_from_sqlite(pool["db"])
    repaired = next(d for d in docs if d["document_id"] == "CAMPUS_009")
    assert repaired["metadata_hash"] == ingest.metadata_hash(changed)
    assert "stale raw file" not in repaired["raw_text"]


@pytest.mark.parametrize("mutation,reason", [
    ("UPDATE ingestion_state SET audit_finished=0", "audit is incomplete"),
    ("UPDATE ingestion_state SET accepted_ids_json='[]'", "Accepted/rejected"),
    ("UPDATE ingestion_state SET failed_ids_json='[]'", "Accepted/rejected"),
    ("UPDATE ingestion_state SET valid_counts_json='{}'", "Readiness counts"),
    ("UPDATE documents SET url='https://example.org/altered' WHERE document_id='CAMPUS_001'", "Stale"),
    ("UPDATE documents SET raw_text='corrupt' WHERE document_id='CAMPUS_001'", "Stale"),
])
def test_loader_rejects_inconsistent_audited_state(candidate_pool, mutation, reason):
    with sqlite3.connect(candidate_pool["db"]) as conn:
        conn.execute(mutation)
    with pytest.raises(ValueError, match=reason):
        load_corpus_from_sqlite(candidate_pool["db"])


def test_legacy_complete_flag_requires_explicit_reaudit(candidate_pool):
    pool = candidate_pool
    with sqlite3.connect(pool["db"]) as conn:
        conn.execute("DROP TABLE ingestion_state")
        conn.execute("CREATE TABLE ingestion_state (id INTEGER, metadata_json TEXT, complete INTEGER)")
        conn.execute("INSERT INTO ingestion_state VALUES (1, '{}', 1)")
    with pytest.raises(ValueError, match="Legacy ingestion state"):
        load_corpus_from_sqlite(pool["db"])
    assert pool["run"]()["model_ready"]
    assert len(load_corpus_from_sqlite(pool["db"])) == 304
