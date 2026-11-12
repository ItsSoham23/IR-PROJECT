"""Balanced corpus construction with train-only vocabulary and versioned sources."""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
import sqlite3
import csv
import re
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

from src.preprocess import TextPreprocessor
from src.ingest_corpus import metadata_hash, text_hash, validate_metadata
from src.source_extract import EXTRACTION_VERSION, validate_extraction

DEFAULT_CATEGORIES = ("Campus", "Politics", "Sports", "Technology")
DEFAULT_LENGTH_PERCENTILE = 90


def normalize_document_length(text, max_words):
    """A deterministic prefix, ending at a sentence boundary where available."""
    if max_words < 1:
        raise ValueError("max_words must be positive")
    words = list(re.finditer(r"\S+", text))
    if len(words) <= max_words:
        return text.strip()
    prefix = text[:words[max_words - 1].end()]
    # Keep a meaningful portion of the budget when PDF punctuation is sparse.
    endings = list(re.finditer(r'[.!?]["\u201d\')]*\s+', prefix))
    if endings and len(prefix[:endings[-1].end()].split()) >= max_words / 2:
        prefix = prefix[:endings[-1].end()]
    return prefix.strip()


def choose_length_limit(train_docs, percentile=DEFAULT_LENGTH_PERCENTILE):
    """Learn the cap from training articles, excluding the long Campus magazines."""
    if not 0 < percentile <= 100:
        raise ValueError("length_percentile must be in (0, 100]")
    reference = [d for d in train_docs if d["category"] != "Campus"]
    reference = reference or train_docs
    return max(1, int(np.ceil(np.percentile(
        [len(TextPreprocessor.clean_text(d["raw_text"]).split()) for d in reference],
        percentile))))


def save_experiment_manifest(output_path, processed_dir="data/processed"):
    """Attach the actual split, source fingerprint and length policy to results."""
    manifest = json.loads(Path(processed_dir, "splits.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "ready" or manifest.get("schema_version") != 2:
        raise ValueError("Processed corpus must be rebuilt before exporting results")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def load_corpus_from_sqlite(db_path: str = "data/corpus.sqlite") -> List[Dict]:
    """Reject legacy, interrupted, altered, or metadata-mismatched ingestion."""
    rebuild = "Rebuild with python -m src.ingest_corpus before preparing the corpus."
    if not Path(db_path).is_file():
        raise ValueError(f"Corpus database is missing. {rebuild}")
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            state = conn.execute("SELECT * FROM ingestion_state WHERE id = 1").fetchone()
            docs = [dict(r) for r in conn.execute(
                "SELECT * FROM documents ORDER BY category, document_id")]
        if not state or not state["complete"]:
            raise ValueError(f"Ingestion is incomplete. {rebuild}")
        expected = json.loads(state["metadata_json"])
        snapshot = Path(db_path).parent / "metadata.csv"
        with snapshot.open(encoding="utf-8-sig", newline="") as handle:
            rows = validate_metadata(list(csv.DictReader(handle)))
        if expected != {r["document_id"]: metadata_hash(r) for r in rows}:
            raise ValueError(f"Metadata changed since ingestion. {rebuild}")
        if {d["document_id"]: d["metadata_hash"] for d in docs} != expected:
            raise ValueError(f"Corpus sources do not match ingestion manifest. {rebuild}")
        for doc in docs:
            if (doc["extraction_version"] != EXTRACTION_VERSION
                    or doc["metadata_hash"] != metadata_hash(doc)
                    or doc["text_hash"] != text_hash(doc["raw_text"])):
                raise ValueError(f"Stale or modified source: {doc['document_id']}. {rebuild}")
            quality = validate_extraction(doc["raw_text"], doc["source_type"])
            if quality["word_count"] != doc["word_count"]:
                raise ValueError(f"Incorrect stored word count: {doc['document_id']}. {rebuild}")
        return docs
    except (sqlite3.Error, OSError, KeyError) as exc:
        raise ValueError(f"Legacy or invalid corpus. {rebuild}") from exc


def prepare_document_corpus(
    seed: int = 42,
    target_docs_per_cat: int = 80,
    train_docs_per_cat: int = 64,
    db_path: str = "data/corpus.sqlite",
    output_dir: str = "data/processed",
    expected_categories=DEFAULT_CATEGORIES,
    max_document_words=None,
    length_percentile=DEFAULT_LENGTH_PERCENTILE,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], List[Dict], List[Dict]]:
    """Use every source at the target size; select a balanced subset of larger pools."""
    if not 0 < train_docs_per_cat < target_docs_per_cat:
        raise ValueError("Require 0 < train_docs_per_cat < target_docs_per_cat")
    os.makedirs(output_dir, exist_ok=True)
    # Written before validation, so a failed rebuild cannot advertise an old split.
    Path(output_dir, "splits.json").write_text(
        json.dumps({"status": "rebuild_required"}), encoding="utf-8")
    all_docs = load_corpus_from_sqlite(db_path)

    rng = np.random.RandomState(seed)
    categories = sorted(list(set(d["category"] for d in all_docs)))
    if expected_categories is not None and categories != sorted(expected_categories):
        raise ValueError(f"Expected categories {sorted(expected_categories)}, found {categories}")
    if not categories:
        raise ValueError("Corpus is empty")

    selected_docs = []
    train_docs = []
    test_docs = []

    cat_to_id = {c: i for i, c in enumerate(categories)}

    for cat in categories:
        cat_candidates = [d for d in all_docs if d["category"] == cat]
        if len(cat_candidates) < target_docs_per_cat:
            raise ValueError(f"{cat}: need {target_docs_per_cat} valid documents, "
                             f"found {len(cat_candidates)}; repair failed sources and re-ingest")
        # Retain median selection only for a larger future candidate pool.
        if len(cat_candidates) > target_docs_per_cat:
            cat_candidates = sorted(cat_candidates, key=lambda x: (x["word_count"], x["document_id"]))
            offset = (len(cat_candidates) - target_docs_per_cat) // 2
            chosen = cat_candidates[offset : offset + target_docs_per_cat]
        else:
            chosen = cat_candidates
        chosen = sorted(chosen, key=lambda x: x["document_id"])

        # Randomly permute with seed for train/test split
        perm = rng.permutation(len(chosen))
        train_idx = perm[:train_docs_per_cat]

        for i, doc in enumerate(chosen):
            doc_copy = dict(doc)
            doc_copy["label"] = cat_to_id[cat]
            if i in train_idx:
                doc_copy["split"] = "train"
                train_docs.append(doc_copy)
            else:
                doc_copy["split"] = "test"
                test_docs.append(doc_copy)
            selected_docs.append(doc_copy)

    length_limit = (choose_length_limit(train_docs, length_percentile)
                    if max_document_words is None else max_document_words)
    if not isinstance(length_limit, int) or length_limit < 1:
        raise ValueError("max_document_words must be a positive integer")
    for doc in selected_docs:
        doc["model_text"] = normalize_document_length(
            TextPreprocessor.clean_text(doc["raw_text"]), length_limit)
        doc["model_word_count"] = len(doc["model_text"].split())
    train_texts = [d["model_text"] for d in train_docs]
    test_texts = [d["model_text"] for d in test_docs]

    y_train = np.array([d["label"] for d in train_docs], dtype=np.int64)
    y_test = np.array([d["label"] for d in test_docs], dtype=np.int64)

    # Build vocabulary strictly on training split
    preprocessor = TextPreprocessor(min_df=2, max_df_ratio=0.85, min_word_len=3)
    X_train = preprocessor.fit_transform(train_texts)
    X_test = preprocessor.transform(test_texts)
    vocab = preprocessor.vocab_

    # Save artifacts to data/processed
    np.save(os.path.join(output_dir, "X_train.npy"), X_train)
    np.save(os.path.join(output_dir, "X_test.npy"), X_test)
    np.save(os.path.join(output_dir, "y_train.npy"), y_train)
    np.save(os.path.join(output_dir, "y_test.npy"), y_test)

    # Full matrix and cleaned text share the exact train-then-test row order.
    X_full = np.vstack([X_train, X_test])
    np.save(os.path.join(output_dir, "count_matrix.npy"), X_full)
    np.save(os.path.join(output_dir, "count_matrix_full.npy"), X_full)

    # Save cleaned / tokenized documents
    cleaned_corpus = []
    for d in train_docs + test_docs:
        tokens = preprocessor.tokenize(d["model_text"])
        cleaned_corpus.append({
            "document_id": d["document_id"],
            "category": d["category"],
            "split": d["split"],
            "extracted_word_count": d["word_count"],
            "model_word_count": d["model_word_count"],
            "tokens": tokens,
            "cleaned_text": " ".join(tokens),
        })
    with open(os.path.join(output_dir, "cleaned_corpus.json"), "w", encoding="utf-8") as f:
        json.dump(cleaned_corpus, f, indent=2)

    with open(os.path.join(output_dir, "vocab.txt"), "w", encoding="utf-8") as f:
        for w in vocab:
            f.write(w + "\n")

    # Save split manifest
    split_info = {
        "status": "ready",
        "schema_version": 2,
        "extraction_version": EXTRACTION_VERSION,
        "seed": seed,
        "target_docs_per_cat": target_docs_per_cat,
        "train_docs_per_cat": train_docs_per_cat,
        "min_df": 2,
        "max_df_ratio": 0.85,
        "length_normalization": {
            "strategy": "sentence_bounded_prefix",
            "max_words": length_limit,
            "reference": "explicit" if max_document_words is not None else "non_campus_training",
            "percentile": length_percentile,
        },
        "corpus_fingerprint": text_hash(json.dumps(
            [(d["document_id"], d["metadata_hash"], d["text_hash"]) for d in all_docs])),
        "categories": categories,
        "cat_to_id": cat_to_id,
        "n_train": len(train_docs),
        "n_test": len(test_docs),
        "vocab_size": len(vocab),
        "train_document_ids": [d["document_id"] for d in train_docs],
        "test_document_ids": [d["document_id"] for d in test_docs],
    }
    # Update SQLite database with split tags
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    # Add split column if not exists
    cur.execute("PRAGMA table_info(documents)")
    cols = [c[1] for c in cur.fetchall()]
    if "split" not in cols:
        cur.execute("ALTER TABLE documents ADD COLUMN split TEXT")

    cur.execute("UPDATE documents SET split = NULL")
    for d in selected_docs:
        cur.execute("UPDATE documents SET split = ? WHERE document_id = ?", (d["split"], d["document_id"]))
    conn.commit()
    conn.close()
    with open(os.path.join(output_dir, "splits.json"), "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2)

    print(f"Corpus prepared successfully:")
    print(f"  Categories: {categories}")
    print(f"  Train documents: {len(train_docs)} ({train_docs_per_cat} per cat)")
    print(f"  Test documents:  {len(test_docs)} ({target_docs_per_cat - train_docs_per_cat} per cat)")
    print(f"  Vocabulary size: {len(vocab)} terms")
    print(f"  Document length cap: {length_limit} words")

    return X_train, X_test, y_train, y_test, vocab, train_docs, test_docs


if __name__ == "__main__":
    prepare_document_corpus()
