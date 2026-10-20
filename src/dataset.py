"""
Corpus Selection, Splitting, and Feature Extraction.

- Selects 25 balanced documents per category (total 100 documents) from the 120 candidates.
- Performs stratified 80/20 train/test split (20 train, 5 test per category).
- Builds vocabulary strictly on the 80 training documents.
- Saves processed matrices, vocabulary, and split metadata.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
import sqlite3
from typing import Dict, List, Tuple
import numpy as np

from src.preprocess import TextPreprocessor


def load_corpus_from_sqlite(db_path: str = "data/corpus.sqlite") -> List[Dict]:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        "SELECT document_id, category, title, source, url, word_count, raw_text FROM documents ORDER BY category, document_id"
    )
    rows = cur.fetchall()
    conn.close()

    docs = []
    for r in rows:
        docs.append({
            "document_id": r[0],
            "category": r[1],
            "title": r[2],
            "source": r[3],
            "url": r[4],
            "word_count": r[5],
            "raw_text": r[6],
        })
    return docs


def prepare_100_document_corpus(
    seed: int = 42,
    target_docs_per_cat: int = 25,
    train_docs_per_cat: int = 20,
    db_path: str = "data/corpus.sqlite",
    output_dir: str = "data/processed",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], List[Dict], List[Dict]]:
    """
    Selects 25 documents per category (100 total), splits 80/20, fits vocabulary on train.
    """
    os.makedirs(output_dir, exist_ok=True)
    all_docs = load_corpus_from_sqlite(db_path)

    rng = np.random.RandomState(seed)
    categories = sorted(list(set(d["category"] for d in all_docs)))

    selected_100 = []
    train_docs = []
    test_docs = []

    cat_to_id = {c: i for i, c in enumerate(categories)}

    for cat in categories:
        cat_candidates = [d for d in all_docs if d["category"] == cat]
        # Sort by moderate word count to avoid extreme outliers if any
        cat_candidates = sorted(cat_candidates, key=lambda x: x["word_count"])
        # Take 25 documents centered around the median length
        if len(cat_candidates) > target_docs_per_cat:
            offset = (len(cat_candidates) - target_docs_per_cat) // 2
            chosen = cat_candidates[offset : offset + target_docs_per_cat]
        else:
            chosen = cat_candidates[:target_docs_per_cat]

        # Randomly permute with seed for train/test split
        perm = rng.permutation(len(chosen))
        train_idx = perm[:train_docs_per_cat]
        test_idx = perm[train_docs_per_cat:]

        for i, doc in enumerate(chosen):
            doc_copy = dict(doc)
            doc_copy["label"] = cat_to_id[cat]
            if i in train_idx:
                doc_copy["split"] = "train"
                train_docs.append(doc_copy)
            else:
                doc_copy["split"] = "test"
                test_docs.append(doc_copy)
            selected_100.append(doc_copy)

    # Preprocess text
    train_texts = [d["raw_text"] for d in train_docs]
    test_texts = [d["raw_text"] for d in test_docs]

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

    with open(os.path.join(output_dir, "vocab.txt"), "w", encoding="utf-8") as f:
        for w in vocab:
            f.write(w + "\n")

    # Save split manifest
    split_info = {
        "categories": categories,
        "cat_to_id": cat_to_id,
        "n_train": len(train_docs),
        "n_test": len(test_docs),
        "vocab_size": len(vocab),
        "train_document_ids": [d["document_id"] for d in train_docs],
        "test_document_ids": [d["document_id"] for d in test_docs],
    }
    with open(os.path.join(output_dir, "splits.json"), "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2)

    # Update SQLite database with split tags
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    # Add split column if not exists
    cur.execute("PRAGMA table_info(documents)")
    cols = [c[1] for c in cur.fetchall()]
    if "split" not in cols:
        cur.execute("ALTER TABLE documents ADD COLUMN split TEXT")

    for d in selected_100:
        cur.execute("UPDATE documents SET split = ? WHERE document_id = ?", (d["split"], d["document_id"]))
    conn.commit()
    conn.close()

    print(f"Corpus prepared successfully:")
    print(f"  Categories: {categories}")
    print(f"  Train documents: {len(train_docs)} ({train_docs_per_cat} per cat)")
    print(f"  Test documents:  {len(test_docs)} ({target_docs_per_cat - train_docs_per_cat} per cat)")
    print(f"  Vocabulary size: {len(vocab)} terms")

    return X_train, X_test, y_train, y_test, vocab, train_docs, test_docs


if __name__ == "__main__":
    prepare_100_document_corpus()
