import json
import sqlite3
from collections import Counter

import numpy as np
import pytest

from src import ingest_corpus as ingest
from src.dataset import (prepare_document_corpus, normalize_document_length,
                         choose_length_limit, DEFAULT_CATEGORIES)


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    rows = [{"document_id": f"{category.upper()}_{i:03}", "category": category,
             "title": f"Story {i}", "source": "fixture",
             "url": f"https://example.org/{category}/{i}", "publication_date": ""}
            for category in DEFAULT_CATEGORIES for i in range(1, 81)]
    def fetch(url, **kwargs):
        category = url.split("/")[-2].lower()
        text = (f"{category} reports meaningful events for local readers. " * 15).strip()
        return text, "html"
    monkeypatch.setattr(ingest, "fetch_source_text", fetch)
    db = tmp_path / "corpus.sqlite"
    ingest.ingest_corpus(tmp_path / "raw", db, metadata_rows=rows, delay=0)
    return db, tmp_path / "processed"


def test_all_320_and_train_only_vocabulary(corpus):
    db, output = corpus
    first = prepare_document_corpus(db_path=db, output_dir=output)
    X_train, X_test, y_train, y_test, vocab, train, test = first
    assert X_train.shape[0] == 256 and X_test.shape[0] == 64
    assert set(Counter(d["category"] for d in train).values()) == {64}
    assert set(Counter(d["category"] for d in test).values()) == {16}
    assert len({d["document_id"] for d in train + test}) == 320
    manifest = json.loads((output / "splits.json").read_text())
    cleaned = json.loads((output / "cleaned_corpus.json").read_text())
    assert [d["document_id"] for d in cleaned] == (
        manifest["train_document_ids"] + manifest["test_document_ids"])
    assert manifest["min_df"] == 2 and manifest["max_df_ratio"] == 0.85
    np.testing.assert_array_equal(
        np.load(output / "count_matrix.npy"), np.vstack([X_train, X_test]))
    # Put an otherwise eligible token in all held-out documents, and nowhere in train.
    with sqlite3.connect(db) as conn:
        for doc in test:
            text = "testexclusive " + doc["raw_text"]
            conn.execute("UPDATE documents SET raw_text=?, text_hash=?, word_count=? WHERE document_id=?",
                         (text, ingest.text_hash(text), len(text.split()), doc["document_id"]))
    second = prepare_document_corpus(db_path=db, output_dir=output)
    assert second[4] == vocab and "testexclusive" not in second[4]
    assert [d["document_id"] for d in second[6]] == [d["document_id"] for d in test]
    np.testing.assert_array_equal(second[0], X_train)
    assert choose_length_limit(second[5]) == choose_length_limit(train)


def test_explicit_demo_subset_and_insufficient_sources(corpus):
    db, output = corpus
    result = prepare_document_corpus(db_path=db, output_dir=output,
                                    target_docs_per_cat=20, train_docs_per_cat=16)
    assert result[0].shape[0] == 64 and result[1].shape[0] == 16
    with pytest.raises(ValueError, match="need 81"):
        prepare_document_corpus(db_path=db, output_dir=output, target_docs_per_cat=81)
    assert json.loads((output / "splits.json").read_text())["status"] == "rebuild_required"


def test_invalid_split_rejected(tmp_path):
    with pytest.raises(ValueError, match="train_docs_per_cat"):
        prepare_document_corpus(target_docs_per_cat=80, train_docs_per_cat=80,
                                output_dir=tmp_path)


def test_normalization_preserves_order_and_sentence_boundary():
    text = "First coherent sentence has five words. Another sentence has more words here. Final sentence."
    normalized = normalize_document_length(text, 12)
    assert normalized == "First coherent sentence has five words."
    assert text.startswith(normalized) and len(normalized.split()) <= 12
    assert normalize_document_length(text, 12) == normalized
    assert normalize_document_length(text, 100) == text
    assert normalize_document_length("one two three four", 3) == "one two three"
    with pytest.raises(ValueError):
        normalize_document_length(text, 0)


def test_length_limit_uses_article_training_counts():
    docs = [{"category": "Sports", "raw_text": "word " * 100},
            {"category": "Politics", "raw_text": "word " * 200},
            {"category": "Campus", "raw_text": "word " * 10000}]
    assert choose_length_limit(docs) == 190


def test_experiment_entry_points_use_new_corpus(corpus, tmp_path, monkeypatch):
    """Short integration runs exercise all consumers, including text-level evasion."""
    import shutil
    from experiments import run_self_collected, lda_baseline, poison_budget, test_time_evasion
    db, _ = corpus
    data = tmp_path / "data"
    data.mkdir()
    shutil.copy2(db, data / "corpus.sqlite")
    shutil.copy2(db.parent / "metadata.csv", data / "metadata.csv")
    monkeypatch.chdir(tmp_path)
    for module in (run_self_collected, poison_budget, test_time_evasion):
        monkeypatch.setattr(module, "MODEL_MAX_ITER", 2)
        monkeypatch.setattr(module, "MODEL_RESTARTS", 1)
    monkeypatch.setattr(lda_baseline, "PLSA_MAX_ITER", 2)
    monkeypatch.setattr(lda_baseline, "PLSA_RESTARTS", 1)
    original_lda = lda_baseline.LdaModel
    def short_lda(*args, **kwargs):
        kwargs["passes"] = 1
        return original_lda(*args, **kwargs)
    monkeypatch.setattr(lda_baseline, "LdaModel", short_lda)
    run_self_collected.run_self_collected_experiments(seeds=[42])
    potency = json.loads((tmp_path / "results/dataset_320/attack_potency.json").read_text())
    assert potency["evaluation_corpus"] == "clean_training"
    assert potency["likelihood_unit"] == "nats_per_token"
    for runs in potency["runs"].values():
        assert "delta_eval_ll_per_token" in runs[0]
        assert "delta_log_likelihood" not in runs[0]
    lda_baseline.run_lda_experiments(seeds=[42])
    poison_budget.run_poison_budget(budgets=(2,))
    test_time_evasion.run_evasion(swap_counts=(0, 1))
    for filename in ("self_collected_dataset_seed_42.json", "lda_dataset_seed_42.json",
                     "poison_budget_curve.json.dataset.json", "evasion_summary.json.dataset.json"):
        manifest = json.loads((tmp_path / "results" / "dataset_320" / filename).read_text())
        assert manifest["n_train"] == 256 and manifest["n_test"] == 64
        assert manifest["length_normalization"]["max_words"] > 0
