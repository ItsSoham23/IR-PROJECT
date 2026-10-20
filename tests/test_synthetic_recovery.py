"""
Test topic recovery and folding-in on synthetic data generated from distinct ground-truth topics.
"""

import numpy as np
import pytest
from src.plsa import PLSA
from src.align import TopicAligner
from src.metrics import compute_classification_metrics, compute_purity


def generate_synthetic_corpus(
    n_docs_per_topic: int = 25,
    vocab_per_topic: int = 15,
    n_topics: int = 3,
    words_per_doc: int = 100,
    noise_ratio: float = 0.05,
    random_state: int = 42,
):
    """
    Generate synthetic documents with well-separated vocabulary blocks.
    Each topic k concentrates on words [k*vocab_per_topic : (k+1)*vocab_per_topic].
    """
    rng = np.random.RandomState(random_state)
    total_vocab = n_topics * vocab_per_topic
    total_docs = n_topics * n_docs_per_topic

    X = np.zeros((total_docs, total_vocab), dtype=np.float64)
    labels = np.zeros(total_docs, dtype=np.int64)

    for k in range(n_topics):
        start_w = k * vocab_per_topic
        end_w = (k + 1) * vocab_per_topic

        # Define topic word distribution: high probability on its block, low elsewhere
        p_w = np.full(total_vocab, noise_ratio / (total_vocab - vocab_per_topic))
        p_w[start_w:end_w] = (1.0 - noise_ratio) / vocab_per_topic

        doc_start = k * n_docs_per_topic
        doc_end = (k + 1) * n_docs_per_topic
        labels[doc_start:doc_end] = k

        for d in range(doc_start, doc_end):
            # Sample words from categorical distribution
            words = rng.choice(total_vocab, size=words_per_doc, p=p_w)
            for w in words:
                X[d, w] += 1.0

    return X, labels


def test_synthetic_topic_recovery():
    X, labels = generate_synthetic_corpus(
        n_docs_per_topic=30, vocab_per_topic=20, n_topics=3, words_per_doc=150, random_state=42
    )

    model = PLSA(n_topics=3, max_iter=60, tol=1e-6, n_restarts=3, random_state=123)
    model.fit(X)

    dominant_topics = np.argmax(model.p_z_d, axis=1)

    aligner = TopicAligner(n_topics=3, n_classes=3)
    aligner.fit(dominant_topics, labels)
    aligned_predictions = aligner.predict(dominant_topics)

    metrics = compute_classification_metrics(labels, aligned_predictions)
    purity = compute_purity(labels, dominant_topics)

    # In clean synthetic data with distinct vocabulary blocks, PLSA should recover topics with high accuracy
    assert metrics["macro_f1"] >= 0.85, f"Macro-F1 too low on synthetic data: {metrics['macro_f1']}"
    assert purity >= 0.85, f"Purity too low: {purity}"


def test_folding_in_consistency():
    X, labels = generate_synthetic_corpus(
        n_docs_per_topic=20, vocab_per_topic=15, n_topics=3, random_state=42
    )

    model = PLSA(n_topics=3, max_iter=50, random_state=42)
    model.fit(X)

    # Fold in the training documents themselves
    p_z_folded = model.transform(X, max_iter=40)

    # Dominant topic between training P(z|d) and folded-in P(z|d) should match on almost all documents
    trained_dominant = np.argmax(model.p_z_d, axis=1)
    folded_dominant = np.argmax(p_z_folded, axis=1)

    agreement = np.mean(trained_dominant == folded_dominant)
    assert agreement >= 0.90, f"Folded-in dominant topics differ from trained model: {agreement}"
