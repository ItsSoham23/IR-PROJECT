"""Shared evaluation helpers for PLSA and Gensim LDA experiments."""

from typing import Any, Mapping, Sequence
import numpy as np

from src.align import TopicAligner
from src.coherence import npmi
from src.metrics import compute_classification_metrics, compute_nmi, compute_purity
from src.retrieval import IRBenchmark


def evaluate_lda(
    model: Any,
    train_corpus: Sequence[list[tuple[int, int]]],
    test_corpus: Sequence[list[tuple[int, int]]],
    y_train: np.ndarray,
    y_test: np.ndarray,
    corpus_tokens: Sequence[Sequence[str]] | None = None,
    n_topics: int | None = None,
) -> dict[str, float]:
    """Fold in LDA test documents and return the common project metrics."""
    topics = n_topics or int(model.num_topics)

    def proportions(corpus: Sequence[list[tuple[int, int]]]) -> np.ndarray:
        values = np.zeros((len(corpus), topics), dtype=float)
        for row, bow in enumerate(corpus):
            for topic, probability in model.get_document_topics(bow, minimum_probability=0):
                values[row, topic] = probability
        return values

    train_topics = proportions(train_corpus)
    test_topics = proportions(test_corpus)
    aligner = TopicAligner(n_topics=topics, n_classes=len(np.unique(y_train)))
    aligner.fit(np.argmax(train_topics, axis=1), y_train)
    predictions = aligner.predict(np.argmax(test_topics, axis=1))
    classification = compute_classification_metrics(y_test, predictions)
    result: dict[str, float] = {
        "macro_f1": classification["macro_f1"],
        "accuracy": classification["accuracy"],
        "purity": compute_purity(y_test, np.argmax(test_topics, axis=1)),
        "nmi": compute_nmi(y_test, np.argmax(test_topics, axis=1)),
        "perplexity": float(np.exp2(-model.log_perplexity(test_corpus))),
        **IRBenchmark.evaluate_retrieval(test_topics, train_topics, y_test, y_train),
    }
    if corpus_tokens is not None:
        top_words = {
            topic: [word for word, _ in model.show_topic(topic, topn=10)]
            for topic in range(topics)
        }
        result["npmi_mean"] = npmi(top_words, corpus_tokens)
    return result
