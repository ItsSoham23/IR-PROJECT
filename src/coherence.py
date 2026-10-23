"""Topic coherence utilities based on document-level word co-occurrence."""

from collections import Counter
from itertools import combinations
from math import log
from typing import Iterable, List, Mapping, Sequence


def npmi(
    topic_words: Mapping[int, Sequence[str]] | Sequence[Sequence[str]],
    corpus: Iterable[Sequence[str]],
    vocab: Iterable[str] | None = None,
) -> float:
    """Return mean NPMI for the supplied topic word lists.

    Each document contributes at most once to a word or word pair.  ``vocab`` is
    accepted for API compatibility and is used only to filter topic words when
    supplied.
    """
    documents = [set(document) for document in corpus]
    if not documents:
        return 0.0

    allowed = set(vocab) if vocab is not None else None
    topics = topic_words.values() if isinstance(topic_words, Mapping) else topic_words
    scores: List[float] = []
    document_count = len(documents)
    word_df = Counter(word for document in documents for word in document)

    for raw_words in topics:
        words = [word for word in raw_words if allowed is None or word in allowed]
        for first, second in combinations(dict.fromkeys(words), 2):
            p_first = word_df[first] / document_count
            p_second = word_df[second] / document_count
            pair_df = sum(first in document and second in document for document in documents)
            p_pair = pair_df / document_count
            if p_first == 0 or p_second == 0 or p_pair == 0 or p_pair == 1:
                continue
            scores.append(log(p_pair / (p_first * p_second)) / -log(p_pair))

    return sum(scores) / len(scores) if scores else 0.0


def topic_npmi(
    topic_words: Mapping[int, Sequence[str]] | Sequence[Sequence[str]],
    corpus: Iterable[Sequence[str]],
    vocab: Iterable[str] | None = None,
) -> dict[int, float]:
    """Return NPMI per topic using the same convention as :func:`npmi`."""
    documents = [set(document) for document in corpus]
    allowed = set(vocab) if vocab is not None else None
    topics = topic_words.items() if isinstance(topic_words, Mapping) else enumerate(topic_words)
    document_count = len(documents)
    word_df = Counter(word for document in documents for word in document)
    scores: dict[int, float] = {}
    for topic, raw_words in topics:
        words = [word for word in raw_words if allowed is None or word in allowed]
        values = []
        for first, second in combinations(dict.fromkeys(words), 2):
            p_first = word_df[first] / document_count if document_count else 0
            p_second = word_df[second] / document_count if document_count else 0
            pair_df = sum(first in document and second in document for document in documents)
            p_pair = pair_df / document_count if document_count else 0
            if p_first and p_second and p_pair and p_pair < 1:
                values.append(log(p_pair / (p_first * p_second)) / -log(p_pair))
        scores[int(topic)] = sum(values) / len(values) if values else 0.0
    return scores
