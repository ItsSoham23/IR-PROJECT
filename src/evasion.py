"""Test-time token perturbations for topic-model evasion experiments."""

from typing import Sequence
import numpy as np


def perturb_document(
    doc_tokens: Sequence[str],
    n_swaps: int = 1,
    topic_words: Sequence[str] | None = None,
    random_state: int | np.random.RandomState | None = None,
) -> list[str]:
    """Replace up to ``n_swaps`` non-stopword tokens with target-topic words.

    The caller supplies the target topic's high-probability words.  Without
    them, the function returns a copy because it cannot infer a topic safely.
    """
    if n_swaps < 0:
        raise ValueError("n_swaps must be non-negative")
    if not topic_words or not doc_tokens or n_swaps == 0:
        return list(doc_tokens)
    rng = random_state if isinstance(random_state, np.random.RandomState) else np.random.RandomState(random_state)
    output = list(doc_tokens)
    candidates = np.arange(len(output))
    swap_count = min(n_swaps, len(output))
    for index in rng.choice(candidates, size=swap_count, replace=False):
        output[int(index)] = str(topic_words[rng.randint(len(topic_words))])
    return output
