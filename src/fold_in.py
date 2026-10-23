"""Text-level folding-in helpers shared by evasion experiments."""

from typing import Sequence
import numpy as np

from src.evasion import perturb_document


def fold_in_documents(
    model,
    documents: Sequence[Sequence[str]],
    preprocessor,
    perturb: bool = False,
    n_swaps: int = 1,
    target_topic: int | None = None,
    random_state: int | None = None,
) -> np.ndarray:
    """Transform tokenized documents, optionally applying topic-word evasion."""
    topic_words = None
    if perturb:
        if target_topic is None or model.p_w_z is None:
            raise ValueError("perturb=True requires a fitted PLSA model and target_topic.")
        topic_words = [
            word
            for word, _ in model.get_top_words(preprocessor.vocab_, n_top_words=50)[target_topic]
        ]
    rng = np.random.RandomState(random_state)
    transformed = [
        perturb_document(list(tokens), n_swaps=n_swaps, topic_words=topic_words, random_state=rng)
        if perturb
        else list(tokens)
        for tokens in documents
    ]
    counts = preprocessor.transform([" ".join(tokens) for tokens in transformed])
    return model.transform(counts, random_state=random_state)
