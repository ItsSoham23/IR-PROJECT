"""Measure PLSA test-time evasion without retraining the model."""

import json
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from src.align import TopicAligner
from src.dataset import prepare_100_document_corpus
from src.evasion import perturb_document
from src.metrics import compute_classification_metrics
from src.plsa import PLSA
from src.preprocess import TextPreprocessor


def run_evasion(
    swap_counts=(0, 1, 2, 3),
    seed=42,
    output_path="results/evasion_summary.json",
):
    X_train, X_test, y_train, y_test, vocab, _, test_docs = prepare_100_document_corpus(seed=seed)
    preprocessor = TextPreprocessor(min_df=2, max_df_ratio=0.85)
    preprocessor.vocab_ = vocab
    preprocessor.word_to_idx_ = {word: index for index, word in enumerate(vocab)}
    model = PLSA(n_topics=4, max_iter=50, random_state=seed).fit(X_train)
    aligner = TopicAligner(n_topics=4).fit(np.argmax(model.p_z_d, axis=1), y_train)
    target_topic = aligner.class_to_topic_[0]
    target_words = [word for word, _ in model.get_top_words(vocab, 50)[target_topic]]
    records = []
    for swaps in swap_counts:
        tokens = [
            perturb_document(
                preprocessor.tokenize(doc["raw_text"]),
                swaps,
                target_words,
                seed + index,
            )
            for index, doc in enumerate(test_docs)
        ]
        p_z_d = model.transform(preprocessor.transform([" ".join(doc) for doc in tokens]))
        predictions = aligner.predict(np.argmax(p_z_d, axis=1))
        records.append({
            "swaps": swaps,
            "macro_f1": compute_classification_metrics(y_test, predictions)["macro_f1"],
        })
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump({"seed": seed, "runs": records}, handle, indent=2)
    return records


if __name__ == "__main__":
    run_evasion()
