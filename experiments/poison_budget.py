"""Run a reproducible PLSA poison-budget sensitivity sweep on the local corpus."""

import json
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from src.align import TopicAligner
from src.attack import AdversarialPoisonGenerator
from src.coherence import npmi
from src.dataset import prepare_100_document_corpus
from src.metrics import compute_classification_metrics
from src.plsa import PLSA
from src.preprocess import TextPreprocessor
from src.retrieval import IRBenchmark


def _evaluate(model, X_test, y_test, y_train, clean_count, vocab, corpus_tokens):
    train_topics = np.argmax(model.p_z_d[:clean_count], axis=1)
    aligner = TopicAligner(n_topics=model.n_topics, n_classes=len(np.unique(y_train)))
    aligner.fit(train_topics, y_train)
    test_proportions = model.transform(X_test)
    test_topics = np.argmax(test_proportions, axis=1)
    metrics = compute_classification_metrics(y_test, aligner.predict(test_topics))
    top_words = {k: [word for word, _ in pairs] for k, pairs in model.get_top_words(vocab, 10).items()}
    return {
        "macro_f1": metrics["macro_f1"],
        "map": IRBenchmark.evaluate_retrieval(
            test_proportions, model.p_z_d[:clean_count], y_test, y_train
        )["MAP"],
        "perplexity": model.perplexity(X_test, test_proportions),
        "npmi_mean": npmi(top_words, corpus_tokens, vocab),
    }


def run_poison_budget(
    budgets=(2, 5, 10, 15, 20),
    seed=42,
    output_path="results/poison_budget_curve.json",
):
    X_train, X_test, y_train, y_test, vocab, _, _ = prepare_100_document_corpus(seed=seed)
    preprocessor = TextPreprocessor(min_df=2, max_df_ratio=0.85)
    preprocessor.vocab_ = vocab
    preprocessor.word_to_idx_ = {word: index for index, word in enumerate(vocab)}
    with open("data/processed/cleaned_corpus.json", encoding="utf-8") as handle:
        cleaned = json.load(handle)
    corpus_tokens = [item["tokens"] for item in cleaned]

    clean_model = PLSA(n_topics=4, max_iter=50, random_state=seed).fit(X_train)
    aligner = TopicAligner(n_topics=4).fit(np.argmax(clean_model.p_z_d, axis=1), y_train)
    target_words = [word for word, _ in clean_model.get_top_words(vocab, 20)[aligner.class_to_topic_[0]]]
    generator = AdversarialPoisonGenerator(vocab, random_state=seed)
    results = []
    for budget in budgets:
        poison = generator.generate_keyword_repetition_attack(target_words, budget, 80)
        X_poison = preprocessor.transform(poison)
        X_condition = np.vstack([X_train, X_poison])
        record = {"poison_budget": budget}
        for name, beta in (("plsa", 1.0), ("plsa_tempered", 0.85)):
            model = PLSA(n_topics=4, max_iter=50, beta=beta, random_state=seed).fit(X_condition)
            record[name] = _evaluate(
                model, X_test, y_test, y_train, len(X_train), vocab, corpus_tokens
            )
        results.append(record)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump({"seed": seed, "budgets": list(budgets), "runs": results}, handle, indent=2)
    return results


if __name__ == "__main__":
    run_poison_budget()
