"""Short offline demonstration of the project's PLSA pipeline.

Run from any directory with:
    python demo.py
"""

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.align import TopicAligner
from src.attack import AdversarialPoisonGenerator
from src.dataset import prepare_100_document_corpus
from src.defense import DefenseFilter
from src.metrics import compute_classification_metrics
from src.plsa import PLSA
from src.preprocess import TextPreprocessor
from src.retrieval import IRBenchmark


def main() -> None:
    print("=" * 68)
    print("PLSA LIVE DEMO")
    print("=" * 68)
    print("Preparing the included local corpus...")

    processed = ROOT / "data" / "processed"
    artifacts = [
        processed / name
        for name in ("X_train.npy", "X_test.npy", "y_train.npy", "y_test.npy", "vocab.txt")
    ]
    if all(path.exists() for path in artifacts):
        X_train = np.load(processed / "X_train.npy")
        X_test = np.load(processed / "X_test.npy")
        y_train = np.load(processed / "y_train.npy")
        y_test = np.load(processed / "y_test.npy")
        vocab = (processed / "vocab.txt").read_text(encoding="utf-8").splitlines()
    else:
        X_train, X_test, y_train, y_test, vocab, _, _ = prepare_100_document_corpus(
            seed=42,
            db_path=str(ROOT / "data" / "corpus.sqlite"),
            output_dir=str(processed),
        )

    print(f"Training documents: {len(X_train)}")
    print(f"Held-out documents: {len(X_test)}")
    print(f"Vocabulary terms:   {len(vocab)}")
    print("\nRunning clean, poisoning, control, and defense conditions...")
    preprocessor = TextPreprocessor(min_df=2, max_df_ratio=0.85)
    preprocessor.vocab_ = vocab
    preprocessor.word_to_idx_ = {word: i for i, word in enumerate(vocab)}

    def fit_and_score(name, X_model, y_model, beta=1.0, clean_count=len(X_train)):
        model = PLSA(
            n_topics=4, max_iter=60, tol=1e-5, n_restarts=1,
            beta=beta, random_state=42,
        )
        model.fit(X_model)
        clean_topics = np.argmax(model.p_z_d[:clean_count], axis=1)
        aligner = TopicAligner(n_topics=4).fit(
            clean_topics, y_model[:clean_count]
        )
        test_proportions = model.transform(X_test)
        predictions = aligner.predict(np.argmax(test_proportions, axis=1))
        metrics = compute_classification_metrics(y_test, predictions)
        retrieval = IRBenchmark.evaluate_retrieval(
            test_proportions, model.p_z_d[:clean_count], y_test, y_model[:clean_count]
        )
        score = (
            metrics["macro_f1"], metrics["accuracy"], retrieval["MAP"],
            model.perplexity(X_test, test_proportions),
        )
        print(
            f"{name:<30} | F1 {score[0]:.3f} | Acc {score[1]:.3f} "
            f"| MAP {score[2]:.3f} | PPL {score[3]:.1f}"
        )
        return model, aligner

    clean_model, clean_aligner = fit_and_score(
        "Clean baseline", X_train, y_train
    )
    target_words = [
        word for word, _ in clean_model.get_top_words(
            vocab, n_top_words=20
        )[clean_aligner.class_to_topic_[0]]
    ]
    confuser_words = [
        word for word, _ in clean_model.get_top_words(
            vocab, n_top_words=20
        )[clean_aligner.class_to_topic_[1]]
    ]
    generator = AdversarialPoisonGenerator(vocab, random_state=42)
    attacks = {
        "Keyword repetition": generator.generate_keyword_repetition_attack(
            target_words, n_docs=10, doc_length=80
        ),
        "Bridging attack": generator.generate_bridging_attack(
            target_words, confuser_words, n_docs=10, doc_length=80
        ),
        "Random control": generator.generate_random_control(
            n_docs=10, doc_length=80
        ),
    }
    matrices = {
        name: np.vstack([X_train, preprocessor.transform(documents)])
        for name, documents in attacks.items()
    }
    for name, matrix in matrices.items():
        fit_and_score(name, matrix, y_train)

    poisoned = matrices["Keyword repetition"]
    deduped, kept = DefenseFilter.deduplicate_documents(
        poisoned, threshold=0.75, protected_indices=list(range(len(X_train)))
    )
    clean_kept = [index for index in kept if index < len(X_train)]
    fit_and_score(
        "Deduplication defense",
        deduped,
        y_train[clean_kept],
        clean_count=len(clean_kept),
    )
    fit_and_score(
        "Tempered EM defense",
        poisoned,
        y_train,
        beta=0.85,
    )
    print("\nTop words from the clean model:")
    for topic, words in clean_model.get_top_words(vocab, n_top_words=8).items():
        category = clean_aligner.topic_to_class_.get(topic, "?")
        print(f"  Topic {topic} -> category {category}: " + ", ".join(word for word, _ in words))

    print("\nDemo complete.")


if __name__ == "__main__":
    main()
