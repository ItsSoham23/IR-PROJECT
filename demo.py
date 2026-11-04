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
from src.dataset import prepare_100_document_corpus
from src.metrics import compute_classification_metrics
from src.plsa import PLSA
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
    print("\nTraining a 4-topic PLSA model with EM...")

    model = PLSA(
        n_topics=4,
        max_iter=60,
        tol=1e-5,
        n_restarts=1,
        random_state=42,
    )
    model.fit(X_train)

    train_topics = np.argmax(model.p_z_d, axis=1)
    aligner = TopicAligner(n_topics=4).fit(train_topics, y_train)
    test_proportions = model.transform(X_test)
    test_topics = np.argmax(test_proportions, axis=1)
    predictions = aligner.predict(test_topics)
    metrics = compute_classification_metrics(y_test, predictions)
    retrieval = IRBenchmark.evaluate_retrieval(
        test_proportions, model.p_z_d, y_test, y_train
    )

    print(f"EM iterations:      {model.n_iter_}")
    print(f"Converged:           {model.converged_}")
    print(f"Held-out Macro-F1:   {metrics['macro_f1']:.3f}")
    print(f"Held-out accuracy:   {metrics['accuracy']:.3f}")
    print(f"Retrieval MAP:       {retrieval['MAP']:.3f}")
    print(f"Held-out perplexity: {model.perplexity(X_test, test_proportions):.1f}")

    print("\nLearned topics (top words):")
    for topic, words in model.get_top_words(vocab, n_top_words=8).items():
        category = aligner.topic_to_class_.get(topic, "?")
        print(f"  Topic {topic} -> category {category}: " + ", ".join(word for word, _ in words))

    print("\nDemo complete.")


if __name__ == "__main__":
    main()
