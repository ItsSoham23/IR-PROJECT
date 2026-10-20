"""
Stage 0 Mini-Replication on 20 Newsgroups.

Validates the from-scratch PLSA implementation on a 4-category public subset:
- 'sci.space'
- 'comp.graphics'
- 'rec.sport.baseball'
- 'talk.politics.mideast'

Compares:
- Standard PLSA (beta=1.0)
- Tempered PLSA (beta=0.85)
- LSI baseline (TruncatedSVD + KMeans)
- TF-IDF + KMeans baseline

Evaluates both topic classification (Macro-F1, Purity, NMI) and IR ranking (MAP, P@5, P@10).
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer

from src.plsa import PLSA
from src.align import TopicAligner
from src.baselines import LSIBaseline, TFIDFBaseline
from src.metrics import compute_classification_metrics, compute_purity, compute_nmi
from src.retrieval import IRBenchmark, benchmark_resource_usage


def run_stage0(max_docs_per_cat: int = 75, n_topics: int = 4, random_state: int = 42):
    print("=" * 70)
    print("STAGE 0: Mini-Replication on 20 Newsgroups (4 Categories)")
    print("=" * 70)

    categories = [
        "sci.space",
        "comp.graphics",
        "rec.sport.baseball",
        "talk.politics.mideast",
    ]

    print("Fetching 20 Newsgroups dataset...")
    # Fetch train split
    train_data = fetch_20newsgroups(
        subset="train",
        categories=categories,
        remove=("headers", "footers", "quotes"),
        random_state=random_state,
    )
    # Fetch test split
    test_data = fetch_20newsgroups(
        subset="test",
        categories=categories,
        remove=("headers", "footers", "quotes"),
        random_state=random_state,
    )

    # Subsample balanced documents per category for fast, representative execution
    train_indices = []
    for c_idx in range(len(categories)):
        cat_indices = np.where(train_data.target == c_idx)[0]
        train_indices.extend(cat_indices[:max_docs_per_cat])
    train_indices = np.array(train_indices)

    test_indices = []
    for c_idx in range(len(categories)):
        cat_indices = np.where(test_data.target == c_idx)[0]
        test_indices.extend(cat_indices[: max_docs_per_cat // 2])
    test_indices = np.array(test_indices)

    train_texts = [train_data.data[i] for i in train_indices]
    y_train = train_data.target[train_indices]

    test_texts = [test_data.data[i] for i in test_indices]
    y_test = test_data.target[test_indices]

    print(f"Training documents: {len(train_texts)} ({len(categories)} categories x {max_docs_per_cat})")
    print(f"Test documents:     {len(test_texts)}")

    # Vectorize strictly using training vocabulary
    vectorizer = CountVectorizer(
        max_df=0.85,
        min_df=3,
        max_features=1200,
        stop_words="english",
        token_pattern=r"\b[a-zA-Z]{3,}\b",
    )
    X_train = vectorizer.fit_transform(train_texts).toarray()
    X_test = vectorizer.transform(test_texts).toarray()
    vocab = vectorizer.get_feature_names_out()

    print(f"Vocabulary size:    {len(vocab)}")
    print("-" * 70)

    results = []

    # 1. Standard PLSA (beta=1.0)
    print("\n[1/4] Fitting Standard PLSA (beta=1.0)...")
    plsa_std = PLSA(n_topics=n_topics, max_iter=40, tol=1e-5, beta=1.0, random_state=random_state)
    (_, fit_time, mem_mb) = benchmark_resource_usage(plsa_std.fit, X_train)

    train_topics = np.argmax(plsa_std.p_z_d, axis=1)
    aligner_std = TopicAligner(n_topics=n_topics, n_classes=n_topics).fit(train_topics, y_train)

    # Folding-in on test documents
    p_z_test_std = plsa_std.transform(X_test)
    test_topics = np.argmax(p_z_test_std, axis=1)
    test_preds = aligner_std.predict(test_topics)

    clf_metrics = compute_classification_metrics(y_test, test_preds)
    test_perp = plsa_std.perplexity(X_test, p_z_test_std)
    retrieval_res = IRBenchmark.evaluate_retrieval(p_z_test_std, plsa_std.p_z_d, y_test, y_train)

    results.append({
        "Model": "Standard PLSA (beta=1.0)",
        "Macro-F1": clf_metrics["macro_f1"],
        "Accuracy": clf_metrics["accuracy"],
        "Purity": compute_purity(y_test, test_topics),
        "NMI": compute_nmi(y_test, test_topics),
        "Perplexity": test_perp,
        "MAP": retrieval_res["MAP"],
        "P@5": retrieval_res["P@5"],
        "Fit Time (s)": fit_time,
        "Peak Mem (MB)": mem_mb,
    })

    # Print top words
    print("Standard PLSA Top-5 Words per Topic:")
    for top_idx, words in plsa_std.get_top_words(vocab, n_top_words=5).items():
        aligned_class = aligner_std.topic_to_class_.get(top_idx, -1)
        cat_name = categories[aligned_class] if aligned_class >= 0 else "unaligned"
        word_str = ", ".join([w for w, _ in words])
        print(f"  Topic {top_idx} (aligned -> {cat_name}): {word_str}")

    # 2. Tempered PLSA (beta=0.85)
    print("\n[2/4] Fitting Tempered PLSA (beta=0.85)...")
    plsa_temp = PLSA(n_topics=n_topics, max_iter=40, tol=1e-5, beta=0.85, random_state=random_state)
    (_, fit_time_t, mem_mb_t) = benchmark_resource_usage(plsa_temp.fit, X_train)

    train_topics_t = np.argmax(plsa_temp.p_z_d, axis=1)
    aligner_t = TopicAligner(n_topics=n_topics, n_classes=n_topics).fit(train_topics_t, y_train)

    p_z_test_t = plsa_temp.transform(X_test)
    test_topics_t = np.argmax(p_z_test_t, axis=1)
    test_preds_t = aligner_t.predict(test_topics_t)

    clf_metrics_t = compute_classification_metrics(y_test, test_preds_t)
    test_perp_t = plsa_temp.perplexity(X_test, p_z_test_t)
    retrieval_res_t = IRBenchmark.evaluate_retrieval(p_z_test_t, plsa_temp.p_z_d, y_test, y_train)

    results.append({
        "Model": "Tempered PLSA (beta=0.85)",
        "Macro-F1": clf_metrics_t["macro_f1"],
        "Accuracy": clf_metrics_t["accuracy"],
        "Purity": compute_purity(y_test, test_topics_t),
        "NMI": compute_nmi(y_test, test_topics_t),
        "Perplexity": test_perp_t,
        "MAP": retrieval_res_t["MAP"],
        "P@5": retrieval_res_t["P@5"],
        "Fit Time (s)": fit_time_t,
        "Peak Mem (MB)": mem_mb_t,
    })

    # 3. LSI Baseline
    print("\n[3/4] Fitting LSI Baseline (TruncatedSVD + KMeans)...")
    lsi = LSIBaseline(n_components=n_topics, random_state=random_state)
    (_, fit_time_lsi, mem_mb_lsi) = benchmark_resource_usage(lsi.fit, X_train, y_train)
    lsi_preds = lsi.predict(X_test)
    lsi_metrics = compute_classification_metrics(y_test, lsi_preds)

    # Retrieval using LSI document embeddings
    train_lsi_emb = lsi.transform(X_train)
    test_lsi_emb = lsi.transform(X_test)
    lsi_retrieval = IRBenchmark.evaluate_retrieval(test_lsi_emb, train_lsi_emb, y_test, y_train)

    results.append({
        "Model": "LSI Baseline",
        "Macro-F1": lsi_metrics["macro_f1"],
        "Accuracy": lsi_metrics["accuracy"],
        "Purity": compute_purity(y_test, lsi_preds),
        "NMI": compute_nmi(y_test, lsi_preds),
        "Perplexity": float("nan"),
        "MAP": lsi_retrieval["MAP"],
        "P@5": lsi_retrieval["P@5"],
        "Fit Time (s)": fit_time_lsi,
        "Peak Mem (MB)": mem_mb_lsi,
    })

    # 4. TF-IDF + KMeans Baseline
    print("\n[4/4] Fitting TF-IDF + KMeans Baseline...")
    tfidf_bm = TFIDFBaseline(n_clusters=n_topics, random_state=random_state)
    (_, fit_time_tf, mem_mb_tf) = benchmark_resource_usage(tfidf_bm.fit, X_train, y_train)
    tfidf_preds = tfidf_bm.predict(X_test)
    tfidf_metrics = compute_classification_metrics(y_test, tfidf_preds)

    train_tfidf_emb = tfidf_bm.tfidf.transform(X_train).toarray()
    test_tfidf_emb = tfidf_bm.tfidf.transform(X_test).toarray()
    tfidf_retrieval = IRBenchmark.evaluate_retrieval(test_tfidf_emb, train_tfidf_emb, y_test, y_train)

    results.append({
        "Model": "TF-IDF + KMeans",
        "Macro-F1": tfidf_metrics["macro_f1"],
        "Accuracy": tfidf_metrics["accuracy"],
        "Purity": compute_purity(y_test, tfidf_preds),
        "NMI": compute_nmi(y_test, tfidf_preds),
        "Perplexity": float("nan"),
        "MAP": tfidf_retrieval["MAP"],
        "P@5": tfidf_retrieval["P@5"],
        "Fit Time (s)": fit_time_tf,
        "Peak Mem (MB)": mem_mb_tf,
    })

    # Print summary table
    print("\n" + "=" * 90)
    print("STAGE 0 COMPARISON RESULTS")
    print("=" * 90)
    header = f"{'Model':<28} | {'Macro-F1':<8} | {'Purity':<8} | {'NMI':<8} | {'Perp':<10} | {'MAP':<8} | {'Time (s)':<8}"
    print(header)
    print("-" * 90)
    for r in results:
        perp_str = f"{r['Perplexity']:.1f}" if not np.isnan(r['Perplexity']) else "N/A"
        print(f"{r['Model']:<28} | {r['Macro-F1']:<8.4f} | {r['Purity']:<8.4f} | {r['NMI']:<8.4f} | {perp_str:<10} | {r['MAP']:<8.4f} | {r['Fit Time (s)']:<8.3f}")
    print("=" * 90)

    return results


if __name__ == "__main__":
    run_stage0()
