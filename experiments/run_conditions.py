"""
Full Evaluation Pipeline across Conditions A, B, and C with Multi-Seed Runs.

Conditions:
- Condition A (Clean Baseline): 80 clean training documents (20 per category).
- Condition B (Adversarial Poisoning): 80 clean + 10 injected poison documents targeting Category 0.
  Variants:
  - B_repeat: Keyword repetition of top target words.
  - B_bridge: Bridging vocabulary between Category 0 and Category 1.
- Condition C (Random Control): 80 clean + 10 random vocabulary documents.
- Defenses:
  - Cosine Deduplication Filter on Condition B.
  - Tempered EM (beta=0.85) on Condition B.

Evaluates on 20 held-out test documents across 5 stratified seeds.
"""

import sys
import os
import json
from typing import Dict, List, Tuple
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import StratifiedShuffleSplit

from src.plsa import PLSA
from src.align import TopicAligner
from src.preprocess import TextPreprocessor
from src.attack import AdversarialPoisonGenerator
from src.defense import DefenseFilter
from src.metrics import compute_classification_metrics, compute_purity, compute_nmi, compute_attack_potency
from src.retrieval import IRBenchmark


def run_experiment_pipeline(
    seeds: List[int] = [42, 43, 44, 45, 46],
    n_topics: int = 4,
    n_train_per_cat: int = 20,
    n_test_per_cat: int = 5,
    n_poison_docs: int = 10,
    output_dir: str = "results",
):
    os.makedirs(output_dir, exist_ok=True)
    categories = [
        "sci.space",
        "comp.graphics",
        "rec.sport.baseball",
        "talk.politics.mideast",
    ]

    print("Loading 20 Newsgroups corpus pool (4 categories)...")
    dataset = fetch_20newsgroups(
        subset="all",
        categories=categories,
        remove=("headers", "footers", "quotes"),
        random_state=42,
    )

    all_texts = np.array(dataset.data)
    all_targets = np.array(dataset.target)

    # Filter out empty or trivially short documents (< 50 chars)
    valid_mask = np.array([len(t.strip()) > 50 for t in all_texts])
    all_texts = all_texts[valid_mask]
    all_targets = all_targets[valid_mask]

    # Pre-select balanced pool of 25 docs per category = 100 documents total per split
    # Container for all run results across seeds
    condition_records = {
        "Condition A (Clean)": [],
        "Condition B (Poison - Repetition)": [],
        "Condition B (Poison - Bridging)": [],
        "Condition C (Random Control)": [],
        "Defense: Deduplication (on B)": [],
        "Defense: Tempered EM (on B)": [],
    }

    print(f"\nStarting evaluation across {len(seeds)} random seeds: {seeds}")

    for seed_idx, seed in enumerate(seeds):
        print(f"\n--- Seed {seed_idx + 1}/{len(seeds)} (Seed: {seed}) ---")
        rng = np.random.RandomState(seed)

        # Sample exactly 25 docs per category
        selected_indices = []
        for c in range(len(categories)):
            cat_idx = np.where(all_targets == c)[0]
            chosen = rng.choice(cat_idx, size=(n_train_per_cat + n_test_per_cat), replace=False)
            selected_indices.extend(chosen)
        selected_indices = np.array(selected_indices)

        sub_texts = all_texts[selected_indices]
        sub_labels = all_targets[selected_indices]

        # Stratified 80/20 train/test split
        sss = StratifiedShuffleSplit(n_splits=1, test_size=(len(categories) * n_test_per_cat), random_state=seed)
        train_idx, test_idx = next(sss.split(sub_texts, sub_labels))

        train_texts = list(sub_texts[train_idx])
        y_train = sub_labels[train_idx]
        test_texts = list(sub_texts[test_idx])
        y_test = sub_labels[test_idx]

        # Fit preprocessor strictly on clean training split
        preprocessor = TextPreprocessor(min_df=2, max_df_ratio=0.85)
        X_train_clean = preprocessor.fit_transform(train_texts)
        X_test = preprocessor.transform(test_texts)
        vocab = preprocessor.vocab_

        # ----------------------------------------------------
        # 1. Condition A (Clean Baseline)
        # ----------------------------------------------------
        plsa_a = PLSA(n_topics=n_topics, max_iter=40, beta=1.0, random_state=seed)
        plsa_a.fit(X_train_clean)

        train_topics_a = np.argmax(plsa_a.p_z_d, axis=1)
        aligner_a = TopicAligner(n_topics=n_topics).fit(train_topics_a, y_train)

        p_z_test_a = plsa_a.transform(X_test)
        test_topics_a = np.argmax(p_z_test_a, axis=1)
        test_preds_a = aligner_a.predict(test_topics_a)

        metrics_a = compute_classification_metrics(y_test, test_preds_a)
        perp_a = plsa_a.perplexity(X_test, p_z_test_a)
        retrieval_a = IRBenchmark.evaluate_retrieval(p_z_test_a, plsa_a.p_z_d, y_test, y_train)

        condition_records["Condition A (Clean)"].append({
            "Macro-F1": metrics_a["macro_f1"],
            "Accuracy": metrics_a["accuracy"],
            "Purity": compute_purity(y_test, test_topics_a),
            "NMI": compute_nmi(y_test, test_topics_a),
            "Perplexity": perp_a,
            "MAP": retrieval_a["MAP"],
            "P@5": retrieval_a["P@5"],
        })

        # Identify target topic for category 0
        target_topic = aligner_a.class_to_topic_[0]
        # Top 20 words for target topic
        top_words_k = [w for w, _ in plsa_a.get_top_words(vocab, n_top_words=20)[target_topic]]
        # Confuser topic for category 1
        confuser_topic = aligner_a.class_to_topic_[1]
        confuser_words_k = [w for w, _ in plsa_a.get_top_words(vocab, n_top_words=20)[confuser_topic]]

        # Generator
        gen = AdversarialPoisonGenerator(vocab, random_state=seed)

        # ----------------------------------------------------
        # 2. Condition B (Poison - Repetition)
        # ----------------------------------------------------
        poison_docs_rep = gen.generate_keyword_repetition_attack(
            target_words=top_words_k, n_docs=n_poison_docs, doc_length=80
        )
        X_poison_rep = preprocessor.transform(poison_docs_rep)
        X_train_b_rep = np.vstack([X_train_clean, X_poison_rep])

        plsa_b_rep = PLSA(n_topics=n_topics, max_iter=40, beta=1.0, random_state=seed)
        plsa_b_rep.fit(X_train_b_rep)

        # Align using clean train documents only
        clean_train_topics_b = np.argmax(plsa_b_rep.p_z_d[:len(train_texts)], axis=1)
        aligner_b_rep = TopicAligner(n_topics=n_topics).fit(clean_train_topics_b, y_train)

        p_z_test_b = plsa_b_rep.transform(X_test)
        test_topics_b = np.argmax(p_z_test_b, axis=1)
        test_preds_b = aligner_b_rep.predict(test_topics_b)

        metrics_b = compute_classification_metrics(y_test, test_preds_b)
        perp_b = plsa_b_rep.perplexity(X_test, p_z_test_b)
        retrieval_b = IRBenchmark.evaluate_retrieval(p_z_test_b, plsa_b_rep.p_z_d[:len(train_texts)], y_test, y_train)

        condition_records["Condition B (Poison - Repetition)"].append({
            "Macro-F1": metrics_b["macro_f1"],
            "Accuracy": metrics_b["accuracy"],
            "Purity": compute_purity(y_test, test_topics_b),
            "NMI": compute_nmi(y_test, test_topics_b),
            "Perplexity": perp_b,
            "MAP": retrieval_b["MAP"],
            "P@5": retrieval_b["P@5"],
        })

        # ----------------------------------------------------
        # 3. Condition B (Poison - Bridging)
        # ----------------------------------------------------
        poison_docs_br = gen.generate_bridging_attack(
            target_words=top_words_k, confuser_words=confuser_words_k, n_docs=n_poison_docs, doc_length=80
        )
        X_poison_br = preprocessor.transform(poison_docs_br)
        X_train_b_br = np.vstack([X_train_clean, X_poison_br])

        plsa_b_br = PLSA(n_topics=n_topics, max_iter=40, beta=1.0, random_state=seed)
        plsa_b_br.fit(X_train_b_br)

        clean_train_topics_br = np.argmax(plsa_b_br.p_z_d[:len(train_texts)], axis=1)
        aligner_b_br = TopicAligner(n_topics=n_topics).fit(clean_train_topics_br, y_train)

        p_z_test_br = plsa_b_br.transform(X_test)
        test_topics_br = np.argmax(p_z_test_br, axis=1)
        test_preds_br = aligner_b_br.predict(test_topics_br)

        metrics_br = compute_classification_metrics(y_test, test_preds_br)
        perp_br = plsa_b_br.perplexity(X_test, p_z_test_br)
        retrieval_br = IRBenchmark.evaluate_retrieval(p_z_test_br, plsa_b_br.p_z_d[:len(train_texts)], y_test, y_train)

        condition_records["Condition B (Poison - Bridging)"].append({
            "Macro-F1": metrics_br["macro_f1"],
            "Accuracy": metrics_br["accuracy"],
            "Purity": compute_purity(y_test, test_topics_br),
            "NMI": compute_nmi(y_test, test_topics_br),
            "Perplexity": perp_br,
            "MAP": retrieval_br["MAP"],
            "P@5": retrieval_br["P@5"],
        })

        # ----------------------------------------------------
        # 4. Condition C (Random Control)
        # ----------------------------------------------------
        control_docs = gen.generate_random_control(n_docs=n_poison_docs, doc_length=80)
        X_control = preprocessor.transform(control_docs)
        X_train_c = np.vstack([X_train_clean, X_control])

        plsa_c = PLSA(n_topics=n_topics, max_iter=40, beta=1.0, random_state=seed)
        plsa_c.fit(X_train_c)

        clean_train_topics_c = np.argmax(plsa_c.p_z_d[:len(train_texts)], axis=1)
        aligner_c = TopicAligner(n_topics=n_topics).fit(clean_train_topics_c, y_train)

        p_z_test_c = plsa_c.transform(X_test)
        test_topics_c = np.argmax(p_z_test_c, axis=1)
        test_preds_c = aligner_c.predict(test_topics_c)

        metrics_c = compute_classification_metrics(y_test, test_preds_c)
        perp_c = plsa_c.perplexity(X_test, p_z_test_c)
        retrieval_c = IRBenchmark.evaluate_retrieval(p_z_test_c, plsa_c.p_z_d[:len(train_texts)], y_test, y_train)

        condition_records["Condition C (Random Control)"].append({
            "Macro-F1": metrics_c["macro_f1"],
            "Accuracy": metrics_c["accuracy"],
            "Purity": compute_purity(y_test, test_topics_c),
            "NMI": compute_nmi(y_test, test_topics_c),
            "Perplexity": perp_c,
            "MAP": retrieval_c["MAP"],
            "P@5": retrieval_c["P@5"],
        })

        # ----------------------------------------------------
        # 5. Defense: Deduplication Filter on Condition B (Repetition)
        # ----------------------------------------------------
        X_deduped, kept_ids = DefenseFilter.deduplicate_documents(X_train_b_rep, threshold=0.75)
        plsa_def_dedup = PLSA(n_topics=n_topics, max_iter=40, beta=1.0, random_state=seed)
        plsa_def_dedup.fit(X_deduped)

        # Align with retained clean training documents
        clean_kept = [i for i in kept_ids if i < len(train_texts)]
        clean_topics_dedup = np.argmax(plsa_def_dedup.p_z_d[:len(clean_kept)], axis=1)
        aligner_dedup = TopicAligner(n_topics=n_topics).fit(clean_topics_dedup, y_train[clean_kept])

        p_z_test_dedup = plsa_def_dedup.transform(X_test)
        test_topics_dedup = np.argmax(p_z_test_dedup, axis=1)
        test_preds_dedup = aligner_dedup.predict(test_topics_dedup)

        metrics_dedup = compute_classification_metrics(y_test, test_preds_dedup)
        perp_dedup = plsa_def_dedup.perplexity(X_test, p_z_test_dedup)
        retrieval_dedup = IRBenchmark.evaluate_retrieval(
            p_z_test_dedup, plsa_def_dedup.p_z_d[:len(clean_kept)], y_test, y_train[clean_kept]
        )

        condition_records["Defense: Deduplication (on B)"].append({
            "Macro-F1": metrics_dedup["macro_f1"],
            "Accuracy": metrics_dedup["accuracy"],
            "Purity": compute_purity(y_test, test_topics_dedup),
            "NMI": compute_nmi(y_test, test_topics_dedup),
            "Perplexity": perp_dedup,
            "MAP": retrieval_dedup["MAP"],
            "P@5": retrieval_dedup["P@5"],
        })

        # ----------------------------------------------------
        # 6. Defense: Tempered EM on Condition B (beta=0.85)
        # ----------------------------------------------------
        plsa_temp_b = PLSA(n_topics=n_topics, max_iter=40, beta=0.85, random_state=seed)
        plsa_temp_b.fit(X_train_b_rep)

        clean_topics_temp = np.argmax(plsa_temp_b.p_z_d[:len(train_texts)], axis=1)
        aligner_temp = TopicAligner(n_topics=n_topics).fit(clean_topics_temp, y_train)

        p_z_test_temp = plsa_temp_b.transform(X_test)
        test_topics_temp = np.argmax(p_z_test_temp, axis=1)
        test_preds_temp = aligner_temp.predict(test_topics_temp)

        metrics_temp = compute_classification_metrics(y_test, test_preds_temp)
        perp_temp = plsa_temp_b.perplexity(X_test, p_z_test_temp)
        retrieval_temp = IRBenchmark.evaluate_retrieval(
            p_z_test_temp, plsa_temp_b.p_z_d[:len(train_texts)], y_test, y_train
        )

        condition_records["Defense: Tempered EM (on B)"].append({
            "Macro-F1": metrics_temp["macro_f1"],
            "Accuracy": metrics_temp["accuracy"],
            "Purity": compute_purity(y_test, test_topics_temp),
            "NMI": compute_nmi(y_test, test_topics_temp),
            "Perplexity": perp_temp,
            "MAP": retrieval_temp["MAP"],
            "P@5": retrieval_temp["P@5"],
        })

    # Summary Statistics: Mean ± Std across seeds
    summary_table = {}
    print("\n" + "=" * 96)
    print("EXPERIMENTAL RESULTS: CONDITIONS A, B, C AND DEFENSES (MEAN ± STD ACROSS 5 SEEDS)")
    print("=" * 96)
    header = f"{'Condition':<34} | {'Macro-F1':<16} | {'Purity':<16} | {'MAP':<16} | {'Perplexity':<16}"
    print(header)
    print("-" * 96)

    for cond, runs in condition_records.items():
        f1_vals = [r["Macro-F1"] for r in runs]
        pur_vals = [r["Purity"] for r in runs]
        map_vals = [r["MAP"] for r in runs]
        perp_vals = [r["Perplexity"] for r in runs]

        f1_str = f"{np.mean(f1_vals):.3f} ± {np.std(f1_vals):.3f}"
        pur_str = f"{np.mean(pur_vals):.3f} ± {np.std(pur_vals):.3f}"
        map_str = f"{np.mean(map_vals):.3f} ± {np.std(map_vals):.3f}"
        perp_str = f"{np.mean(perp_vals):.1f} ± {np.std(perp_vals):.1f}"

        print(f"{cond:<34} | {f1_str:<16} | {pur_str:<16} | {map_str:<16} | {perp_str:<16}")

        summary_table[cond] = {
            "Macro-F1_mean": float(np.mean(f1_vals)),
            "Macro-F1_std": float(np.std(f1_vals)),
            "Purity_mean": float(np.mean(pur_vals)),
            "Purity_std": float(np.std(pur_vals)),
            "MAP_mean": float(np.mean(map_vals)),
            "MAP_std": float(np.std(map_vals)),
            "Perplexity_mean": float(np.mean(perp_vals)),
            "Perplexity_std": float(np.std(perp_vals)),
            "raw_runs": runs,
        }
    print("=" * 96)

    # Save to results/condition_results.json
    results_path = os.path.join(output_dir, "condition_results.json")
    with open(results_path, "w") as f:
        json.dump(summary_table, f, indent=2)
    print(f"\nSaved detailed results to {results_path}")

    return summary_table


if __name__ == "__main__":
    run_experiment_pipeline()
