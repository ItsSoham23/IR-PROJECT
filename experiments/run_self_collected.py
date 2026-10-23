"""
Full Experimental Pipeline and Adversarial Stress Test on Self-Collected Corpus.

Dataset: 100 documents across 4 categories ('Campus', 'Politics', 'Sports', 'Technology')
- 80 training documents (20 per category)
- 20 test documents (5 per category)

Conditions:
- Condition A: Clean Baseline (80 docs)
- Condition B_repeat: 80 clean + 10 keyword repetition poison documents targeting Category 0
- Condition B_bridge: 80 clean + 10 bridging poison documents (Category 0 + Category 1)
- Condition C: 80 clean + 10 random vocabulary control documents
- Defenses on B: Deduplication filtering, Tempered EM (beta=0.85), Likelihood trimming

Multi-seed evaluation across 5 seeds with Bootstrap 95% Confidence Intervals,
metric plots, and complete results logging.
"""

import sys
import os
import json
import time
from typing import Dict, List, Tuple
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt

from src.dataset import prepare_100_document_corpus
from src.plsa import PLSA
from src.align import TopicAligner
from src.baselines import LSIBaseline, TFIDFBaseline
from src.attack import AdversarialPoisonGenerator
from src.defense import DefenseFilter
from src.preprocess import TextPreprocessor
from src.metrics import (
    compute_classification_metrics,
    compute_purity,
    compute_nmi,
    compute_attack_potency,
    compute_npmi_coherence,
)
from src.retrieval import IRBenchmark, benchmark_resource_usage
from src.coherence import npmi

MODEL_MAX_ITER = 1000
MODEL_RESTARTS = 5


def bootstrap_ci(data: List[float], n_boot: int = 1000, ci: float = 0.95) -> Tuple[float, float]:
    """Calculate non-parametric bootstrap confidence interval."""
    if len(data) == 0:
        return 0.0, 0.0
    arr = np.array(data)
    rng = np.random.RandomState(42)
    boot_means = [np.mean(rng.choice(arr, size=len(arr), replace=True)) for _ in range(n_boot)]
    lower = float(np.percentile(boot_means, (1.0 - ci) / 2.0 * 100))
    upper = float(np.percentile(boot_means, (1.0 + ci) / 2.0 * 100))
    return lower, upper


def run_self_collected_experiments(
    seeds: List[int] = [42, 101, 202, 303, 404],
    n_topics: int = 4,
    output_dir: str = "results",
):
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 85)
    print("EXPERIMENTS ON SELF-COLLECTED CORPUS (100 DOCS, 4 CATEGORIES)")
    print("=" * 85)

    condition_names = [
        "Condition A (Clean)",
        "Condition B (Poison - Repetition)",
        "Condition B (Poison - Bridging)",
        "Condition C (Random Control)",
        "Defense: Deduplication (on B)",
        "Defense: Tempered EM (on B)",
        "Baseline: LSI",
        "Baseline: TF-IDF + KMeans",
    ]

    records = {name: [] for name in condition_names}
    potency_records = {"B_repeat": [], "B_bridge": [], "Condition_C": []}

    target_cat_idx = 0  # Campus / Category 0
    confuser_cat_idx = 1  # Politics / Category 1

    sample_top_words = None

    for seed_idx, seed in enumerate(seeds):
        print(f"\n>>> Running Seed {seed_idx + 1}/{len(seeds)} (Seed: {seed}) <<<")
        # Prepare stratified split for this seed
        X_train, X_test, y_train, y_test, vocab, train_docs, test_docs = prepare_100_document_corpus(
            seed=seed, output_dir=os.path.join("data", "processed")
        )
        with open(os.path.join("data", "processed", "cleaned_corpus.json"), encoding="utf-8") as handle:
            corpus_tokens = [item["tokens"] for item in json.load(handle)]

        n_train = len(train_docs)
        n_test = len(test_docs)
        gen = AdversarialPoisonGenerator(vocab, random_state=seed)

        # -------------------------------------------------------------------
        # 1. Condition A (Clean Baseline PLSA)
        # -------------------------------------------------------------------
        plsa_a = PLSA(n_topics=n_topics, max_iter=MODEL_MAX_ITER, tol=1e-6, n_restarts=MODEL_RESTARTS, beta=1.0, random_state=seed)
        _, fit_time_a, mem_mb_a = benchmark_resource_usage(plsa_a.fit, X_train)

        train_topics_a = np.argmax(plsa_a.p_z_d, axis=1)
        aligner_a = TopicAligner(n_topics=n_topics).fit(train_topics_a, y_train)

        p_z_test_a = plsa_a.transform(X_test)
        test_topics_a = np.argmax(p_z_test_a, axis=1)
        test_preds_a = aligner_a.predict(test_topics_a)

        clf_a = compute_classification_metrics(y_test, test_preds_a)
        perp_a = plsa_a.perplexity(X_test, p_z_test_a)
        retrieval_a = IRBenchmark.evaluate_retrieval(p_z_test_a, plsa_a.p_z_d, y_test, y_train)

        records["Condition A (Clean)"].append({
            "Macro-F1": clf_a["macro_f1"],
            "Accuracy": clf_a["accuracy"],
            "Purity": compute_purity(y_test, test_topics_a),
            "NMI": compute_nmi(y_test, test_topics_a),
            "NPMI": npmi(
                {k: [word for word, _ in words] for k, words in plsa_a.get_top_words(vocab, 10).items()},
                corpus_tokens,
                vocab,
            ),
            "Perplexity": perp_a,
            "MAP": retrieval_a["MAP"],
            "P@5": retrieval_a["P@5"],
            "Fit Time (s)": fit_time_a,
            "Peak Mem (MB)": mem_mb_a,
            "EM iterations": plsa_a.n_iter_,
            "EM converged": plsa_a.converged_,
        })

        if sample_top_words is None:
            sample_top_words = (plsa_a.get_top_words(vocab, n_top_words=8), aligner_a.topic_to_class_)

        # Extract target topic words
        target_topic = aligner_a.class_to_topic_[target_cat_idx]
        confuser_topic = aligner_a.class_to_topic_[confuser_cat_idx]
        target_top_words = [w for w, _ in plsa_a.get_top_words(vocab, n_top_words=20)[target_topic]]
        confuser_top_words = [w for w, _ in plsa_a.get_top_words(vocab, n_top_words=20)[confuser_topic]]
        target_word_indices = [gen.word_to_idx[w] for w in target_top_words if w in gen.word_to_idx]

        # -------------------------------------------------------------------
        # 2. Condition B (Poison - Repetition Attack)
        # -------------------------------------------------------------------
        poison_docs_rep = gen.generate_keyword_repetition_attack(
            target_words=target_top_words, n_docs=10, doc_length=80, noise_ratio=0.05
        )
        preprocessor = TextPreprocessor(min_df=2, max_df_ratio=0.85)
        preprocessor.vocab_ = vocab
        preprocessor.word_to_idx_ = {w: i for i, w in enumerate(vocab)}

        X_poison_rep = preprocessor.transform(poison_docs_rep)
        X_train_b_rep = np.vstack([X_train, X_poison_rep])

        plsa_b_rep = PLSA(n_topics=n_topics, max_iter=MODEL_MAX_ITER, tol=1e-6, n_restarts=MODEL_RESTARTS, beta=1.0, random_state=seed)
        _, fit_time_b, mem_mb_b = benchmark_resource_usage(plsa_b_rep.fit, X_train_b_rep)

        # Align with clean training documents only
        clean_train_topics_b = np.argmax(plsa_b_rep.p_z_d[:n_train], axis=1)
        aligner_b_rep = TopicAligner(n_topics=n_topics).fit(clean_train_topics_b, y_train)

        p_z_test_b = plsa_b_rep.transform(X_test)
        test_topics_b = np.argmax(p_z_test_b, axis=1)
        test_preds_b = aligner_b_rep.predict(test_topics_b)

        clf_b = compute_classification_metrics(y_test, test_preds_b)
        perp_b = plsa_b_rep.perplexity(X_test, p_z_test_b)
        retrieval_b = IRBenchmark.evaluate_retrieval(p_z_test_b, plsa_b_rep.p_z_d[:n_train], y_test, y_train)

        records["Condition B (Poison - Repetition)"].append({
            "Macro-F1": clf_b["macro_f1"],
            "Accuracy": clf_b["accuracy"],
            "Purity": compute_purity(y_test, test_topics_b),
            "NMI": compute_nmi(y_test, test_topics_b),
            "NPMI": npmi(
                {k: [word for word, _ in words] for k, words in plsa_b_rep.get_top_words(vocab, 10).items()},
                corpus_tokens,
                vocab,
            ),
            "Perplexity": perp_b,
            "MAP": retrieval_b["MAP"],
            "P@5": retrieval_b["P@5"],
            "Fit Time (s)": fit_time_b,
            "Peak Mem (MB)": mem_mb_b,
            "EM iterations": plsa_b_rep.n_iter_,
            "EM converged": plsa_b_rep.converged_,
        })

        # Attack potency
        non_target_mask = (y_test != target_cat_idx)
        potency_rep = compute_attack_potency(
            plsa_b_rep.p_w_z,
            target_topic,
            target_word_indices,
            p_z_test_b,
            baseline_test_p_z_d=p_z_test_a,
            non_target_mask=non_target_mask,
            train_ll=plsa_b_rep.final_log_likelihood_,
            baseline_train_ll=plsa_a.final_log_likelihood_,
        )
        potency_records["B_repeat"].append(potency_rep)

        # -------------------------------------------------------------------
        # 3. Condition B (Poison - Bridging Attack)
        # -------------------------------------------------------------------
        poison_docs_br = gen.generate_bridging_attack(
            target_words=target_top_words,
            confuser_words=confuser_top_words,
            n_docs=10,
            doc_length=80,
            target_ratio=0.5,
        )
        X_poison_br = preprocessor.transform(poison_docs_br)
        X_train_b_br = np.vstack([X_train, X_poison_br])

        plsa_b_br = PLSA(n_topics=n_topics, max_iter=MODEL_MAX_ITER, tol=1e-6, n_restarts=MODEL_RESTARTS, beta=1.0, random_state=seed)
        _, fit_time_br, mem_mb_br = benchmark_resource_usage(plsa_b_br.fit, X_train_b_br)

        clean_train_topics_br = np.argmax(plsa_b_br.p_z_d[:n_train], axis=1)
        aligner_b_br = TopicAligner(n_topics=n_topics).fit(clean_train_topics_br, y_train)

        p_z_test_br = plsa_b_br.transform(X_test)
        test_topics_br = np.argmax(p_z_test_br, axis=1)
        test_preds_br = aligner_b_br.predict(test_topics_br)

        clf_br = compute_classification_metrics(y_test, test_preds_br)
        perp_br = plsa_b_br.perplexity(X_test, p_z_test_br)
        retrieval_br = IRBenchmark.evaluate_retrieval(p_z_test_br, plsa_b_br.p_z_d[:n_train], y_test, y_train)

        records["Condition B (Poison - Bridging)"].append({
            "Macro-F1": clf_br["macro_f1"],
            "Accuracy": clf_br["accuracy"],
            "Purity": compute_purity(y_test, test_topics_br),
            "NMI": compute_nmi(y_test, test_topics_br),
            "NPMI": npmi(
                {k: [word for word, _ in words] for k, words in plsa_b_br.get_top_words(vocab, 10).items()},
                corpus_tokens,
                vocab,
            ),
            "Perplexity": perp_br,
            "MAP": retrieval_br["MAP"],
            "P@5": retrieval_br["P@5"],
            "Fit Time (s)": fit_time_br,
            "Peak Mem (MB)": mem_mb_br,
            "EM iterations": plsa_b_br.n_iter_,
            "EM converged": plsa_b_br.converged_,
        })

        potency_br = compute_attack_potency(
            plsa_b_br.p_w_z,
            target_topic,
            target_word_indices,
            p_z_test_br,
            baseline_test_p_z_d=p_z_test_a,
            non_target_mask=non_target_mask,
            train_ll=plsa_b_br.final_log_likelihood_,
            baseline_train_ll=plsa_a.final_log_likelihood_,
        )
        potency_records["B_bridge"].append(potency_br)

        # -------------------------------------------------------------------
        # 4. Condition C (Random Control)
        # -------------------------------------------------------------------
        control_docs = gen.generate_random_control(n_docs=10, doc_length=80)
        X_control = preprocessor.transform(control_docs)
        X_train_c = np.vstack([X_train, X_control])

        plsa_c = PLSA(n_topics=n_topics, max_iter=MODEL_MAX_ITER, tol=1e-6, n_restarts=MODEL_RESTARTS, beta=1.0, random_state=seed)
        _, fit_time_c, mem_mb_c = benchmark_resource_usage(plsa_c.fit, X_train_c)

        clean_train_topics_c = np.argmax(plsa_c.p_z_d[:n_train], axis=1)
        aligner_c = TopicAligner(n_topics=n_topics).fit(clean_train_topics_c, y_train)

        p_z_test_c = plsa_c.transform(X_test)
        test_topics_c = np.argmax(p_z_test_c, axis=1)
        test_preds_c = aligner_c.predict(test_topics_c)

        clf_c = compute_classification_metrics(y_test, test_preds_c)
        perp_c = plsa_c.perplexity(X_test, p_z_test_c)
        retrieval_c = IRBenchmark.evaluate_retrieval(p_z_test_c, plsa_c.p_z_d[:n_train], y_test, y_train)

        records["Condition C (Random Control)"].append({
            "Macro-F1": clf_c["macro_f1"],
            "Accuracy": clf_c["accuracy"],
            "Purity": compute_purity(y_test, test_topics_c),
            "NMI": compute_nmi(y_test, test_topics_c),
            "NPMI": npmi(
                {k: [word for word, _ in words] for k, words in plsa_c.get_top_words(vocab, 10).items()},
                corpus_tokens,
                vocab,
            ),
            "Perplexity": perp_c,
            "MAP": retrieval_c["MAP"],
            "P@5": retrieval_c["P@5"],
            "Fit Time (s)": fit_time_c,
            "Peak Mem (MB)": mem_mb_c,
            "EM iterations": plsa_c.n_iter_,
            "EM converged": plsa_c.converged_,
        })

        potency_c = compute_attack_potency(
            plsa_c.p_w_z,
            target_topic,
            target_word_indices,
            p_z_test_c,
            baseline_test_p_z_d=p_z_test_a,
            non_target_mask=non_target_mask,
            train_ll=plsa_c.final_log_likelihood_,
            baseline_train_ll=plsa_a.final_log_likelihood_,
        )
        potency_records["Condition_C"].append(potency_c)

        # -------------------------------------------------------------------
        # 5. Defense: Deduplication Filter on Condition B
        # -------------------------------------------------------------------
        X_deduped, kept_ids = DefenseFilter.deduplicate_documents(
            X_train_b_rep, threshold=0.75, protected_indices=list(range(n_train))
        )
        plsa_def_dedup = PLSA(n_topics=n_topics, max_iter=MODEL_MAX_ITER, tol=1e-6, n_restarts=MODEL_RESTARTS, beta=1.0, random_state=seed)
        _, fit_time_dedup, mem_mb_dedup = benchmark_resource_usage(plsa_def_dedup.fit, X_deduped)

        clean_kept = [i for i in kept_ids if i < n_train]
        clean_topics_dedup = np.argmax(plsa_def_dedup.p_z_d[:len(clean_kept)], axis=1)
        aligner_dedup = TopicAligner(n_topics=n_topics).fit(clean_topics_dedup, y_train[clean_kept])

        p_z_test_dedup = plsa_def_dedup.transform(X_test)
        test_topics_dedup = np.argmax(p_z_test_dedup, axis=1)
        test_preds_dedup = aligner_dedup.predict(test_topics_dedup)

        clf_dedup = compute_classification_metrics(y_test, test_preds_dedup)
        perp_dedup = plsa_def_dedup.perplexity(X_test, p_z_test_dedup)
        retrieval_dedup = IRBenchmark.evaluate_retrieval(
            p_z_test_dedup, plsa_def_dedup.p_z_d[:len(clean_kept)], y_test, y_train[clean_kept]
        )

        records["Defense: Deduplication (on B)"].append({
            "Macro-F1": clf_dedup["macro_f1"],
            "Accuracy": clf_dedup["accuracy"],
            "Purity": compute_purity(y_test, test_topics_dedup),
            "NMI": compute_nmi(y_test, test_topics_dedup),
            "NPMI": npmi(
                {k: [word for word, _ in words] for k, words in plsa_def_dedup.get_top_words(vocab, 10).items()},
                corpus_tokens,
                vocab,
            ),
            "Perplexity": perp_dedup,
            "MAP": retrieval_dedup["MAP"],
            "P@5": retrieval_dedup["P@5"],
            "Fit Time (s)": fit_time_dedup,
            "Peak Mem (MB)": mem_mb_dedup,
            "EM iterations": plsa_def_dedup.n_iter_,
            "EM converged": plsa_def_dedup.converged_,
        })

        # -------------------------------------------------------------------
        # 6. Defense: Tempered EM on Condition B (beta=0.85)
        # -------------------------------------------------------------------
        plsa_temp_b = PLSA(n_topics=n_topics, max_iter=MODEL_MAX_ITER, tol=1e-6, n_restarts=MODEL_RESTARTS, beta=0.85, random_state=seed)
        _, fit_time_temp, mem_mb_temp = benchmark_resource_usage(plsa_temp_b.fit, X_train_b_rep)

        clean_topics_temp = np.argmax(plsa_temp_b.p_z_d[:n_train], axis=1)
        aligner_temp = TopicAligner(n_topics=n_topics).fit(clean_topics_temp, y_train)

        p_z_test_temp = plsa_temp_b.transform(X_test)
        test_topics_temp = np.argmax(p_z_test_temp, axis=1)
        test_preds_temp = aligner_temp.predict(test_topics_temp)

        clf_temp = compute_classification_metrics(y_test, test_preds_temp)
        perp_temp = plsa_temp_b.perplexity(X_test, p_z_test_temp)
        retrieval_temp = IRBenchmark.evaluate_retrieval(
            p_z_test_temp, plsa_temp_b.p_z_d[:n_train], y_test, y_train
        )

        records["Defense: Tempered EM (on B)"].append({
            "Macro-F1": clf_temp["macro_f1"],
            "Accuracy": clf_temp["accuracy"],
            "Purity": compute_purity(y_test, test_topics_temp),
            "NMI": compute_nmi(y_test, test_topics_temp),
            "NPMI": npmi(
                {k: [word for word, _ in words] for k, words in plsa_temp_b.get_top_words(vocab, 10).items()},
                corpus_tokens,
                vocab,
            ),
            "Perplexity": perp_temp,
            "MAP": retrieval_temp["MAP"],
            "P@5": retrieval_temp["P@5"],
            "Fit Time (s)": fit_time_temp,
            "Peak Mem (MB)": mem_mb_temp,
            "EM iterations": plsa_temp_b.n_iter_,
            "EM converged": plsa_temp_b.converged_,
        })

        # -------------------------------------------------------------------
        # 7. Baseline: LSI
        # -------------------------------------------------------------------
        lsi = LSIBaseline(n_components=n_topics, random_state=seed)
        _, fit_time_lsi, mem_mb_lsi = benchmark_resource_usage(lsi.fit, X_train, y_train)
        lsi_preds = lsi.predict(X_test)
        clf_lsi = compute_classification_metrics(y_test, lsi_preds)

        train_lsi_emb = lsi.transform(X_train)
        test_lsi_emb = lsi.transform(X_test)
        retrieval_lsi = IRBenchmark.evaluate_retrieval(test_lsi_emb, train_lsi_emb, y_test, y_train)

        records["Baseline: LSI"].append({
            "Macro-F1": clf_lsi["macro_f1"],
            "Accuracy": clf_lsi["accuracy"],
            "Purity": compute_purity(y_test, lsi_preds),
            "NMI": compute_nmi(y_test, lsi_preds),
            "Perplexity": float("nan"),
            "MAP": retrieval_lsi["MAP"],
            "P@5": retrieval_lsi["P@5"],
            "Fit Time (s)": fit_time_lsi,
            "Peak Mem (MB)": mem_mb_lsi,
        })

        # -------------------------------------------------------------------
        # 8. Baseline: TF-IDF + KMeans
        # -------------------------------------------------------------------
        tfidf_bm = TFIDFBaseline(n_clusters=n_topics, random_state=seed)
        _, fit_time_tfidf, mem_mb_tfidf = benchmark_resource_usage(tfidf_bm.fit, X_train, y_train)
        tfidf_preds = tfidf_bm.predict(X_test)
        clf_tfidf = compute_classification_metrics(y_test, tfidf_preds)

        train_tfidf_emb = tfidf_bm.tfidf.transform(X_train).toarray()
        test_tfidf_emb = tfidf_bm.tfidf.transform(X_test).toarray()
        retrieval_tfidf = IRBenchmark.evaluate_retrieval(test_tfidf_emb, train_tfidf_emb, y_test, y_train)

        records["Baseline: TF-IDF + KMeans"].append({
            "Macro-F1": clf_tfidf["macro_f1"],
            "Accuracy": clf_tfidf["accuracy"],
            "Purity": compute_purity(y_test, tfidf_preds),
            "NMI": compute_nmi(y_test, tfidf_preds),
            "Perplexity": float("nan"),
            "MAP": retrieval_tfidf["MAP"],
            "P@5": retrieval_tfidf["P@5"],
            "Fit Time (s)": fit_time_tfidf,
            "Peak Mem (MB)": mem_mb_tfidf,
        })

    # =======================================================================
    # Print Comprehensive Results Table with Bootstrap 95% CIs
    # =======================================================================
    categories = ["Campus", "Politics", "Sports", "Technology"]
    print("\n" + "=" * 105)
    print("SELF-COLLECTED EXPERIMENT RESULTS (MEAN ± STD & 95% BOOTSTRAP CI OVER 5 SEEDS)")
    print("=" * 105)
    header = f"{'Method / Condition':<34} | {'Macro-F1 [95% CI]':<24} | {'Purity':<14} | {'MAP':<14} | {'Perplexity':<14}"
    print(header)
    print("-" * 105)

    summary_export = {}

    for name in condition_names:
        runs = records[name]
        f1s = [r["Macro-F1"] for r in runs]
        purities = [r["Purity"] for r in runs]
        maps = [r["MAP"] for r in runs]
        perps = [r["Perplexity"] for r in runs if not np.isnan(r["Perplexity"])]
        npmis = [r["NPMI"] for r in runs if "NPMI" in r]

        f1_mean, f1_std = np.mean(f1s), np.std(f1s)
        f1_ci_l, f1_ci_u = bootstrap_ci(f1s)

        pur_mean, pur_std = np.mean(purities), np.std(purities)
        map_mean, map_std = np.mean(maps), np.std(maps)

        perp_str = f"{np.mean(perps):.1f} ± {np.std(perps):.1f}" if perps else "N/A"
        f1_str = f"{f1_mean:.3f} [{f1_ci_l:.3f}, {f1_ci_u:.3f}]"

        print(f"{name:<34} | {f1_str:<24} | {pur_mean:.3f} ± {pur_std:.3f}  | {map_mean:.3f} ± {map_std:.3f}  | {perp_str:<14}")

        summary_export[name] = {
            "Macro-F1_mean": float(f1_mean),
            "Macro-F1_std": float(f1_std),
            "Macro-F1_ci95": [f1_ci_l, f1_ci_u],
            "Purity_mean": float(pur_mean),
            "Purity_std": float(pur_std),
            "MAP_mean": float(map_mean),
            "MAP_std": float(map_std),
            "Perplexity_mean": float(np.mean(perps)) if perps else None,
            "Perplexity_std": float(np.std(perps)) if perps else None,
            "NPMI_mean": float(np.mean(npmis)) if npmis else None,
            "NPMI_std": float(np.std(npmis)) if npmis else None,
            "runs": runs,
        }
    print("=" * 105)

    # Print Top Words Sample
    if sample_top_words:
        top_words_dict, topic_to_class = sample_top_words
        print("\nRecovered Top Words per Topic (Condition A, Seed 42):")
        for t_idx, w_list in top_words_dict.items():
            c_idx = topic_to_class.get(t_idx, -1)
            c_name = categories[c_idx] if c_idx >= 0 else "Unaligned"
            words_joined = ", ".join([w for w, _ in w_list])
            print(f"  Topic {t_idx} (Aligned -> {c_name}): {words_joined}")

    # Print Potency Table
    print("\n" + "=" * 90)
    print("ADVERSARIAL POTENCY QUANTITIES (MEAN OVER 5 SEEDS)")
    print("=" * 90)
    print(f"{'Condition':<30} | {'Injected Word Mass':<20} | {'Test Drift to Target':<22} | {'Delta LL':<16}")
    print("-" * 90)
    for k, pot_list in potency_records.items():
        masses = [p["injected_words_target_mass"] for p in pot_list]
        shifts = [p["target_prob_shift"] for p in pot_list]
        dlls = [p["delta_log_likelihood"] for p in pot_list]
        print(f"{k:<30} | {np.mean(masses):<20.4f} | {np.mean(shifts):<22.4f} | {np.mean(dlls):<16.1f}")
    print("=" * 90)

    # Save JSON summary
    summary_path = os.path.join(output_dir, "self_collected_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary_export, f, indent=2)
    print(f"\nSaved summary data to {summary_path}")

    # Generate Visualization Plot
    plot_path = os.path.join(output_dir, "condition_comparison.png")
    plot_results(summary_export, plot_path)
    print(f"Generated comparison plot at {plot_path}")

    return summary_export


def plot_results(summary: Dict, output_path: str):
    """Generate bar chart comparing Macro-F1 and MAP across conditions."""
    conditions = [
        "Cond A\n(Clean)",
        "Cond B\n(Repetition)",
        "Cond B\n(Bridging)",
        "Cond C\n(Control)",
        "Def: Dedup\n(on B)",
        "Def: Temp EM\n(on B)",
        "LSI\nBaseline",
        "TF-IDF\nBaseline",
    ]
    key_mapping = [
        "Condition A (Clean)",
        "Condition B (Poison - Repetition)",
        "Condition B (Poison - Bridging)",
        "Condition C (Random Control)",
        "Defense: Deduplication (on B)",
        "Defense: Tempered EM (on B)",
        "Baseline: LSI",
        "Baseline: TF-IDF + KMeans",
    ]

    f1_means = [summary[k]["Macro-F1_mean"] for k in key_mapping]
    f1_stds = [summary[k]["Macro-F1_std"] for k in key_mapping]
    map_means = [summary[k]["MAP_mean"] for k in key_mapping]
    map_stds = [summary[k]["MAP_std"] for k in key_mapping]

    x = np.arange(len(conditions))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=300)
    rects1 = ax.bar(x - width/2, f1_means, width, yerr=f1_stds, label="Classification Macro-F1", capsize=4, color="#1f77b4", alpha=0.88)
    rects2 = ax.bar(x + width/2, map_means, width, yerr=map_stds, label="Retrieval MAP", capsize=4, color="#ff7f0e", alpha=0.88)

    ax.set_ylabel("Score", fontsize=11, fontweight="bold")
    ax.set_title("PLSA Replication & Adversarial Stress Test: Performance Across Conditions", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, fontsize=9)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_ylim(0, 1.0)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


if __name__ == "__main__":
    run_self_collected_experiments()
