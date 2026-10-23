"""
Step 1 – PLSA vs. LDA (Gensim) Baseline Comparison.

Trains Gensim LDA on the same 80‑doc training split under all three
adversarial conditions (A – clean, B – bridging & repeat, C – random control),
evaluates on the 20‑doc held‑out test set, and compares results against the
previously computed PLSA metrics.

Usage
-----
    python experiments/lda_baseline.py
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gensim.corpora import Dictionary
from gensim.models import LdaModel

from src.dataset import prepare_100_document_corpus
from src.plsa import PLSA
from src.align import TopicAligner
from src.attack import AdversarialPoisonGenerator
from src.defense import DefenseFilter
from src.preprocess import TextPreprocessor
from src.metrics import (
    compute_classification_metrics,
    compute_purity,
    compute_nmi,
    compute_npmi_coherence,
)
from src.coherence import npmi
from src.retrieval import IRBenchmark


# ── helpers ────────────────────────────────────────────────────────────
def _count_to_bow(X_row, vocab):
    """Convert a single dense count‑vector to Gensim bag‑of‑words."""
    return [(int(j), int(X_row[j])) for j in np.nonzero(X_row)[0]]


def _count_matrix_to_corpus(X, vocab):
    """Convert (D, V) count matrix to Gensim‑compatible list‑of‑bow."""
    return [_count_to_bow(X[i], vocab) for i in range(X.shape[0])]


def _infer_topic_proportions(lda, bow_corpus, n_topics):
    """Get dense (D, K) topic‑proportion matrix from a Gensim LdaModel."""
    D = len(bow_corpus)
    p_z_d = np.zeros((D, n_topics))
    for i, bow in enumerate(bow_corpus):
        for tid, prob in lda.get_document_topics(bow, minimum_probability=0):
            p_z_d[i, tid] = prob
    return p_z_d


def _lda_perplexity(lda, bow_corpus):
    """Approximate per‑word perplexity (Gensim provides log_perplexity)."""
    lp = lda.log_perplexity(bow_corpus)      # log2 perplexity per word
    return float(np.exp2(-lp))


# ── main ───────────────────────────────────────────────────────────────
def run_lda_experiments(
    seeds=(42, 101, 202, 303, 404),
    n_topics=4,
    output_dir="results",
):
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 85)
    print("LDA BASELINE EXPERIMENTS (GENSIM)")
    print("=" * 85)

    condition_names = [
        "LDA – Cond A (Clean)",
        "LDA – Cond B (Repetition)",
        "LDA – Cond B (Bridging)",
        "LDA – Cond C (Random Control)",
    ]
    records = {n: [] for n in condition_names}

    # Also run PLSA under the same seeds so we have an aligned comparison
    plsa_records = {
        "PLSA – Cond A (Clean)": [],
        "PLSA – Cond B (Repetition)": [],
        "PLSA – Cond B (Bridging)": [],
        "PLSA – Cond C (Random Control)": [],
    }

    target_cat_idx = 0  # Campus
    confuser_cat_idx = 1  # Politics

    for si, seed in enumerate(seeds):
        print(f"\n>>> Seed {si+1}/{len(seeds)} (seed={seed}) <<<")

        # ── data ───────────────────────────────────────────────────
        X_train, X_test, y_train, y_test, vocab, train_docs, test_docs = \
            prepare_100_document_corpus(seed=seed, output_dir="data/processed")
        with open("data/processed/cleaned_corpus.json", encoding="utf-8") as handle:
            corpus_tokens = [item["tokens"] for item in json.load(handle)]

        n_train = X_train.shape[0]
        preprocessor = TextPreprocessor(min_df=2, max_df_ratio=0.85)
        preprocessor.vocab_ = vocab
        preprocessor.word_to_idx_ = {w: i for i, w in enumerate(vocab)}

        gen = AdversarialPoisonGenerator(vocab, random_state=seed)

        # Build Gensim dictionary from our fixed vocab
        id2word = {i: w for i, w in enumerate(vocab)}
        gensim_dict = Dictionary()
        gensim_dict.id2token = id2word
        gensim_dict.token2id = {w: i for i, w in id2word.items()}

        # ── helper: train + eval for a (X_train_cond, condition_name) pair ─
        def _run_condition(X_tr, cond_lda, cond_plsa):
            bow_train = _count_matrix_to_corpus(X_tr, vocab)
            bow_test = _count_matrix_to_corpus(X_test, vocab)

            # ---- LDA ----
            lda = LdaModel(
                corpus=bow_train,
                id2word=gensim_dict,
                num_topics=n_topics,
                passes=30,
                alpha="auto",
                eta="auto",
                random_state=seed,
            )
            if cond_lda == "LDA – Cond A (Clean)" and si == 0:
                lda.save(os.path.join(output_dir, "lda.model"))
                np.save(os.path.join(output_dir, "lda_topic_word.npy"), lda.get_topics())
            pz_train = _infer_topic_proportions(lda, bow_train, n_topics)
            pz_test  = _infer_topic_proportions(lda, bow_test, n_topics)

            # align on clean train docs only
            clean_topics = np.argmax(pz_train[:n_train], axis=1)
            aligner = TopicAligner(n_topics=n_topics).fit(clean_topics, y_train)

            test_topics = np.argmax(pz_test, axis=1)
            test_preds  = aligner.predict(test_topics)

            clf = compute_classification_metrics(y_test, test_preds)
            perp = _lda_perplexity(lda, bow_test)
            ret  = IRBenchmark.evaluate_retrieval(pz_test, pz_train[:n_train], y_test, y_train)

            # top words for NPMI
            top_words = {}
            for k in range(n_topics):
                tw = lda.show_topic(k, topn=10)
                top_words[k] = [(w, float(p)) for w, p in tw]

            records[cond_lda].append({
                "Macro-F1": clf["macro_f1"], "Accuracy": clf["accuracy"],
                "Purity": compute_purity(y_test, test_topics),
                "NMI": compute_nmi(y_test, test_topics),
                "Perplexity": perp, "MAP": ret["MAP"], "P@5": ret["P@5"],
                "npmi_mean": npmi(
                    {k: [word for word, _ in words] for k, words in top_words.items()},
                    corpus_tokens,
                    vocab,
                ),
            })

            # ---- PLSA (same data) ----
            plsa = PLSA(n_topics=n_topics, max_iter=50, tol=1e-5, beta=1.0, random_state=seed)
            plsa.fit(X_tr)
            pz_train_p = plsa.p_z_d
            pz_test_p  = plsa.transform(X_test)

            clean_topics_p = np.argmax(pz_train_p[:n_train], axis=1)
            aligner_p = TopicAligner(n_topics=n_topics).fit(clean_topics_p, y_train)
            test_topics_p = np.argmax(pz_test_p, axis=1)
            test_preds_p  = aligner_p.predict(test_topics_p)

            clf_p = compute_classification_metrics(y_test, test_preds_p)
            perp_p = plsa.perplexity(X_test, pz_test_p)
            ret_p  = IRBenchmark.evaluate_retrieval(pz_test_p, pz_train_p[:n_train], y_test, y_train)

            plsa_records[cond_plsa].append({
                "Macro-F1": clf_p["macro_f1"], "Accuracy": clf_p["accuracy"],
                "Purity": compute_purity(y_test, test_topics_p),
                "NMI": compute_nmi(y_test, test_topics_p),
                "Perplexity": perp_p, "MAP": ret_p["MAP"], "P@5": ret_p["P@5"],
                "npmi_mean": npmi(
                    {
                        k: [word for word, _ in words]
                        for k, words in plsa.get_top_words(vocab, 10).items()
                    },
                    corpus_tokens,
                    vocab,
                ),
            })

        # ── Condition A (clean) ────────────────────────────────────
        _run_condition(X_train,
                       "LDA – Cond A (Clean)", "PLSA – Cond A (Clean)")

        # ── extract target / confuser keywords for poisoning ───────
        plsa_a = PLSA(n_topics=n_topics, max_iter=50, tol=1e-5, random_state=seed)
        plsa_a.fit(X_train)
        train_topics_a = np.argmax(plsa_a.p_z_d, axis=1)
        aligner_a = TopicAligner(n_topics=n_topics).fit(train_topics_a, y_train)
        target_topic   = aligner_a.class_to_topic_[target_cat_idx]
        confuser_topic = aligner_a.class_to_topic_[confuser_cat_idx]
        target_top   = [w for w, _ in plsa_a.get_top_words(vocab, 20)[target_topic]]
        confuser_top = [w for w, _ in plsa_a.get_top_words(vocab, 20)[confuser_topic]]

        # ── Condition B – Repetition ───────────────────────────────
        poison_rep = gen.generate_keyword_repetition_attack(target_top, 10, 80, 0.05)
        X_poison_rep = preprocessor.transform(poison_rep)
        X_b_rep = np.vstack([X_train, X_poison_rep])
        _run_condition(X_b_rep,
                       "LDA – Cond B (Repetition)", "PLSA – Cond B (Repetition)")

        # ── Condition B – Bridging ─────────────────────────────────
        poison_br = gen.generate_bridging_attack(target_top, confuser_top, 10, 80, 0.5)
        X_poison_br = preprocessor.transform(poison_br)
        X_b_br = np.vstack([X_train, X_poison_br])
        _run_condition(X_b_br,
                       "LDA – Cond B (Bridging)", "PLSA – Cond B (Bridging)")

        # ── Condition C – Random control ───────────────────────────
        control = gen.generate_random_control(10, 80)
        X_ctrl = preprocessor.transform(control)
        X_c = np.vstack([X_train, X_ctrl])
        _run_condition(X_c,
                       "LDA – Cond C (Random Control)", "PLSA – Cond C (Random Control)")

    # ── aggregate & print ──────────────────────────────────────────
    all_records = {**records, **plsa_records}
    summary = {}
    print("\n" + "=" * 110)
    print("LDA vs PLSA  —  MEAN ± STD OVER 5 SEEDS")
    print("=" * 110)
    header = f"{'Condition':<35} | {'Macro-F1':>10} | {'Purity':>10} | {'MAP':>10} | {'Perplexity':>12}"
    print(header)
    print("-" * 110)

    for name in list(records.keys()) + list(plsa_records.keys()):
        runs = all_records[name]
        f1s  = [r["Macro-F1"] for r in runs]
        purs = [r["Purity"] for r in runs]
        maps = [r["MAP"] for r in runs]
        perps= [r["Perplexity"] for r in runs]
        f1m, f1s_ = np.mean(f1s), np.std(f1s)
        pm, ps = np.mean(purs), np.std(purs)
        mm, ms = np.mean(maps), np.std(maps)
        pp, pps = np.mean(perps), np.std(perps)
        print(f"{name:<35} | {f1m:.3f}±{f1s_:.3f} | {pm:.3f}±{ps:.3f} | {mm:.3f}±{ms:.3f} | {pp:.1f}±{pps:.1f}")
        summary[name] = {
            "Macro-F1_mean": float(f1m), "Macro-F1_std": float(f1s_),
            "Purity_mean": float(pm), "MAP_mean": float(mm),
            "Perplexity_mean": float(pp), "runs": runs,
            "npmi_mean": float(np.mean([r["npmi_mean"] for r in runs])),
            "npmi_std": float(np.std([r["npmi_mean"] for r in runs])),
        }
    print("=" * 110)

    # ── save JSON ──────────────────────────────────────────────────
    out = os.path.join(output_dir, "lda_summary.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved -> {out}")

    # ── comparison plot ────────────────────────────────────────────
    condition_keys = [
        ("Clean", "A", "Clean"),
        ("Repetition", "B", "Repetition"),
        ("Bridging", "B", "Bridging"),
        ("Random", "C", "Random Control"),
    ]
    conds_short = [item[0] for item in condition_keys]
    lda_f1 = [summary[f"LDA – Cond {code} ({label})"]["Macro-F1_mean"] for _, code, label in condition_keys]
    plsa_f1 = [summary[f"PLSA – Cond {code} ({label})"]["Macro-F1_mean"] for _, code, label in condition_keys]
    lda_map = [summary[f"LDA – Cond {code} ({label})"]["MAP_mean"] for _, code, label in condition_keys]
    plsa_map = [summary[f"PLSA – Cond {code} ({label})"]["MAP_mean"] for _, code, label in condition_keys]

    x = np.arange(len(conds_short))
    w = 0.2
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=200)

    axes[0].bar(x - w/2, plsa_f1, w, label="PLSA", color="#1f77b4")
    axes[0].bar(x + w/2, lda_f1, w, label="LDA", color="#2ca02c")
    axes[0].set_ylabel("Macro‑F1"); axes[0].set_title("Classification: PLSA vs LDA")
    axes[0].set_xticks(x); axes[0].set_xticklabels(conds_short)
    axes[0].legend(); axes[0].set_ylim(0, 1); axes[0].grid(axis="y", ls="--", alpha=.4)

    axes[1].bar(x - w/2, plsa_map, w, label="PLSA", color="#1f77b4")
    axes[1].bar(x + w/2, lda_map, w, label="LDA", color="#2ca02c")
    axes[1].set_ylabel("MAP"); axes[1].set_title("Retrieval MAP: PLSA vs LDA")
    axes[1].set_xticks(x); axes[1].set_xticklabels(conds_short)
    axes[1].legend(); axes[1].set_ylim(0, 1); axes[1].grid(axis="y", ls="--", alpha=.4)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "lda_vs_plsa.png")
    plt.savefig(plot_path); plt.close()
    print(f"Plot  -> {plot_path}")

    return summary


if __name__ == "__main__":
    run_lda_experiments()
