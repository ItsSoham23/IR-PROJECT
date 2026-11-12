"""
Evaluation metrics for PLSA topic modeling, classification, and adversarial stress testing.

Includes:
- Classification: Macro Precision, Recall, F1, Accuracy
- Clustering: Purity, Normalized Mutual Information (NMI)
- Topic Quality: NPMI (Normalized Pointwise Mutual Information) coherence
- Adversarial Potency metrics:
  * Injected word mass on target topic
  * Shift of non-target test documents toward target topic
  * Common-corpus log-likelihood change per token
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, normalized_mutual_info_score


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Compute macro-averaged Precision, Recall, F1 and Accuracy.
    """
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    acc = accuracy_score(y_true, y_pred)
    return {
        "macro_precision": float(prec),
        "macro_recall": float(rec),
        "macro_f1": float(f1),
        "accuracy": float(acc),
    }


def compute_purity(y_true: np.ndarray, cluster_assignments: np.ndarray) -> float:
    """
    Compute clustering purity:
    Purity = (1 / N) * sum_k max_c |w_k cap c_c|
    """
    contingency = np.zeros(
        (np.max(cluster_assignments) + 1, np.max(y_true) + 1), dtype=np.int64
    )
    for k, c in zip(cluster_assignments, y_true):
        contingency[k, c] += 1
    return float(np.sum(np.amax(contingency, axis=1)) / len(y_true))


def compute_nmi(y_true: np.ndarray, cluster_assignments: np.ndarray) -> float:
    """
    Compute Normalized Mutual Information (NMI).
    """
    return float(normalized_mutual_info_score(y_true, cluster_assignments))


def compute_npmi_coherence(
    top_words_per_topic: Dict[int, List[Tuple[str, float]]],
    corpus_docs: List[List[str]],
    top_n: int = 10,
    eps: float = 1e-12,
) -> Dict[str, Union[float, Dict[int, float]]]:
    """
    Compute Normalized Pointwise Mutual Information (NPMI) coherence for each topic
    using co-occurrence frequencies over documents.

    NPMI(w_i, w_j) = log( (P(w_i, w_j) + eps) / (P(w_i) * P(w_j) + eps) ) / ( -log(P(w_i, w_j) + eps) )
    """
    total_docs = len(corpus_docs)
    if total_docs == 0:
        return {"mean_npmi": 0.0, "topic_npmi": {}}

    # Pre-calculate document frequencies for words in top_n
    doc_sets = [set(doc) for doc in corpus_docs]

    topic_npmi = {}
    all_npmis = []

    for topic_idx, words_probs in top_words_per_topic.items():
        words = [w for w, _ in words_probs[:top_n]]
        M = len(words)
        if M < 2:
            topic_npmi[topic_idx] = 0.0
            continue

        pair_scores = []
        for i in range(M):
            w_i = words[i]
            # Doc frequency of w_i
            df_i = sum(1 for d in doc_sets if w_i in d)
            p_i = (df_i + eps) / total_docs

            for j in range(i + 1, M):
                w_j = words[j]
                df_j = sum(1 for d in doc_sets if w_j in d)
                p_j = (df_j + eps) / total_docs

                # Co-occurrence
                df_ij = sum(1 for d in doc_sets if (w_i in d and w_j in d))
                p_ij = (df_ij + eps) / total_docs

                # NPMI formula
                pmi = np.log(p_ij / (p_i * p_j))
                norm = -np.log(p_ij)
                npmi = pmi / norm if norm > 0 else 0.0
                pair_scores.append(npmi)

        avg_npmi = float(np.mean(pair_scores)) if pair_scores else 0.0
        topic_npmi[topic_idx] = avg_npmi
        all_npmis.append(avg_npmi)

    return {
        "mean_npmi": float(np.mean(all_npmis)) if all_npmis else 0.0,
        "topic_npmi": topic_npmi,
    }


def mean_log_likelihood_per_token(counts, p_w_z, p_z_d):
    """Conditional log-likelihood in nats/token on one explicit evaluation matrix.

    L(X; model) = sum_dw X_dw log(sum_k theta_dk phi_kw) / sum_dw X_dw.
    Use identical X (rows, vocabulary and counts) for model comparisons. Document
    proportions must come from the same inference protocol, not training rows
    from datasets of different sizes. A common 1e-12 probability floor is used.
    """
    from scipy import sparse
    total = float(counts.sum())
    if total <= 0:
        raise ValueError("Evaluation corpus must contain tokens")
    probabilities = np.maximum(np.asarray(p_z_d) @ np.asarray(p_w_z), 1e-12)
    if probabilities.shape != counts.shape:
        raise ValueError("Evaluation counts and model probabilities must have identical shapes")
    if sparse.issparse(counts):
        coo = counts.tocoo()
        likelihood = np.sum(coo.data * np.log(probabilities[coo.row, coo.col]))
    else:
        likelihood = np.sum(counts * np.log(probabilities))
    return float(likelihood / total)


def compute_attack_potency(
    model,
    baseline_model,
    *,
    aligner,
    baseline_aligner,
    target_class_idx: int,
    injected_word_indices: List[int],
    test_p_z_d: np.ndarray,
    baseline_test_p_z_d: np.ndarray,
    evaluation_counts: np.ndarray,
    non_target_mask: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Measure semantic target drift and common-corpus likelihood change.

    Word mass uses this model's target-class topic. Test probabilities are
    independently aligned to class order before comparison. For likelihood,
    BOTH frozen models fold in the same clean training matrix X, including the
    same vocabulary and token counts; their augmented training objectives are
    never used. Delta = L(X; condition) - L(X; baseline), in nats/token.
    Negative delta means worse fit to this shared corpus; this is an in-sample
    diagnostic, not an independent generalization score or proof of attack success.
    """
    target_topic = aligner.class_to_topic_[target_class_idx]
    injected_mass = float(np.sum(model.p_w_z[target_topic, injected_word_indices]))
    aligned = aligner.align_proportions(test_p_z_d)
    baseline_aligned = baseline_aligner.align_proportions(baseline_test_p_z_d)
    if aligned.shape != baseline_aligned.shape:
        raise ValueError("Both conditions must evaluate the same test documents/classes")
    mask = non_target_mask if non_target_mask is not None else np.ones(len(aligned), dtype=bool)
    if not np.any(mask):
        raise ValueError("No test documents selected for target drift")
    target_probability = float(np.mean(aligned[mask, target_class_idx]))
    baseline_probability = float(np.mean(baseline_aligned[mask, target_class_idx]))
    likelihood = mean_log_likelihood_per_token(
        evaluation_counts, model.p_w_z, model.transform(evaluation_counts))
    baseline_likelihood = mean_log_likelihood_per_token(
        evaluation_counts, baseline_model.p_w_z, baseline_model.transform(evaluation_counts))
    return {
        "injected_words_target_mass": injected_mass,
        "avg_test_target_prob": target_probability,
        "target_prob_shift": target_probability - baseline_probability,
        "eval_ll_per_token": likelihood,
        "baseline_eval_ll_per_token": baseline_likelihood,
        "delta_eval_ll_per_token": likelihood - baseline_likelihood,
    }
