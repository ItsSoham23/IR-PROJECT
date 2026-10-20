"""
Evaluation metrics for PLSA topic modeling, classification, and adversarial stress testing.

Includes:
- Classification: Macro Precision, Recall, F1, Accuracy
- Clustering: Purity, Normalized Mutual Information (NMI)
- Topic Quality: NPMI (Normalized Pointwise Mutual Information) coherence
- Adversarial Potency metrics:
  * Injected word mass on target topic
  * Shift of non-target test documents toward target topic
  * Relative log-likelihood degradation
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


def compute_attack_potency(
    p_w_z: np.ndarray,
    target_topic_idx: int,
    injected_word_indices: List[int],
    test_p_z_d: np.ndarray,
    baseline_test_p_z_d: Optional[np.ndarray] = None,
    non_target_mask: Optional[np.ndarray] = None,
    train_ll: Optional[float] = None,
    baseline_train_ll: Optional[float] = None,
) -> Dict[str, float]:
    """
    Compute adversarial potency quantities defined in the plan:
    1. Target-topic probability mass concentrated on the 20 injected poison words:
       Mass = sum_{w in injected} P(w | target_topic)
    2. Average test-document shift toward the target topic (especially non-target documents):
       Mean( P(target_topic | d) ) or change vs clean baseline.
    3. Change in log-likelihood (Delta LL = LL_poisoned - LL_baseline).
    """
    # 1. Word mass concentration
    target_p_w = p_w_z[target_topic_idx]
    injected_mass = float(np.sum(target_p_w[injected_word_indices]))

    # 2. Topic shift on held-out documents
    if non_target_mask is not None:
        filtered_p_z_d = test_p_z_d[non_target_mask]
    else:
        filtered_p_z_d = test_p_z_d

    avg_target_prob = float(np.mean(filtered_p_z_d[:, target_topic_idx]))

    prob_shift = 0.0
    if baseline_test_p_z_d is not None:
        base_filtered = (
            baseline_test_p_z_d[non_target_mask]
            if non_target_mask is not None
            else baseline_test_p_z_d
        )
        base_target_prob = float(np.mean(base_filtered[:, target_topic_idx]))
        prob_shift = avg_target_prob - base_target_prob

    # 3. Delta LL
    delta_ll = 0.0
    if train_ll is not None and baseline_train_ll is not None:
        delta_ll = train_ll - baseline_train_ll

    return {
        "injected_words_target_mass": injected_mass,
        "avg_test_target_prob": avg_target_prob,
        "target_prob_shift": prob_shift,
        "delta_log_likelihood": delta_ll,
    }
