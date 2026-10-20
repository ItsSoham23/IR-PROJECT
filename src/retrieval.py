"""
Information Retrieval Evaluation and Ranking Module.

Implements:
- Ranked retrieval: PLSA topic-mixture similarity, TF-IDF cosine, BM25, and LSI.
- Evaluation metrics: Precision@K, Recall@K, MAP, NDCG@K.
- Latency and memory benchmarking.
"""

import time
import tracemalloc
from typing import Dict, List, Optional, Set, Tuple, Union
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def compute_precision_at_k(retrieved_ids: List[int], relevant_ids: Set[int], k: int) -> float:
    """Precision at rank K."""
    if k <= 0:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return hits / k


def compute_recall_at_k(retrieved_ids: List[int], relevant_ids: Set[int], k: int) -> float:
    """Recall at rank K."""
    if not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return hits / len(relevant_ids)


def compute_average_precision(retrieved_ids: List[int], relevant_ids: Set[int]) -> float:
    """Average Precision (AP) for a single query."""
    if not relevant_ids:
        return 0.0

    hits = 0
    sum_precisions = 0.0
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            hits += 1
            sum_precisions += hits / rank

    return sum_precisions / len(relevant_ids)


def compute_ndcg_at_k(retrieved_ids: List[int], relevant_ids: Set[int], k: int) -> float:
    """NDCG at rank K with binary relevance."""
    if not relevant_ids or k <= 0:
        return 0.0

    top_k = retrieved_ids[:k]
    dcg = 0.0
    for rank, doc_id in enumerate(top_k, start=1):
        rel = 1.0 if doc_id in relevant_ids else 0.0
        dcg += rel / np.log2(rank + 1)

    # Ideal DCG
    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1.0 / np.log2(rank + 1) for rank in range(1, ideal_hits + 1))

    return dcg / idcg if idcg > 0 else 0.0


class IRBenchmark:
    """
    Evaluates IR retrieval performance across a set of queries and ranked methods.
    """

    @staticmethod
    def evaluate_retrieval(
        query_embeddings: np.ndarray,
        doc_embeddings: np.ndarray,
        query_labels: np.ndarray,
        doc_labels: np.ndarray,
        k_values: Tuple[int, ...] = (5, 10),
    ) -> Dict[str, float]:
        """
        Computes Mean Precision@K, Mean Recall@K, MAP, and Mean NDCG@K
        where documents sharing the same category as the query are defined as relevant.
        """
        # Similarity matrix: (n_queries, n_docs)
        sim_matrix = cosine_similarity(query_embeddings, doc_embeddings)

        n_queries = len(query_labels)
        aps = []
        p_at_k = {k: [] for k in k_values}
        r_at_k = {k: [] for k in k_values}
        ndcg_at_k = {k: [] for k in k_values}

        for q_idx in range(n_queries):
            q_label = query_labels[q_idx]
            # Relevant docs in corpus
            rel_docs = set(np.where(doc_labels == q_label)[0])
            if not rel_docs:
                continue

            # Ranked retrieved doc ids
            ranked_ids = list(np.argsort(sim_matrix[q_idx])[::-1])

            aps.append(compute_average_precision(ranked_ids, rel_docs))
            for k in k_values:
                p_at_k[k].append(compute_precision_at_k(ranked_ids, rel_docs, k))
                r_at_k[k].append(compute_recall_at_k(ranked_ids, rel_docs, k))
                ndcg_at_k[k].append(compute_ndcg_at_k(ranked_ids, rel_docs, k))

        results = {
            "MAP": float(np.mean(aps)) if aps else 0.0,
        }
        for k in k_values:
            results[f"P@{k}"] = float(np.mean(p_at_k[k])) if p_at_k[k] else 0.0
            results[f"R@{k}"] = float(np.mean(r_at_k[k])) if r_at_k[k] else 0.0
            results[f"NDCG@{k}"] = float(np.mean(ndcg_at_k[k])) if ndcg_at_k[k] else 0.0

        return results


def benchmark_resource_usage(func, *args, **kwargs) -> Tuple[any, float, float]:
    """
    Measures execution wall-clock time (seconds) and peak memory (MB).
    """
    tracemalloc.start()
    start_time = time.perf_counter()

    result = func(*args, **kwargs)

    elapsed_time = time.perf_counter() - start_time
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    peak_mb = peak_bytes / (1024 * 1024)
    return result, elapsed_time, peak_mb
