"""
Defense mechanisms against adversarial topic poisoning:
1. Cosine similarity deduplication / near-duplicate filtering.
2. Per-document log-likelihood trimming / outlier detection.
3. MAP (Maximum A Posteriori) Dirichlet pseudo-count smoothing.
"""

from typing import List, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class DefenseFilter:
    """
    Data hygiene and filtering defenses prior to model fitting.
    """

    @staticmethod
    def deduplicate_documents(
        X: np.ndarray,
        threshold: float = 0.85,
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Detects and filters out near-duplicate documents with cosine similarity > threshold.

        Returns:
            X_filtered: Filtered count matrix.
            kept_indices: Indices of documents retained.
        """
        D = X.shape[0]
        # Normalize rows for cosine similarity
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        X_norm = X / norms

        sim_matrix = np.dot(X_norm, X_norm.T)

        kept_indices = []
        dropped_indices = set()

        for i in range(D):
            if i in dropped_indices:
                continue
            kept_indices.append(i)
            # Find duplicate neighbors
            dupes = np.where(sim_matrix[i] > threshold)[0]
            for d in dupes:
                if d != i:
                    dropped_indices.add(d)

        return X[kept_indices], kept_indices

    @staticmethod
    def trim_outliers_by_likelihood(
        X: np.ndarray,
        per_doc_ll: np.ndarray,
        percentile_cutoff: float = 5.0,
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Trims documents in the bottom percentile of log-likelihood fit.
        """
        cutoff_val = np.percentile(per_doc_ll, percentile_cutoff)
        kept_indices = [i for i, val in enumerate(per_doc_ll) if val >= cutoff_val]
        return X[kept_indices], kept_indices
