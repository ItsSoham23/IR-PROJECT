"""
Topic-to-Category Alignment using Hungarian Algorithm (linear_sum_assignment).

Computes the optimal one-to-one mapping between learned latent topics
and ground-truth document categories to maximize alignment agreement.
"""

from typing import Dict, Optional, Tuple, Union
import numpy as np
from scipy.optimize import linear_sum_assignment


class TopicAligner:
    """
    Aligns K latent topics to C ground-truth category labels via the Hungarian algorithm.
    """

    def __init__(self, n_topics: int = 4, n_classes: Optional[int] = None):
        self.n_topics = n_topics
        self.n_classes = n_classes if n_classes is not None else n_topics
        # Mapping from topic index (0..K-1) to category index (0..C-1)
        self.topic_to_class_: Dict[int, int] = {}
        # Mapping from category index to topic index
        self.class_to_topic_: Dict[int, int] = {}
        self.contingency_matrix_: Optional[np.ndarray] = None

    def fit(
        self,
        dominant_topics: np.ndarray,
        ground_truth_labels: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> "TopicAligner":
        """
        Fit the optimal one-to-one topic-to-class alignment on real documents.

        Args:
            dominant_topics: Array of shape (D,) with topic indices in [0, K-1].
            ground_truth_labels: Array of shape (D,) with category labels in [0, C-1].
            mask: Optional boolean array of shape (D,) where True indicates valid real documents
                  (excludes injected poison/control documents).
        """
        if mask is not None:
            dominant_topics = dominant_topics[mask]
            ground_truth_labels = ground_truth_labels[mask]

        D = len(dominant_topics)
        if D == 0:
            raise ValueError("No documents provided for alignment.")

        # Build contingency matrix: contingency[k, c] = count of docs with topic k and label c
        contingency = np.zeros((self.n_topics, self.n_classes), dtype=np.int64)
        for k, c in zip(dominant_topics, ground_truth_labels):
            if 0 <= k < self.n_topics and 0 <= c < self.n_classes:
                contingency[k, c] += 1

        self.contingency_matrix_ = contingency

        # Cost matrix = -contingency because linear_sum_assignment minimizes cost
        cost_matrix = -contingency
        topic_ind, class_ind = linear_sum_assignment(cost_matrix)

        self.topic_to_class_ = {int(t): int(c) for t, c in zip(topic_ind, class_ind)}
        self.class_to_topic_ = {int(c): int(t) for t, c in zip(topic_ind, class_ind)}

        return self

    def predict(self, dominant_topics: np.ndarray) -> np.ndarray:
        """
        Map an array of topic indices to their aligned category labels.
        """
        if not self.topic_to_class_:
            raise ValueError("Aligner must be fitted before predicting.")

        return np.array([self.topic_to_class_.get(int(t), -1) for t in dominant_topics])

    def align_proportions(self, p_z_d: np.ndarray) -> np.ndarray:
        """
        Reorder topic proportions matrix P(z|d) of shape (D, K) to match class indices (D, C).
        """
        if not self.class_to_topic_:
            raise ValueError("Aligner must be fitted before reordering.")

        # For each class c, retrieve the corresponding topic k
        reordered = np.zeros((p_z_d.shape[0], self.n_classes), dtype=p_z_d.dtype)
        for c in range(self.n_classes):
            if c in self.class_to_topic_:
                t = self.class_to_topic_[c]
                reordered[:, c] = p_z_d[:, t]
        return reordered
