import numpy as np
import pytest

from src.align import TopicAligner
from src.metrics import compute_attack_potency, mean_log_likelihood_per_token


class FrozenModel:
    def __init__(self, phi, theta):
        self.p_w_z = np.asarray(phi, dtype=float)
        self.theta = np.asarray(theta, dtype=float)
        self.seen = []
        # Deliberately incomparable objectives; potency must never read them.
        self.final_log_likelihood_ = -999999.0

    def transform(self, counts):
        self.seen.append(counts.copy())
        return self.theta


def test_potency_compares_classes_despite_topic_permutation():
    labels = np.array([0, 1, 2])
    clean_aligner = TopicAligner(3).fit(np.array([2, 0, 1]), labels)
    condition_aligner = TopicAligner(3).fit(np.array([0, 1, 2]), labels)
    # Rows in class order: Campus, Politics, Sports.
    phi = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
    theta = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
    counts = np.array([[8., 1., 1.], [1., 8., 1.], [1., 1., 8.]])
    clean = FrozenModel(phi[[1, 2, 0]], theta[:, [1, 2, 0]])
    condition = FrozenModel(phi, theta)
    clean.final_log_likelihood_ = -2.0
    condition.p_z_d = np.ones((13, 3)) / 3  # Augmented training rows are irrelevant.
    result = compute_attack_potency(
        condition, clean, aligner=condition_aligner, baseline_aligner=clean_aligner,
        target_class_idx=0, injected_word_indices=[0],
        test_p_z_d=np.array([[0.3, 0.6, 0.1]]),
        baseline_test_p_z_d=np.array([[0.8, 0.1, 0.1]]),
        evaluation_counts=counts)
    assert result["injected_words_target_mass"] == pytest.approx(0.8)
    assert result["target_prob_shift"] == pytest.approx(0.2)
    assert result["delta_eval_ll_per_token"] == pytest.approx(0, abs=1e-12)
    np.testing.assert_array_equal(clean.seen[0], counts)
    np.testing.assert_array_equal(condition.seen[0], counts)


def test_common_corpus_metric_math_and_size_invariance():
    counts = np.array([[3., 1.]])
    phi = np.array([[0.75, 0.25]])
    theta = np.array([[1.]])
    expected = (3 * np.log(.75) + np.log(.25)) / 4
    assert mean_log_likelihood_per_token(counts, phi, theta) == pytest.approx(expected)
    assert mean_log_likelihood_per_token(np.vstack([counts] * 3), phi,
                                         np.vstack([theta] * 3)) == pytest.approx(expected)
    worse = mean_log_likelihood_per_token(counts, np.array([[.25, .75]]), theta)
    assert worse - expected < 0
    with pytest.raises(ValueError, match="contain tokens"):
        mean_log_likelihood_per_token(np.zeros((1, 2)), phi, theta)
