"""
Test that training log-likelihood is monotonically non-decreasing across standard EM iterations.
"""

import numpy as np
import pytest
from src.plsa import PLSA


@pytest.fixture
def test_dataset():
    rng = np.random.RandomState(123)
    # 30 docs, 60 words
    counts = rng.poisson(lam=3.0, size=(30, 60)).astype(np.float64)
    counts[counts.sum(axis=1) == 0, 0] = 1.0
    return counts


def test_asymmetric_em_monotone(test_dataset):
    model = PLSA(n_topics=3, max_iter=25, tol=1e-8, beta=1.0, random_state=42, asymmetric=True)
    model.fit(test_dataset)

    ll_history = model.log_likelihood_history_
    assert len(ll_history) > 1, "EM terminated in 1 iteration"

    # In theoretical EM, log-likelihood cannot decrease.
    # Allow tiny numerical precision tolerance (-1e-5)
    for i in range(1, len(ll_history)):
        diff = ll_history[i] - ll_history[i - 1]
        assert diff >= -1e-5, f"Log-likelihood decreased at iteration {i}: {diff}"


def test_symmetric_em_monotone(test_dataset):
    model = PLSA(n_topics=3, max_iter=25, tol=1e-8, beta=1.0, random_state=42, asymmetric=False)
    model.fit(test_dataset)

    ll_history = model.log_likelihood_history_
    assert len(ll_history) > 1

    for i in range(1, len(ll_history)):
        diff = ll_history[i] - ll_history[i - 1]
        assert diff >= -1e-5, f"Log-likelihood decreased at iteration {i}: {diff}"
