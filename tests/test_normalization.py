"""
Test that all probability distributions in PLSA sum to 1 and contain no NaNs/zeros.
"""

import numpy as np
import pytest
from src.plsa import PLSA


@pytest.fixture
def sample_count_matrix():
    # 20 documents, 50 words vocabulary
    rng = np.random.RandomState(42)
    counts = rng.poisson(lam=2.0, size=(20, 50)).astype(np.float64)
    # Ensure no empty documents
    counts[counts.sum(axis=1) == 0, 0] = 1.0
    return counts


def test_asymmetric_normalization(sample_count_matrix):
    model = PLSA(n_topics=4, max_iter=15, asymmetric=True, random_state=42)
    model.fit(sample_count_matrix)

    # Check P(w|z): sum over w = 1 for each topic z
    np.testing.assert_allclose(
        model.p_w_z.sum(axis=1),
        np.ones(model.n_topics),
        rtol=1e-5,
        err_msg="P(w|z) does not sum to 1 across vocabulary",
    )
    assert np.all(model.p_w_z > 0), "P(w|z) contains non-positive values"
    assert not np.isnan(model.p_w_z).any(), "P(w|z) contains NaNs"

    # Check P(z|d): sum over z = 1 for each document d
    np.testing.assert_allclose(
        model.p_z_d.sum(axis=1),
        np.ones(sample_count_matrix.shape[0]),
        rtol=1e-5,
        err_msg="P(z|d) does not sum to 1 across topics",
    )
    assert np.all(model.p_z_d > 0), "P(z|d) contains non-positive values"
    assert not np.isnan(model.p_z_d).any(), "P(z|d) contains NaNs"


def test_symmetric_normalization(sample_count_matrix):
    model = PLSA(n_topics=4, max_iter=15, asymmetric=False, random_state=42)
    model.fit(sample_count_matrix)

    # Check P(z): sum = 1
    assert abs(model.p_z.sum() - 1.0) < 1e-5, "P(z) does not sum to 1"
    assert not np.isnan(model.p_z).any()

    # Check P(d|z): sum over d = 1 for each topic z
    np.testing.assert_allclose(
        model.p_d_z.sum(axis=0),
        np.ones(model.n_topics),
        rtol=1e-5,
        err_msg="P(d|z) does not sum to 1 across documents",
    )

    # Check P(w|z): sum over w = 1 for each topic z
    np.testing.assert_allclose(
        model.p_w_z.sum(axis=1),
        np.ones(model.n_topics),
        rtol=1e-5,
        err_msg="P(w|z) does not sum to 1 across vocabulary",
    )


def test_folding_in_normalization(sample_count_matrix):
    model = PLSA(n_topics=4, max_iter=15, asymmetric=True, random_state=42)
    model.fit(sample_count_matrix)

    new_docs = sample_count_matrix[:5]
    p_z_new = model.transform(new_docs)

    np.testing.assert_allclose(
        p_z_new.sum(axis=1),
        np.ones(5),
        rtol=1e-5,
        err_msg="Folded-in P(z|d_new) does not sum to 1",
    )
    assert np.all(p_z_new > 0)
    assert not np.isnan(p_z_new).any()
