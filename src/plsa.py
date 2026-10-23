"""
Probabilistic Latent Semantic Analysis (PLSA / PLSI) implementation from scratch.

References:
- Hofmann, T. (1999). Probabilistic Latent Semantic Analysis. UAI 1999.
- Hofmann, T. (1999). Probabilistic Latent Semantic Indexing. SIGIR 1999.
- Hofmann, T. (2001). Unsupervised Learning by Probabilistic Latent Semantic Analysis.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import scipy.sparse as sp


class PLSA:
    """
    Probabilistic Latent Semantic Analysis (PLSA) model trained via Expectation-Maximization (EM).

    Supports:
    - Symmetric aspect model: P(d, w) = sum_z P(z) P(d|z) P(w|z)
    - Asymmetric model: P(w|d) = sum_z P(z|d) P(w|z)
    - Tempered EM (beta <= 1.0)
    - Folding-in inference for held-out documents
    - Multiple random restarts
    - Perplexity and Log-Likelihood tracking
    """

    def __init__(
        self,
        n_topics: int = 4,
        max_iter: int = 1000,
        tol: float = 1e-5,
        beta: float = 1.0,
        smoothing: float = 1e-12,
        n_restarts: int = 1,
        random_state: Optional[int] = None,
        asymmetric: bool = True,
    ):
        """
        Args:
            n_topics: Number of latent topics (K).
            max_iter: Maximum number of EM iterations.
            tol: Convergence tolerance for relative change in log-likelihood.
            beta: Inverse temperature for Tempered EM (1.0 = standard EM).
            smoothing: Small constant added to avoid zero division or log(0).
            n_restarts: Number of random restarts (keeps model with highest LL).
            random_state: Seed for random number generation.
            asymmetric: If True, uses P(w|d) = sum_z P(z|d) P(w|z); else symmetric.
        """
        self.n_topics = n_topics
        self.max_iter = max_iter
        self.tol = tol
        self.beta = beta
        self.smoothing = smoothing
        self.n_restarts = n_restarts
        self.random_state = random_state
        self.asymmetric = asymmetric

        # Model parameters
        # Asymmetric:
        # P(w|z): shape (K, V) -> probability of word w given topic z
        # P(z|d): shape (D, K) -> probability of topic z given document d
        # Symmetric:
        # P(z): shape (K,)
        # P(d|z): shape (D, K)
        # P(w|z): shape (K, V)
        self.p_w_z: Optional[np.ndarray] = None
        self.p_z_d: Optional[np.ndarray] = None
        self.p_z: Optional[np.ndarray] = None
        self.p_d_z: Optional[np.ndarray] = None

        self.log_likelihood_history_: List[float] = []
        self.converged_: bool = False
        self.n_iter_: int = 0
        self.final_log_likelihood_: float = -np.inf

    def _init_params(
        self, D: int, V: int, rng: np.random.RandomState
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """Initialize parameters with Dirichlet/Uniform random values normalized properly."""
        if self.asymmetric:
            # P(z|d): shape (D, K), sum over z = 1 for each d
            p_z_d = rng.uniform(0.1, 1.0, size=(D, self.n_topics))
            p_z_d /= p_z_d.sum(axis=1, keepdims=True)

            # P(w|z): shape (K, V), sum over w = 1 for each z
            p_w_z = rng.uniform(0.1, 1.0, size=(self.n_topics, V))
            p_w_z /= p_w_z.sum(axis=1, keepdims=True)

            return p_w_z, p_z_d, None, None
        else:
            # Symmetric
            # P(z): shape (K,), sum = 1
            p_z = rng.uniform(0.1, 1.0, size=self.n_topics)
            p_z /= p_z.sum()

            # P(d|z): shape (D, K), sum over d = 1 for each z
            p_d_z = rng.uniform(0.1, 1.0, size=(D, self.n_topics))
            p_d_z /= p_d_z.sum(axis=0, keepdims=True)

            # P(w|z): shape (K, V), sum over w = 1 for each z
            p_w_z = rng.uniform(0.1, 1.0, size=(self.n_topics, V))
            p_w_z /= p_w_z.sum(axis=1, keepdims=True)

            return p_w_z, None, p_z, p_d_z

    def _compute_log_likelihood(
        self,
        X: np.ndarray,
        p_w_z: np.ndarray,
        p_z_d: Optional[np.ndarray] = None,
        p_z: Optional[np.ndarray] = None,
        p_d_z: Optional[np.ndarray] = None,
    ) -> float:
        """
        Compute total log-likelihood: sum_{d,w} n(d,w) * log P(d,w) or sum_{d,w} n(d,w) * log P(w|d).
        """
        if self.asymmetric:
            # P(w|d) = sum_z P(z|d) * P(w|z) -> shape (D, V)
            # Use dot product: (D, K) @ (K, V) -> (D, V)
            p_w_given_d = np.dot(p_z_d, p_w_z)
            p_w_given_d = np.maximum(p_w_given_d, self.smoothing)
            # LL = sum_{d,w} X[d,w] * log P(w|d)
            if sp.issparse(X):
                rows, cols = X.nonzero()
                data = X.data
                log_prob = np.log(p_w_given_d[rows, cols])
                return float(np.sum(data * log_prob))
            else:
                return float(np.sum(X * np.log(p_w_given_d)))
        else:
            # Symmetric: P(d, w) = sum_z P(z) * P(d|z) * P(w|z)
            # P(d, w) = (P(d|z) * P(z)) @ P(w|z)
            weighted_p_d_z = p_d_z * p_z[np.newaxis, :]  # (D, K)
            p_dw = np.dot(weighted_p_d_z, p_w_z)  # (D, V)
            p_dw = np.maximum(p_dw, self.smoothing)
            if sp.issparse(X):
                rows, cols = X.nonzero()
                data = X.data
                log_prob = np.log(p_dw[rows, cols])
                return float(np.sum(data * log_prob))
            else:
                return float(np.sum(X * np.log(p_dw)))

    def _fit_single_run(
        self, X: np.ndarray, rng: np.random.RandomState
    ) -> Tuple[
        np.ndarray,
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        List[float],
        bool,
        int,
    ]:
        """Perform one EM run from random initialization to convergence."""
        D, V = X.shape
        p_w_z, p_z_d, p_z, p_d_z = self._init_params(D, V, rng)
        ll_history = []
        converged = False

        is_sparse = sp.issparse(X)
        if is_sparse:
            # For sparse X, extract non-zero indices for efficiency
            X_csr = X.tocsr()
            rows, cols = X_csr.nonzero()
            counts = np.array(X_csr.data, dtype=np.float64)
        else:
            X_dense = np.asarray(X, dtype=np.float64)

        prev_ll = -np.inf
        n_iter = 0

        for it in range(self.max_iter):
            n_iter = it + 1

            if self.asymmetric:
                # E-step:
                # P(z|d, w) = [P(z|d) * P(w|z)]^beta / sum_z' [P(z'|d) * P(w|z')]^beta
                if is_sparse:
                    # For each non-zero (d, w):
                    # unnorm[idx, z] = (p_z_d[d, z] * p_w_z[z, w]) ** beta
                    # Vectorized across nonzeros:
                    pz_d_nz = p_z_d[rows, :]  # (N_nz, K)
                    pw_z_nz = p_w_z[:, cols].T  # (N_nz, K)
                    p_z_dw = pz_d_nz * pw_z_nz  # (N_nz, K)
                    if self.beta != 1.0:
                        p_z_dw = np.power(np.maximum(p_z_dw, self.smoothing), self.beta)
                    p_z_dw /= np.maximum(p_z_dw.sum(axis=1, keepdims=True), self.smoothing)

                    # M-step:
                    # p_w_z[z, w] proportional to sum_d X[d, w] * P(z|d, w)
                    # p_z_d[d, z] proportional to sum_w X[d, w] * P(z|d, w)
                    weighted_counts = counts[:, np.newaxis] * p_z_dw  # (N_nz, K)

                    # Accumulate for p_w_z: (K, V)
                    new_p_w_z = np.zeros((self.n_topics, V), dtype=np.float64)
                    for k in range(self.n_topics):
                        np.add.at(new_p_w_z[k], cols, weighted_counts[:, k])
                    new_p_w_z += self.smoothing
                    new_p_w_z /= new_p_w_z.sum(axis=1, keepdims=True)

                    # Accumulate for p_z_d: (D, K)
                    new_p_z_d = np.zeros((D, self.n_topics), dtype=np.float64)
                    for k in range(self.n_topics):
                        np.add.at(new_p_z_d[:, k], rows, weighted_counts[:, k])
                    new_p_z_d += self.smoothing
                    new_p_z_d /= new_p_z_d.sum(axis=1, keepdims=True)

                else:
                    # Fully vectorized dense update
                    # p_z_d is (D, K), p_w_z is (K, V)
                    # p_z_dw: (D, V, K)
                    p_z_dw = p_z_d[:, np.newaxis, :] * p_w_z.T[np.newaxis, :, :]  # (D, V, K)
                    if self.beta != 1.0:
                        p_z_dw = np.power(np.maximum(p_z_dw, self.smoothing), self.beta)
                    norm = p_z_dw.sum(axis=2, keepdims=True)
                    p_z_dw /= np.maximum(norm, self.smoothing)

                    # M-step:
                    # weighted_X = X[:, :, np.newaxis] * p_z_dw  (D, V, K)
                    weighted_X = X_dense[:, :, np.newaxis] * p_z_dw

                    # new_p_w_z: sum over d -> (V, K) -> transpose to (K, V)
                    new_p_w_z = np.sum(weighted_X, axis=0).T  # (K, V)
                    new_p_w_z += self.smoothing
                    new_p_w_z /= new_p_w_z.sum(axis=1, keepdims=True)

                    # new_p_z_d: sum over w -> (D, K)
                    new_p_z_d = np.sum(weighted_X, axis=1)  # (D, K)
                    new_p_z_d += self.smoothing
                    new_p_z_d /= new_p_z_d.sum(axis=1, keepdims=True)

                p_w_z = new_p_w_z
                p_z_d = new_p_z_d

            else:
                # Symmetric formulation:
                # E-step: P(z|d, w) proportional to P(z) * [P(d|z) * P(w|z)]^beta
                if is_sparse:
                    pd_z_nz = p_d_z[rows, :]  # (N_nz, K)
                    pw_z_nz = p_w_z[:, cols].T  # (N_nz, K)
                    base = pd_z_nz * pw_z_nz
                    if self.beta != 1.0:
                        base = np.power(np.maximum(base, self.smoothing), self.beta)
                    p_z_dw = p_z[np.newaxis, :] * base
                    p_z_dw /= np.maximum(p_z_dw.sum(axis=1, keepdims=True), self.smoothing)

                    weighted_counts = counts[:, np.newaxis] * p_z_dw  # (N_nz, K)

                    new_p_w_z = np.zeros((self.n_topics, V), dtype=np.float64)
                    for k in range(self.n_topics):
                        np.add.at(new_p_w_z[k], cols, weighted_counts[:, k])
                    new_p_w_z += self.smoothing
                    new_p_w_z /= new_p_w_z.sum(axis=1, keepdims=True)

                    new_p_d_z = np.zeros((D, self.n_topics), dtype=np.float64)
                    for k in range(self.n_topics):
                        np.add.at(new_p_d_z[:, k], rows, weighted_counts[:, k])
                    new_p_d_z += self.smoothing
                    new_p_d_z /= new_p_d_z.sum(axis=0, keepdims=True)

                    new_p_z = weighted_counts.sum(axis=0) + self.smoothing
                    new_p_z /= new_p_z.sum()

                else:
                    p_z_dw = (
                        p_z[np.newaxis, np.newaxis, :]
                        * (p_d_z[:, np.newaxis, :] * p_w_z.T[np.newaxis, :, :]) ** self.beta
                    )
                    norm = p_z_dw.sum(axis=2, keepdims=True)
                    p_z_dw /= np.maximum(norm, self.smoothing)

                    weighted_X = X_dense[:, :, np.newaxis] * p_z_dw

                    new_p_w_z = np.sum(weighted_X, axis=0).T  # (K, V)
                    new_p_w_z += self.smoothing
                    new_p_w_z /= new_p_w_z.sum(axis=1, keepdims=True)

                    new_p_d_z = np.sum(weighted_X, axis=1)  # (D, K)
                    new_p_d_z += self.smoothing
                    new_p_d_z /= new_p_d_z.sum(axis=0, keepdims=True)

                    new_p_z = np.sum(weighted_X, axis=(0, 1))  # (K,)
                    new_p_z += self.smoothing
                    new_p_z /= new_p_z.sum()

                p_w_z = new_p_w_z
                p_d_z = new_p_d_z
                p_z = new_p_z

            # Check log-likelihood
            ll = self._compute_log_likelihood(X, p_w_z, p_z_d, p_z, p_d_z)
            ll_history.append(ll)

            if it > 0:
                rel_change = abs(ll - prev_ll) / (abs(prev_ll) + self.smoothing)
                if rel_change < self.tol:
                    converged = True
                    break
            prev_ll = ll

        return p_w_z, p_z_d, p_z, p_d_z, ll_history, converged, n_iter

    def fit(self, X: Union[np.ndarray, sp.spmatrix]) -> "PLSA":
        """
        Fit PLSA model on count matrix X (D x V).
        If n_restarts > 1, fits multiple models with different seeds and keeps the best LL.
        """
        base_rng = np.random.RandomState(self.random_state)
        best_ll = -np.inf
        best_run = None

        for restart_idx in range(self.n_restarts):
            run_seed = base_rng.randint(0, 2**31 - 1)
            rng = np.random.RandomState(run_seed)
            run_result = self._fit_single_run(X, rng)
            final_ll = run_result[4][-1] if run_result[4] else -np.inf

            if final_ll > best_ll:
                best_ll = final_ll
                best_run = run_result

        assert best_run is not None, "Fitting failed to produce any results"
        (
            self.p_w_z,
            self.p_z_d,
            self.p_z,
            self.p_d_z,
            self.log_likelihood_history_,
            self.converged_,
            self.n_iter_,
        ) = best_run
        self.final_log_likelihood_ = best_ll

        return self

    def transform(
        self,
        X_new: Union[np.ndarray, sp.spmatrix],
        max_iter: int = 50,
        tol: float = 1e-5,
        random_state: Optional[int] = None,
    ) -> np.ndarray:
        """
        Folding-in algorithm for unseen documents (Hofmann, 1999).
        Freezes P(w|z) and estimates P(z|d_new).

        Args:
            X_new: Count matrix of unseen documents (D_new x V).
            max_iter: Max folding-in EM iterations.
            tol: Tolerance on P(z|d) update change.
            random_state: Random seed for initializing P(z|d).

        Returns:
            P(z|d_new): Matrix of shape (D_new, K) representing topic proportions.
        """
        if self.p_w_z is None:
            raise ValueError("Model must be fitted before transforming new documents.")

        D_new, V = X_new.shape
        if V != self.p_w_z.shape[1]:
            raise ValueError(
                f"Vocabulary size mismatch: expected {self.p_w_z.shape[1]}, got {V}"
            )

        rng = np.random.RandomState(
            random_state if random_state is not None else self.random_state
        )
        # Initialize P(z|d_new)
        p_z_dnew = rng.uniform(0.1, 1.0, size=(D_new, self.n_topics))
        p_z_dnew /= p_z_dnew.sum(axis=1, keepdims=True)

        is_sparse = sp.issparse(X_new)
        if is_sparse:
            X_csr = X_new.tocsr()
            rows, cols = X_csr.nonzero()
            counts = np.array(X_csr.data, dtype=np.float64)
        else:
            X_dense = np.asarray(X_new, dtype=np.float64)

        for _ in range(max_iter):
            p_z_dnew_prev = p_z_dnew.copy()

            if is_sparse:
                # E-step
                pz_d_nz = p_z_dnew[rows, :]  # (N_nz, K)
                pw_z_nz = self.p_w_z[:, cols].T  # (N_nz, K)
                p_z_dw = pz_d_nz * pw_z_nz
                p_z_dw /= np.maximum(p_z_dw.sum(axis=1, keepdims=True), self.smoothing)

                # M-step: update p_z_dnew only
                weighted_counts = counts[:, np.newaxis] * p_z_dw  # (N_nz, K)
                new_p_z_d = np.zeros((D_new, self.n_topics), dtype=np.float64)
                for k in range(self.n_topics):
                    np.add.at(new_p_z_d[:, k], rows, weighted_counts[:, k])
                new_p_z_d += self.smoothing
                new_p_z_d /= new_p_z_d.sum(axis=1, keepdims=True)
                p_z_dnew = new_p_z_d

            else:
                # E-step
                p_z_dw = p_z_dnew[:, np.newaxis, :] * self.p_w_z.T[np.newaxis, :, :]  # (D, V, K)
                norm = p_z_dw.sum(axis=2, keepdims=True)
                p_z_dw /= np.maximum(norm, self.smoothing)

                # M-step
                weighted_X = X_dense[:, :, np.newaxis] * p_z_dw
                new_p_z_d = np.sum(weighted_X, axis=1)  # (D, K)
                new_p_z_d += self.smoothing
                new_p_z_d /= new_p_z_d.sum(axis=1, keepdims=True)
                p_z_dnew = new_p_z_d

            if np.max(np.abs(p_z_dnew - p_z_dnew_prev)) < tol:
                break

        return p_z_dnew

    def perplexity(
        self,
        X: Union[np.ndarray, sp.spmatrix],
        p_z_d: Optional[np.ndarray] = None,
    ) -> float:
        """
        Compute perplexity on document count matrix X:
        Perplexity = exp( - sum_{d,w} n(d,w) * log P(w|d) / sum_{d,w} n(d,w) )
        """
        if self.p_w_z is None:
            raise ValueError("Model must be fitted before computing perplexity.")

        if p_z_d is None:
            if self.asymmetric and self.p_z_d is not None and self.p_z_d.shape[0] == X.shape[0]:
                p_z_d = self.p_z_d
            else:
                p_z_d = self.transform(X)

        # P(w|d) = sum_z P(z|d) * P(w|z) -> (D, V)
        p_w_given_d = np.dot(p_z_d, self.p_w_z)
        p_w_given_d = np.maximum(p_w_given_d, self.smoothing)

        if sp.issparse(X):
            rows, cols = X.nonzero()
            counts = np.array(X.data, dtype=np.float64)
            total_words = counts.sum()
            if total_words == 0:
                return 0.0
            log_prob = np.log(p_w_given_d[rows, cols])
            cross_entropy = -np.sum(counts * log_prob) / total_words
        else:
            total_words = np.sum(X)
            if total_words == 0:
                return 0.0
            cross_entropy = -np.sum(X * np.log(p_w_given_d)) / total_words

        return float(np.exp(cross_entropy))

    def get_top_words(
        self,
        vocab: List[str],
        n_top_words: int = 10,
    ) -> Dict[int, List[Tuple[str, float]]]:
        """
        Return the top N words and their probabilities P(w|z) for each topic.
        """
        if self.p_w_z is None:
            raise ValueError("Model must be fitted to get top words.")

        top_words = {}
        for topic_idx in range(self.n_topics):
            probs = self.p_w_z[topic_idx]
            top_indices = np.argsort(probs)[::-1][:n_top_words]
            top_words[topic_idx] = [(vocab[i], float(probs[i])) for i in top_indices]

        return top_words
