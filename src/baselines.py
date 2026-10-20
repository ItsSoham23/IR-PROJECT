"""
Baseline topic models and classifiers:
1. LSI (Latent Semantic Indexing via TruncatedSVD + K-Means)
2. Keyword baseline (TF-IDF + K-Means)
"""

from typing import Optional, Tuple
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cluster import KMeans
from src.align import TopicAligner


class LSIBaseline:
    """
    Latent Semantic Indexing (Deerwester et al., 1990) baseline.
    Computes low-rank SVD approximation of document-term matrix,
    followed by K-Means clustering or nearest component assignment.
    """

    def __init__(self, n_components: int = 4, random_state: Optional[int] = None):
        self.n_components = n_components
        self.random_state = random_state
        self.svd = TruncatedSVD(n_components=n_components, random_state=random_state)
        self.kmeans = KMeans(n_clusters=n_components, random_state=random_state, n_init=10)
        self.tfidf = TfidfTransformer()
        self.aligner = TopicAligner(n_topics=n_components, n_classes=n_components)

    def fit(self, X: np.ndarray, y_train: Optional[np.ndarray] = None) -> "LSIBaseline":
        X_tfidf = self.tfidf.fit_transform(X)
        self.doc_embeddings_ = self.svd.fit_transform(X_tfidf)
        self.cluster_labels_ = self.kmeans.fit_predict(self.doc_embeddings_)

        if y_train is not None:
            self.aligner.fit(self.cluster_labels_, y_train)

        return self

    def transform(self, X_new: np.ndarray) -> np.ndarray:
        X_tfidf = self.tfidf.transform(X_new)
        return self.svd.transform(X_tfidf)

    def predict(self, X_new: np.ndarray) -> np.ndarray:
        embeddings = self.transform(X_new)
        cluster_preds = self.kmeans.predict(embeddings)
        if self.aligner.topic_to_class_:
            return self.aligner.predict(cluster_preds)
        return cluster_preds


class TFIDFBaseline:
    """
    Keyword TF-IDF + K-Means clustering baseline.
    """

    def __init__(self, n_clusters: int = 4, random_state: Optional[int] = None):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.tfidf = TfidfTransformer()
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        self.aligner = TopicAligner(n_topics=n_clusters, n_classes=n_clusters)

    def fit(self, X: np.ndarray, y_train: Optional[np.ndarray] = None) -> "TFIDFBaseline":
        X_tfidf = self.tfidf.fit_transform(X)
        self.cluster_labels_ = self.kmeans.fit_predict(X_tfidf)

        if y_train is not None:
            self.aligner.fit(self.cluster_labels_, y_train)

        return self

    def predict(self, X_new: np.ndarray) -> np.ndarray:
        X_tfidf = self.tfidf.transform(X_new)
        cluster_preds = self.kmeans.predict(X_tfidf)
        if self.aligner.topic_to_class_:
            return self.aligner.predict(cluster_preds)
        return cluster_preds
