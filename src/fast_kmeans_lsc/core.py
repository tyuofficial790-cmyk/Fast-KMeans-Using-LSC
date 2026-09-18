from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class FastKMeansLSC:
    """Fast K-means using a landmark-based approximation inspired by LSC."""

    n_clusters: int = 3
    n_landmarks: Optional[int] = None
    max_iter: int = 100
    tol: float = 1e-4
    random_state: Optional[int] = None

    def __post_init__(self) -> None:
        if self.n_clusters <= 0:
            raise ValueError("n_clusters must be positive")
        if self.max_iter <= 0:
            raise ValueError("max_iter must be positive")
        if self.tol < 0:
            raise ValueError("tol must be non-negative")

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "FastKMeansLSC":
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array-like object")
        n_samples, n_features = X.shape
        if n_samples == 0:
            raise ValueError("X must contain at least one sample")
        if self.n_clusters > n_samples:
            raise ValueError("n_clusters cannot exceed the number of samples")

        rng = np.random.default_rng(self.random_state)
        n_landmarks = self.n_landmarks or min(max(10, n_samples // 10), n_samples)
        n_landmarks = min(n_landmarks, n_samples)

        landmarks = self._select_landmarks(X, n_landmarks, rng)
        landmark_members = self._nearest_landmark(X, landmarks)
        landmark_data = landmarks

        centers = self._initialize_centers(landmark_data, self.n_clusters, rng)
        labels = np.zeros(n_samples, dtype=int)

        prev_inertia = np.inf
        for _ in range(self.max_iter):
            distances = self._pairwise_distances(X, centers)
            labels = distances.argmin(axis=1)

            new_centers = centers.copy()
            for idx in range(self.n_clusters):
                members = X[labels == idx]
                if len(members) == 0:
                    new_centers[idx] = X[rng.integers(0, n_samples)]
                else:
                    new_centers[idx] = members.mean(axis=0)

            center_shift = np.linalg.norm(new_centers - centers, axis=1).max()
            centers = new_centers

            inertia = self._compute_inertia(X, centers, labels)
            if center_shift <= self.tol or abs(prev_inertia - inertia) <= self.tol:
                break
            prev_inertia = inertia

        self.labels_ = labels
        self.cluster_centers_ = centers
        self.inertia_ = self._compute_inertia(X, centers, labels)
        self.landmarks_ = landmarks
        self.landmark_members_ = landmark_members
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if not hasattr(self, "cluster_centers_"):
            raise ValueError("The model has not been fitted yet.")
        return self._pairwise_distances(X, self.cluster_centers_).argmin(axis=1)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.labels_

    @staticmethod
    def _pairwise_distances(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
        diff = X[:, None, :] - centers[None, :, :]
        return np.linalg.norm(diff, axis=2)

    @staticmethod
    def _compute_inertia(X: np.ndarray, centers: np.ndarray, labels: np.ndarray) -> float:
        distances = FastKMeansLSC._pairwise_distances(X, centers)
        return float(np.sum(distances[np.arange(len(X)), labels] ** 2))

    @staticmethod
    def _select_landmarks(X: np.ndarray, n_landmarks: int, rng: np.random.Generator) -> np.ndarray:
        indices = np.empty(n_landmarks, dtype=int)
        indices[0] = rng.integers(0, X.shape[0])
        min_distances = np.sum((X - X[indices[0]]) ** 2, axis=1)

        for i in range(1, n_landmarks):
            candidate_scores = np.min(
                np.stack([
                    min_distances,
                    np.sum((X - X[rng.integers(0, X.shape[0])]) ** 2, axis=1),
                ], axis=1),
                axis=1,
            )
            next_index = int(np.argmax(candidate_scores))
            indices[i] = next_index
            min_distances = np.minimum(min_distances, np.sum((X - X[next_index]) ** 2, axis=1))

        return X[indices]

    @staticmethod
    def _nearest_landmark(X: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        distances = np.linalg.norm(X[:, None, :] - landmarks[None, :, :], axis=2)
        return distances.argmin(axis=1)

    @staticmethod
    def _initialize_centers(X: np.ndarray, n_clusters: int, rng: np.random.Generator) -> np.ndarray:
        if len(X) <= n_clusters:
            return X.copy()

        init_idx = [rng.integers(0, len(X))]
        min_distances = np.sum((X - X[init_idx[0]]) ** 2, axis=1)

        while len(init_idx) < n_clusters:
            distances = np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2)
            next_idx = int(np.argmax(np.min(distances[:, init_idx], axis=1)))
            init_idx.append(next_idx)
            min_distances = np.minimum(min_distances, np.sum((X - X[next_idx]) ** 2, axis=1))

        return X[np.array(init_idx)]


def lsc_kmeans(X: np.ndarray, n_clusters: int = 3, **kwargs) -> FastKMeansLSC:
    model = FastKMeansLSC(n_clusters=n_clusters, **kwargs)
    return model.fit(X)
