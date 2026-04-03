import mlx.core as mx
import numpy as np
from typing import Optional, Union, Tuple
from ..base.base_estimator import BaseEstimator
from ..spatial.distance import DistanceMetric


class KMeans(BaseEstimator):
    """
    K-Means clustering.

    Parameters:
        n_clusters: Number of clusters to form
        init: Method for initialization: 'k-means++' or 'random'
        n_init: Number of times the algorithm will be run with different centroid seeds
        max_iter: Maximum number of iterations of the k-means algorithm for a single run
        tol: Relative tolerance with regards to inertia to declare convergence
        random_state: Random seed for centroid initialization
        metric: Distance metric to use, default 'euclidean'
    """

    def __init__(
        self,
        n_clusters: int = 8,
        init: str = 'k-means++',
        n_init: int = 10,
        max_iter: int = 300,
        tol: float = 1e-4,
        random_state: Optional[int] = None,
        metric: str = 'euclidean'
    ):
        super().__init__()
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.metric = metric

        self.cluster_centers_: Optional[mx.array] = None
        self.labels_: Optional[mx.array] = None
        self.inertia_: Optional[float] = None
        self.n_iter_: Optional[int] = None
        self.n_features_in_: Optional[int] = None

        valid_init_methods = ['k-means++', 'random']
        if init not in valid_init_methods:
            raise ValueError(f"Invalid init method '{init}'. Valid options: {valid_init_methods}")

        self.distance_fn = getattr(DistanceMetric, metric)

    def _init_centroids(self, X: mx.array, rng: np.random.RandomState) -> mx.array:
        """
        Initialize cluster centroids.
        """
        n_samples, n_features = X.shape

        if self.init == 'random':
            # Randomly select n_clusters samples
            indices = rng.choice(n_samples, self.n_clusters, replace=False)
            indices = mx.array(indices, dtype=mx.int32)
            centroids = X[indices]
        elif self.init == 'k-means++':
            # K-Means++ initialization
            centroids = []
            # Select first centroid randomly
            first_idx = rng.randint(n_samples)
            centroids.append(X[first_idx])

            for _ in range(1, self.n_clusters):
                # Compute distances to nearest existing centroid
                distances = []
                for x in X:
                    min_dist = float('inf')
                    for c in centroids:
                        dist = self.distance_fn(x, c).item()
                        if dist < min_dist:
                            min_dist = dist
                    distances.append(min_dist ** 2)

                # Select next centroid with probability proportional to distance squared
                distances = np.array(distances)
                probabilities = distances / distances.sum()
                next_idx = rng.choice(n_samples, p=probabilities)
                centroids.append(X[next_idx])

            centroids = mx.stack(centroids)

        return centroids

    def _fit_single(self, X: mx.array, rng: np.random.RandomState) -> Tuple[mx.array, mx.array, float, int]:
        """
        Run K-Means once with a single initialization.
        """
        n_samples = X.shape[0]
        centroids = self._init_centroids(X, rng)
        prev_inertia = float('inf')

        for iter_idx in range(self.max_iter):
            # Assign samples to nearest centroid
            distances = []
            for c in centroids:
                dists = self.distance_fn(X, c)
                distances.append(dists)
            distances = mx.stack(distances, axis=1)  # Shape (n_samples, n_clusters)
            labels = mx.argmin(distances, axis=1)

            # Update centroids
            new_centroids = []
            inertia = 0.0
            for k in range(self.n_clusters):
                mask_np = np.array(labels == k)
                if np.sum(mask_np) == 0:
                    # Empty cluster, reinitialize randomly
                    new_centroid = X[rng.randint(n_samples)]
                else:
                    cluster_samples = X[mx.array(np.where(mask_np)[0], dtype=mx.int32)]
                    new_centroid = mx.mean(cluster_samples, axis=0)
                    # Add inertia for this cluster
                    inertia += mx.sum(self.distance_fn(cluster_samples, new_centroid)).item()
                new_centroids.append(new_centroid)

            new_centroids = mx.stack(new_centroids)

            # Check convergence
            if abs(prev_inertia - inertia) < self.tol:
                break

            prev_inertia = inertia
            centroids = new_centroids

        # Final assignment
        distances = []
        for c in centroids:
            dists = self.distance_fn(X, c)
            distances.append(dists)
        distances = mx.stack(distances, axis=1)
        labels = mx.argmin(distances, axis=1)
        inertia = 0.0
        for k in range(self.n_clusters):
            mask_np = np.array(labels == k)
            if np.sum(mask_np) > 0:
                cluster_samples = X[mx.array(np.where(mask_np)[0], dtype=mx.int32)]
                inertia += mx.sum(self.distance_fn(cluster_samples, centroids[k])).item()

        return centroids, labels, inertia, iter_idx + 1

    def fit(self, X: Union[mx.array, np.array], y: Optional[Union[mx.array, np.array]] = None) -> "KMeans":
        """
        Compute k-means clustering.

        Parameters:
            X: Training data of shape (n_samples, n_features)
            y: Ignored, present for API compatibility

        Returns:
            Fitted estimator
        """
        X, _ = self._validate_data(X)
        self.n_features_in_ = X.shape[1]
        n_samples = X.shape[0]

        if self.n_clusters > n_samples:
            raise ValueError(f"n_clusters={self.n_clusters} cannot be larger than number of samples={n_samples}")

        rng = np.random.RandomState(self.random_state)

        best_inertia = float('inf')
        best_centroids = None
        best_labels = None
        best_iter = None

        # Run n_init times and select the best result
        for _ in range(self.n_init):
            centroids, labels, inertia, n_iter = self._fit_single(X, rng)
            if inertia < best_inertia:
                best_inertia = inertia
                best_centroids = centroids
                best_labels = labels
                best_iter = n_iter

        self.cluster_centers_ = best_centroids
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        self.n_iter_ = best_iter

        return self

    def predict(self, X: Union[mx.array, np.array]) -> mx.array:
        """
        Predict the closest cluster each sample in X belongs to.

        Parameters:
            X: Samples of shape (n_samples, n_features)

        Returns:
            Index of the cluster each sample belongs to
        """
        if self.cluster_centers_ is None:
            raise ValueError("Model not fitted yet. Call 'fit' first.")

        X, _ = self._validate_data(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {X.shape[1]}")

        # Compute distances to centroids
        distances = []
        for c in self.cluster_centers_:
            dists = self.distance_fn(X, c)
            distances.append(dists)
        distances = mx.stack(distances, axis=1)

        return mx.argmin(distances, axis=1)

    def fit_predict(self, X: Union[mx.array, np.array], y: Optional[Union[mx.array, np.array]] = None) -> mx.array:
        """
        Compute cluster centers and predict cluster index for each sample.

        Parameters:
            X: Training data of shape (n_samples, n_features)
            y: Ignored, present for API compatibility

        Returns:
            Index of the cluster each sample belongs to
        """
        return self.fit(X).predict(X)

    def transform(self, X: Union[mx.array, np.array]) -> mx.array:
        """
        Transform X to a cluster-distance space.

        Parameters:
            X: Samples of shape (n_samples, n_features)

        Returns:
            Distances to each cluster center of shape (n_samples, n_clusters)
        """
        if self.cluster_centers_ is None:
            raise ValueError("Model not fitted yet. Call 'fit' first.")

        X, _ = self._validate_data(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {X.shape[1]}")

        # Compute distances to all centroids
        distances = []
        for c in self.cluster_centers_:
            dists = self.distance_fn(X, c)
            distances.append(dists)
        return mx.stack(distances, axis=1)
