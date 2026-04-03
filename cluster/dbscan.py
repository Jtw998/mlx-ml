import mlx.core as mx
import numpy as np
from typing import Optional, Union, List
from ..base.base_estimator import BaseEstimator
from ..spatial.distance import get_metric


class DBSCAN(BaseEstimator):
    """
    Density-Based Spatial Clustering of Applications with Noise.

    Parameters:
        eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
        min_samples: The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.
        metric: The metric to use when calculating distance between instances in a feature array.
        n_jobs: The number of parallel jobs to run for neighbors search. Currently not used.
    """

    def __init__(
        self,
        eps: float = 0.5,
        min_samples: int = 5,
        metric: str = 'euclidean',
        n_jobs: Optional[int] = None
    ):
        super().__init__()
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.n_jobs = n_jobs

        self.core_sample_indices_: Optional[np.array] = None
        self.components_: Optional[np.array] = None
        self.labels_: Optional[np.array] = None

    def fit(self, X: Union[mx.array, np.array], y: Optional[Union[mx.array, np.array]] = None) -> "DBSCAN":
        """
        Perform DBSCAN clustering from features or distance matrix.

        Parameters:
            X: Training data of shape (n_samples, n_features)
            y: Not used, present for API consistency.

        Returns:
            Fitted estimator
        """
        X, _ = self._validate_data(X, y)
        X_np = np.array(X)
        n_samples = X_np.shape[0]

        # Compute pairwise distances
        metric_fn = get_metric(self.metric)
        X_mx = mx.array(X_np)
        dist_matrix = np.zeros((n_samples, n_samples), dtype=np.float32)

        for i in range(n_samples):
            # Compute distance from X[i] to all points
            dist_matrix[i] = np.array(metric_fn(X_mx[i:i+1], X_mx))

        # Find core points
        neighbors = [np.where(dist_matrix[i] <= self.eps)[0] for i in range(n_samples)]
        is_core = np.array([len(neighbors[i]) >= self.min_samples for i in range(n_samples)])
        core_indices = np.where(is_core)[0]

        # Initialize labels: -1 means noise/unvisited
        labels = np.full(n_samples, -1, dtype=np.int32)
        current_cluster = 0

        # Process each core point
        for i in core_indices:
            if labels[i] == -1:
                # Start a new cluster
                stack = [i]
                labels[i] = current_cluster

                while stack:
                    point_idx = stack.pop()
                    if is_core[point_idx]:
                        # Add all reachable points
                        for neighbor_idx in neighbors[point_idx]:
                            if labels[neighbor_idx] == -1:
                                labels[neighbor_idx] = current_cluster
                                stack.append(neighbor_idx)

                current_cluster += 1

        self.labels_ = labels
        self.core_sample_indices_ = core_indices
        self.components_ = X_np[core_indices]

        return self

    def fit_predict(self, X: Union[mx.array, np.array], y: Optional[Union[mx.array, np.array]] = None) -> mx.array:
        """
        Perform DBSCAN clustering from features or distance matrix and return cluster labels.

        Parameters:
            X: Training data of shape (n_samples, n_features)
            y: Not used, present for API consistency.

        Returns:
            Cluster labels of shape (n_samples,). Noisy samples are given the label -1.
        """
        self.fit(X, y)
        return mx.array(self.labels_)
