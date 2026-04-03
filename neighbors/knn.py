import mlx.core as mx
import numpy as np
from typing import Optional, Union, Callable
from ..base.base_estimator import BaseEstimator
from ..spatial.distance import get_metric


class KNeighborsBase(BaseEstimator):
    """
    Base class for k-Nearest Neighbors implementations, contains core algorithm logic
    """

    def __init__(
        self,
        n_neighbors: int = 5,
        metric: Union[str, Callable] = 'euclidean',
        algorithm: str = 'brute',
        weights: str = 'uniform',
        metric_params: Optional[dict] = None
    ):
        """
        Initialize kNN model

        Parameters:
            n_neighbors: Number of neighbors to use
            metric: Distance metric to use ('euclidean', 'manhattan', 'cosine', 'chebyshev',
                   'minkowski', 'hamming', 'jaccard', 'mahalanobis', 'canberra', 'braycurtis',
                   'correlation' or custom function)
            algorithm: Search algorithm ('brute' for brute-force search, only supported currently)
            weights: Weighting strategy ('uniform' for equal weights, 'distance' for inverse distance weighting)
            metric_params: Additional parameters to pass to the metric function
        """
        super().__init__()
        self.n_neighbors = n_neighbors
        self.algorithm = algorithm
        self.weights = weights
        self.metric_params = metric_params or {}
        self._metric_name = metric if isinstance(metric, str) else None

        # Set distance metric
        self.metric = get_metric(metric)

    def set_params(self, **params: dict) -> "KNeighborsBase":
        """
        Set parameters, with special handling for metric parameter.
        """
        if 'metric' in params:
            metric = params.pop('metric')
            self._metric_name = metric if isinstance(metric, str) else None
            self.metric = get_metric(metric)
        return super().set_params(**params)

        self._fit_X = None
        self._fit_y = None

    def fit(self, X: Union[mx.array, np.array], y: Optional[Union[mx.array, np.array]] = None):
        """
        Fit the model (store training data)

        Parameters:
            X: Training feature matrix of shape (n_samples, n_features)
            y: Training labels of shape (n_samples,), optional for unsupervised learning
        """
        X, y = self._validate_data(X, y)
        self._fit_X = X
        if y is not None:
            self._fit_y = y
        return self

    def _kneighbors(self, X: mx.array, return_distance: bool = True):
        """
        Core k-nearest neighbor search algorithm

        Parameters:
            X: Query samples of shape (n_queries, n_features)
            return_distance: Whether to return distances along with indices

        Returns:
            distances: Distance matrix of shape (n_queries, n_neighbors), only returned if return_distance=True
            indices: Neighbor index matrix of shape (n_queries, n_neighbors)
        """
        if self._fit_X is None:
            raise ValueError("Model not fitted yet, call fit() first")

        n_queries = X.shape[0]
        n_train = self._fit_X.shape[0]

        # Expand dimensions for broadcasting
        X_expanded = X[:, None, :]  # (n_queries, 1, n_features)
        fit_X_expanded = self._fit_X[None, :, :]  # (1, n_train, n_features)

        # Compute distances from all query points to all training points
        distances = self.metric(X_expanded, fit_X_expanded, **self.metric_params)  # (n_queries, n_train)

        # Get indices of k nearest neighbors
        indices = mx.argpartition(distances, kth=self.n_neighbors, axis=1)[:, :self.n_neighbors]

        # Sort neighbors by distance
        batch_indices = mx.arange(n_queries)[:, None]
        neighbor_distances = distances[batch_indices, indices]
        sorted_order = mx.argsort(neighbor_distances, axis=1)
        sorted_indices = indices[batch_indices, sorted_order]

        if return_distance:
            sorted_distances = neighbor_distances[batch_indices, sorted_order]
            return sorted_distances, sorted_indices
        else:
            return sorted_indices

    def kneighbors(
        self,
        X: Union[mx.array, np.array],
        n_neighbors: Optional[int] = None,
        return_distance: bool = True
    ):
        """
        Find k-nearest neighbors for query samples

        Parameters:
            X: Query samples of shape (n_queries, n_features)
            n_neighbors: Number of neighbors to find, uses initialization value by default
            return_distance: Whether to return distances along with indices

        Returns:
            distances: Distance matrix of shape (n_queries, n_neighbors), only returned if return_distance=True
            indices: Neighbor index matrix of shape (n_queries, n_neighbors)
        """
        X, _ = self._validate_data(X)
        original_n_neighbors = self.n_neighbors

        if n_neighbors is not None:
            self.n_neighbors = n_neighbors

        result = self._kneighbors(X, return_distance=return_distance)

        self.n_neighbors = original_n_neighbors
        return result


class KNeighborsClassifier(KNeighborsBase):
    """
    k-Nearest Neighbors classifier
    """

    def predict(self, X: Union[mx.array, np.array]):
        """
        Predict class labels for query samples

        Parameters:
            X: Query samples of shape (n_queries, n_features)

        Returns:
            y_pred: Predicted class labels of shape (n_queries,)
        """
        X, _ = self._validate_data(X)
        distances, indices = self._kneighbors(X)

        # Get neighbor labels
        neighbor_labels = np.array(self._fit_y[indices])
        fit_y_np = np.array(self._fit_y)

        if self.weights == 'uniform':
            # Uniform weighting: majority vote
            predictions = []
            for i in range(neighbor_labels.shape[0]):
                labels = neighbor_labels[i]
                unique_labels, counts = np.unique(labels, return_counts=True)
                majority_idx = np.argmax(counts)
                predictions.append(unique_labels[majority_idx])
            return mx.array(np.array(predictions, dtype=np.int32))

        elif self.weights == 'distance':
            # Distance weighting: weighted vote
            weights = np.array(1.0 / (distances + 1e-8))  # Avoid division by zero
            predictions = []

            # Get all unique classes for classification
            unique_classes = np.unique(fit_y_np)

            for i in range(neighbor_labels.shape[0]):
                labels = neighbor_labels[i]
                sample_weights = weights[i]

                # Calculate weighted score for each class
                class_scores = np.zeros_like(unique_classes, dtype=np.float32)
                for j, cls in enumerate(unique_classes):
                    mask = (labels == cls)
                    class_scores[j] = np.sum(sample_weights * mask)

                # Select class with highest score
                best_class_idx = np.argmax(class_scores)
                predictions.append(unique_classes[best_class_idx])

            return mx.array(np.array(predictions, dtype=np.int32))

    def predict_proba(self, X: Union[mx.array, np.array]):
        """
        Predict class probabilities for query samples

        Parameters:
            X: Query samples of shape (n_queries, n_features)

        Returns:
            y_proba: Class probability matrix of shape (n_queries, n_classes)
        """
        X, _ = self._validate_data(X)
        distances, indices = self._kneighbors(X)

        # Get neighbor labels
        neighbor_labels = np.array(self._fit_y[indices])
        fit_y_np = np.array(self._fit_y)

        # Get all unique classes
        unique_classes = np.unique(fit_y_np)
        unique_classes = np.sort(unique_classes)
        n_classes = unique_classes.shape[0]
        n_queries = X.shape[0]

        # Initialize probability matrix
        probabilities = np.zeros((n_queries, n_classes), dtype=np.float32)

        if self.weights == 'uniform':
            # Uniform weighting
            for i in range(n_queries):
                labels = neighbor_labels[i]
                for j, cls in enumerate(unique_classes):
                    probabilities[i, j] = np.sum(labels == cls) / self.n_neighbors

        elif self.weights == 'distance':
            # Distance weighting
            weights = np.array(1.0 / (distances + 1e-8))
            for i in range(n_queries):
                labels = neighbor_labels[i]
                sample_weights = weights[i]
                total_weight = np.sum(sample_weights)

                for j, cls in enumerate(unique_classes):
                    mask = (labels == cls)
                    class_weight = np.sum(sample_weights * mask)
                    probabilities[i, j] = class_weight / total_weight

        return mx.array(probabilities)


class KNeighborsRegressor(KNeighborsBase):
    """
    k-Nearest Neighbors regressor
    """

    def predict(self, X: Union[mx.array, np.array]):
        """
        Predict regression values for query samples

        Parameters:
            X: Query samples of shape (n_queries, n_features)

        Returns:
            y_pred: Predicted values of shape (n_queries,)
        """
        X, _ = self._validate_data(X)
        distances, indices = self._kneighbors(X)

        # Get neighbor values
        neighbor_values = self._fit_y[indices]

        if self.weights == 'uniform':
            # Uniform weighting: average
            return mx.mean(neighbor_values, axis=1)

        elif self.weights == 'distance':
            # Distance weighting: weighted average
            weights = 1.0 / (distances + 1e-8)  # Avoid division by zero
            # Normalize weights
            weights_sum = mx.sum(weights, axis=1, keepdims=True)
            normalized_weights = weights / weights_sum
            # Weighted average
            return mx.sum(neighbor_values * normalized_weights, axis=1)


class NearestNeighbors(KNeighborsBase):
    """
    Unsupervised k-nearest neighbor search
    """

    def kneighbors_graph(self, X: Union[mx.array, np.array], mode: str = 'connectivity'):
        """
        Compute k-nearest neighbor graph

        Parameters:
            X: Query samples of shape (n_queries, n_features)
            mode: 'connectivity' for binary adjacency matrix, 'distance' for weighted distance matrix

        Returns:
            graph: Neighbor graph matrix of shape (n_queries, n_samples)
        """
        X, _ = self._validate_data(X)
        n_queries = X.shape[0]
        n_samples = self._fit_X.shape[0]

        distances, indices = self._kneighbors(X)

        # Initialize graph matrix
        graph = mx.zeros((n_queries, n_samples), dtype=mx.float32)

        # Populate neighbors
        batch_indices = mx.arange(n_queries)[:, None]
        if mode == 'connectivity':
            graph[batch_indices, indices] = 1.0
        elif mode == 'distance':
            graph[batch_indices, indices] = distances

        return graph
