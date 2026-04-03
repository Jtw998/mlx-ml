import mlx.core as mx
from typing import Union, Callable
import numpy as np


class DistanceMetric:
    """
    Collection of vectorized distance metric functions optimized for MLX.
    All functions support broadcasting and GPU acceleration.
    """

    @staticmethod
    def euclidean(x1: mx.array, x2: mx.array) -> mx.array:
        """
        Euclidean distance: d(x,y) = sqrt(sum((x-y)^2))
        """
        return mx.sqrt(mx.sum((x1 - x2) ** 2, axis=-1))

    @staticmethod
    def manhattan(x1: mx.array, x2: mx.array) -> mx.array:
        """
        Manhattan (L1) distance: d(x,y) = sum(|x-y|)
        """
        return mx.sum(mx.abs(x1 - x2), axis=-1)

    @staticmethod
    def cosine(x1: mx.array, x2: mx.array) -> mx.array:
        """
        Cosine distance: d(x,y) = 1 - (x·y)/(||x||·||y||)
        """
        dot_product = mx.sum(x1 * x2, axis=-1)
        norm1 = mx.sqrt(mx.sum(x1 ** 2, axis=-1))
        norm2 = mx.sqrt(mx.sum(x2 ** 2, axis=-1))
        return 1 - (dot_product / (norm1 * norm2 + 1e-8))

    @staticmethod
    def chebyshev(x1: mx.array, x2: mx.array) -> mx.array:
        """
        Chebyshev (L∞) distance: d(x,y) = max(|x-y|)
        """
        return mx.max(mx.abs(x1 - x2), axis=-1)

    @staticmethod
    def minkowski(x1: mx.array, x2: mx.array, p: int = 2) -> mx.array:
        """
        Minkowski distance: d(x,y) = (sum(|x-y|^p))^(1/p)
        Special cases: p=1 -> Manhattan, p=2 -> Euclidean, p->∞ -> Chebyshev
        """
        return mx.power(mx.sum(mx.power(mx.abs(x1 - x2), p), axis=-1), 1/p)

    @staticmethod
    def hamming(x1: mx.array, x2: mx.array) -> mx.array:
        """
        Hamming distance: proportion of positions where elements differ
        """
        return mx.mean(x1 != x2, axis=-1)

    @staticmethod
    def jaccard(x1: mx.array, x2: mx.array) -> mx.array:
        """
        Jaccard distance: 1 - (|intersection| / |union|)
        For binary vectors only.
        """
        intersection = mx.sum(x1 * x2, axis=-1)
        union = mx.sum(mx.clip(x1 + x2, 0, 1), axis=-1)
        return 1 - (intersection / (union + 1e-8))

    @staticmethod
    def mahalanobis(x1: mx.array, x2: mx.array, inv_cov: mx.array) -> mx.array:
        """
        Mahalanobis distance: d(x,y) = sqrt((x-y)^T * inv_cov * (x-y))

        Parameters:
            inv_cov: Inverse of the covariance matrix of the dataset
        """
        diff = x1 - x2
        return mx.sqrt(mx.sum(mx.matmul(diff, inv_cov) * diff, axis=-1))

    @staticmethod
    def canberra(x1: mx.array, x2: mx.array) -> mx.array:
        """
        Canberra distance: sum(|x-y| / (|x| + |y|))
        """
        numerator = mx.abs(x1 - x2)
        denominator = mx.abs(x1) + mx.abs(x2)
        return mx.sum(numerator / (denominator + 1e-8), axis=-1)

    @staticmethod
    def braycurtis(x1: mx.array, x2: mx.array) -> mx.array:
        """
        Bray-Curtis distance: sum(|x-y|) / sum(|x+y|)
        """
        return mx.sum(mx.abs(x1 - x2), axis=-1) / (mx.sum(mx.abs(x1 + x2), axis=-1) + 1e-8)

    @staticmethod
    def correlation(x1: mx.array, x2: mx.array) -> mx.array:
        """
        Correlation distance: 1 - Pearson correlation coefficient between x and y
        """
        # Center the vectors
        x1_centered = x1 - mx.mean(x1, axis=-1, keepdims=True)
        x2_centered = x2 - mx.mean(x2, axis=-1, keepdims=True)

        # Compute correlation coefficient
        numerator = mx.sum(x1_centered * x2_centered, axis=-1)
        denominator = (mx.sqrt(mx.sum(x1_centered ** 2, axis=-1)) *
                      mx.sqrt(mx.sum(x2_centered ** 2, axis=-1)))

        return 1 - (numerator / (denominator + 1e-8))


# Metric mapping for string lookup
METRICS = {
    'euclidean': DistanceMetric.euclidean,
    'l2': DistanceMetric.euclidean,
    'manhattan': DistanceMetric.manhattan,
    'cityblock': DistanceMetric.manhattan,
    'l1': DistanceMetric.manhattan,
    'cosine': DistanceMetric.cosine,
    'chebyshev': DistanceMetric.chebyshev,
    'linf': DistanceMetric.chebyshev,
    'minkowski': DistanceMetric.minkowski,
    'hamming': DistanceMetric.hamming,
    'jaccard': DistanceMetric.jaccard,
    'mahalanobis': DistanceMetric.mahalanobis,
    'canberra': DistanceMetric.canberra,
    'braycurtis': DistanceMetric.braycurtis,
    'correlation': DistanceMetric.correlation
}


def get_metric(metric: Union[str, Callable]) -> Callable:
    """
    Get a distance metric function by name or return the custom function.

    Parameters:
        metric: Name of the metric or a custom callable

    Returns:
        Distance metric function
    """
    if isinstance(metric, str):
        if metric not in METRICS:
            raise ValueError(f"Unknown metric '{metric}'. Available metrics: {list(METRICS.keys())}")
        return METRICS[metric]
    elif callable(metric):
        return metric
    else:
        raise ValueError("Metric must be a string or a callable")
