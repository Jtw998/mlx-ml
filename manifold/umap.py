import mlx.core as mx
import numpy as np
from typing import Optional, Union, Tuple
from ..base.base_estimator import BaseEstimator
from ..neighbors import NearestNeighbors


class UMAP(BaseEstimator):
    """
    Uniform Manifold Approximation and Projection.

    Parameters:
        n_neighbors: The size of local neighborhood (in terms of number of neighboring sample points)
            used for manifold approximation.
        n_components: The dimension of the space to embed into.
        min_dist: The effective minimum distance between embedded points.
        spread: The effective scale of embedded points.
        metric: The metric to use to compute distances in high dimensional space.
        learning_rate: The initial learning rate for the optimization.
        n_epochs: The number of training epochs to be used in optimizing the low dimensional embedding.
        random_state: Controls the randomness of the estimator.
    """

    def __init__(
        self,
        n_neighbors: int = 15,
        n_components: int = 2,
        min_dist: float = 0.1,
        spread: float = 1.0,
        metric: str = 'euclidean',
        learning_rate: float = 1.0,
        n_epochs: int = 200,
        random_state: Optional[int] = None
    ):
        super().__init__()
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.min_dist = min_dist
        self.spread = spread
        self.metric = metric
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

        self.embedding_: Optional[np.array] = None

    def _compute_membership_strengths(self, distances: np.array, rho: np.array, sigma: np.array) -> np.array:
        """Compute membership strengths for each edge."""
        n_samples = distances.shape[0]
        weights = np.zeros_like(distances)

        for i in range(n_samples):
            for j in range(self.n_neighbors):
                d = distances[i, j]
                if d > rho[i]:
                    weights[i, j] = np.exp(-(d - rho[i]) / sigma[i])
                else:
                    weights[i, j] = 1.0

        return weights

    def _find_ab_params(self) -> Tuple[float, float]:
        """Find a and b parameters for the low dimensional kernel."""
        # We approximate a and b such that 1/(1 + a*x^(2*b)) matches the curve
        # defined by min_dist and spread.
        from scipy.optimize import curve_fit

        x = np.linspace(0, self.spread * 3, 300)

        def curve(x, a, b):
            return 1.0 / (1.0 + a * x ** (2 * b))

        y = np.zeros_like(x)
        y[x <= self.min_dist] = 1.0
        y[x > self.min_dist] = np.exp(-(x[x > self.min_dist] - self.min_dist) / self.spread)

        (a, b), _ = curve_fit(curve, x, y, p0=[1.0, 1.0])
        return a, b

    def fit_transform(self, X: Union[mx.array, np.array], y: Optional[Union[mx.array, np.array]] = None) -> mx.array:
        """
        Fit X into an embedded space and return that transformed output.

        Parameters:
            X: Training data of shape (n_samples, n_features)
            y: Not used, present for API consistency.

        Returns:
            Embedding of the training data in low-dimensional space, shape (n_samples, n_components)
        """
        X, _ = self._validate_data(X, y)
        X_np = np.array(X)
        n_samples = X_np.shape[0]

        # Step 1: Compute k-nearest neighbors
        print("Finding nearest neighbors...")
        nn = NearestNeighbors(n_neighbors=self.n_neighbors, metric=self.metric)
        nn.fit(X_np)
        distances, indices = nn.kneighbors(X_np)
        distances = np.array(distances)
        indices = np.array(indices)

        # Step 2: Compute rho (distance to nearest neighbor) and sigma
        print("Computing membership strengths...")
        rho = np.min(distances[:, 1:], axis=1)  # Skip distance to self
        sigma = np.zeros(n_samples)

        # Binary search for sigma to get log2(n_neighbors) sum of exp(-(d-rho)/sigma)
        target = np.log2(self.n_neighbors)
        for i in range(n_samples):
            low = 0.0
            high = np.max(distances[i])
            for _ in range(30):
                mid = (low + high) / 2
                ps = np.exp(-(distances[i, distances[i] > rho[i]] - rho[i]) / mid)
                sum_ps = np.sum(ps)
                if abs(sum_ps - target) < 1e-5:
                    break
                if sum_ps > target:
                    high = mid
                else:
                    low = mid
            sigma[i] = (low + high) / 2

        # Step 3: Compute weights and construct graph
        weights = self._compute_membership_strengths(distances, rho, sigma)

        # Make graph symmetric
        graph = np.zeros((n_samples, n_samples), dtype=np.float32)
        for i in range(n_samples):
            for j in range(self.n_neighbors):
                neighbor_idx = indices[i, j]
                graph[i, neighbor_idx] = max(graph[i, neighbor_idx], weights[i, j])
                graph[neighbor_idx, i] = max(graph[neighbor_idx, i], weights[i, j])

        # Get positive edges for sampling
        edges = np.array(np.where(graph > 0)).T
        n_edges = len(edges)
        print(f"Constructed graph with {n_edges} edges")

        # Step 4: Initialize embedding
        print("Initializing embedding...")
        embedding = self.rng.randn(n_samples, self.n_components).astype(np.float32) * 0.001

        # Step 5: Find a and b parameters
        a, b = self._find_ab_params()
        print(f"Fitted kernel parameters: a={a:.4f}, b={b:.4f}")

        # Step 6: Optimize embedding using numpy for easier indexing
        print(f"Optimizing embedding for {self.n_epochs} epochs...")
        embedding_np = embedding
        for epoch in range(self.n_epochs):
            # Shuffle edges
            edge_order = self.rng.permutation(n_edges)

            for i in range(n_edges):
                edge_idx = edge_order[i]
                i_idx, j_idx = edges[edge_idx]
                weight = graph[i_idx, j_idx]

                # Positive sample gradient
                diff = embedding_np[i_idx] - embedding_np[j_idx]
                dist_sq = np.sum(diff ** 2) + 1e-8
                grad_coeff = weight * 2 * a * b * (dist_sq ** (b - 1)) / (1 + a * (dist_sq ** b))
                grad = grad_coeff * diff

                embedding_np[i_idx] -= self.learning_rate * grad
                embedding_np[j_idx] += self.learning_rate * grad

                # Negative sample: sample random k != j
                k_idx = self.rng.randint(0, n_samples)
                while k_idx == i_idx or graph[i_idx, k_idx] > 0:
                    k_idx = self.rng.randint(0, n_samples)

                diff_neg = embedding_np[i_idx] - embedding_np[k_idx]
                dist_sq_neg = np.sum(diff_neg ** 2)
                grad_coeff_neg = 2 * b / ((0.001 + dist_sq_neg) * (1 + a * (dist_sq_neg ** b)))
                grad_neg = -grad_coeff_neg * diff_neg * 0.5  # Negative sample weight

                embedding_np[i_idx] -= self.learning_rate * grad_neg
                embedding_np[k_idx] += self.learning_rate * grad_neg

            # Print progress
            if (epoch + 1) % 50 == 0:
                print(f"Epoch {epoch+1}/{self.n_epochs} completed")

        self.embedding_ = embedding_np

        return mx.array(self.embedding_.astype(np.float32))

    def fit(self, X: Union[mx.array, np.array], y: Optional[Union[mx.array, np.array]] = None) -> "UMAP":
        """
        Fit X into an embedded space.

        Parameters:
            X: Training data of shape (n_samples, n_features)
            y: Not used, present for API consistency.

        Returns:
            Fitted estimator
        """
        self.fit_transform(X, y)
        return self
