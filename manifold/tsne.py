import mlx.core as mx
import numpy as np
from typing import Optional, Union
from ..base.base_estimator import BaseEstimator
from ..spatial.distance import get_metric


class TSNE(BaseEstimator):
    """
    t-distributed Stochastic Neighbor Embedding.

    Parameters:
        n_components: Dimension of the embedded space.
        perplexity: The perplexity is related to the number of nearest neighbors that is used
            in other manifold learning algorithms. Larger datasets usually require a larger perplexity.
        early_exaggeration: Controls how tight natural clusters in the original space are in
            the embedded space and how much space will be between them.
        learning_rate: The learning rate for t-SNE, usually in the range [10.0, 1000.0].
        max_iter: Maximum number of iterations for the optimization.
        metric: The metric to use when calculating distance between instances in a feature array.
        random_state: Controls the randomness of the estimator.
    """

    def __init__(
        self,
        n_components: int = 2,
        perplexity: float = 30.0,
        early_exaggeration: float = 12.0,
        learning_rate: float = 200.0,
        max_iter: int = 1000,
        metric: str = 'euclidean',
        random_state: Optional[int] = None
    ):
        super().__init__()
        self.n_components = n_components
        self.perplexity = perplexity
        self.early_exaggeration = early_exaggeration
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.metric = metric
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

        self.embedding_: Optional[np.array] = None
        self.kl_divergence_: Optional[float] = None

    def _binary_search_perplexity(self, dist_sq: np.array) -> np.array:
        """Binary search to find sigma that gives desired perplexity."""
        n_samples = dist_sq.shape[0]
        P = np.zeros((n_samples, n_samples), dtype=np.float64)
        target_entropy = np.log(self.perplexity)
        eps = 1e-5

        for i in range(n_samples):
            # Exclude diagonal element
            beta = 1.0
            beta_min = -np.inf
            beta_max = np.inf
            dist_i = dist_sq[i, np.arange(n_samples) != i]

            for _ in range(50):
                # Compute entropy and perplexity
                exp_dist = np.exp(-dist_i * beta)
                sum_exp = np.sum(exp_dist) + 1e-10
                entropy = np.log(sum_exp) + beta * np.sum(dist_i * exp_dist) / sum_exp
                entropy_diff = entropy - target_entropy

                if abs(entropy_diff) < eps:
                    break

                if entropy_diff > 0:
                    beta_min = beta
                    if beta_max == np.inf:
                        beta *= 2.0
                    else:
                        beta = (beta + beta_max) / 2.0
                else:
                    beta_max = beta
                    if beta_min == -np.inf:
                        beta /= 2.0
                    else:
                        beta = (beta + beta_min) / 2.0

            # Compute P values
            exp_dist = np.exp(-dist_i * beta)
            sum_exp = np.sum(exp_dist) + 1e-10
            P[i, np.arange(n_samples) != i] = exp_dist / sum_exp

        return P

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

        # Compute pairwise squared distances
        metric_fn = get_metric(self.metric)
        X_mx = mx.array(X_np)
        dist = np.zeros((n_samples, n_samples), dtype=np.float32)
        for i in range(n_samples):
            dist[i] = np.array(metric_fn(X_mx[i:i+1], X_mx))
        dist_sq = dist ** 2

        # Compute joint probabilities P
        P = self._binary_search_perplexity(dist_sq)
        P = (P + P.T) / (2 * n_samples)
        P = np.maximum(P, 1e-12)

        # Early exaggeration
        P *= self.early_exaggeration

        # Initialize embedding
        Y = self.rng.randn(n_samples, self.n_components).astype(np.float32) * 1e-4
        Y_mx = mx.array(Y)

        # Gradient descent parameters
        momentum = 0.5
        final_momentum = 0.8
        momentum_switch_iter = 250
        stop_lying_iter = 250

        dY = np.zeros_like(Y)
        iY = np.zeros_like(Y)
        gains = np.ones_like(Y)

        for iter in range(self.max_iter):
            # Compute low-dimensional affinities Q
            sum_Y = mx.sum(Y_mx ** 2, axis=1)
            num = 1.0 / (1.0 + sum_Y[:, np.newaxis] + sum_Y[np.newaxis, :] - 2 * Y_mx @ Y_mx.T)
            # Set diagonal to 0
            num = num * (1 - mx.eye(n_samples))
            Q = num / mx.sum(num)
            Q = mx.maximum(Q, 1e-12)

            # Compute gradient
            P_minus_Q = mx.array(P) - Q
            PQ_num = P_minus_Q * num
            dY_mx = mx.zeros_like(Y_mx)
            for i in range(n_samples):
                dY_mx[i] = 4 * mx.sum((PQ_num[i, :, np.newaxis]) * (Y_mx[i:i+1, :] - Y_mx), axis=0)
            dY = np.array(dY_mx)

            # Update momentum
            if iter < momentum_switch_iter:
                current_momentum = momentum
            else:
                current_momentum = final_momentum

            # Update gains
            gains = (gains + 0.2) * ((dY > 0) != (iY > 0)) + (gains * 0.8) * ((dY > 0) == (iY > 0))
            gains[gains < 0.01] = 0.01

            # Update embedding
            iY = current_momentum * iY - self.learning_rate * gains * dY
            Y += iY
            Y -= np.mean(Y, axis=0)
            Y_mx = mx.array(Y)

            # Stop early exaggeration
            if iter == stop_lying_iter:
                P /= self.early_exaggeration

            # Print progress every 100 iterations
            if (iter + 1) % 100 == 0:
                kl_div = np.sum(P * np.log(P / np.array(Q)))
                print(f"Iteration {iter+1}/{self.max_iter}, KL divergence: {kl_div:.4f}")

        self.embedding_ = Y
        self.kl_divergence_ = np.sum(P * np.log(P / np.array(Q)))

        return mx.array(Y.astype(np.float32))

    def fit(self, X: Union[mx.array, np.array], y: Optional[Union[mx.array, np.array]] = None) -> "TSNE":
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
