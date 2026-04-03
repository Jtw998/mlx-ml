import mlx.core as mx
import numpy as np
from typing import Optional, Union, Tuple
from ..base.base_estimator import BaseEstimator


class GaussianMixture(BaseEstimator):
    """
    Gaussian Mixture.

    Representation of a Gaussian mixture model probability distribution.
    This class allows to estimate the parameters of a Gaussian mixture distribution.

    Parameters:
        n_components: The number of mixture components.
        covariance_type: String describing the type of covariance parameters to use.
            Currently only supports 'full' and 'diag'.
        tol: The convergence threshold. EM iterations will stop when the lower bound average gain is below this threshold.
        max_iter: The number of EM iterations to perform.
        random_state: Controls the randomness of the estimator.
    """

    def __init__(
        self,
        n_components: int = 1,
        covariance_type: str = 'full',
        tol: float = 1e-3,
        max_iter: int = 100,
        random_state: Optional[int] = None
    ):
        super().__init__()
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

        valid_covariance_types = ['full', 'diag']
        if covariance_type not in valid_covariance_types:
            raise ValueError(f"Invalid covariance_type '{covariance_type}'. Valid options: {valid_covariance_types}")

        self.weights_: Optional[np.array] = None
        self.means_: Optional[np.array] = None
        self.covariances_: Optional[np.array] = None
        self.precisions_: Optional[np.array] = None
        self.converged_: bool = False
        self.lower_bound_: Optional[float] = None

    def _initialize_parameters(self, X: np.array):
        """Initialize model parameters using K-means initialization."""
        n_samples, n_features = X.shape

        # Randomly select n_components points as initial means
        indices = self.rng.choice(n_samples, size=self.n_components, replace=False)
        self.means_ = X[indices].copy()

        # Initialize weights uniformly
        self.weights_ = np.ones(self.n_components) / self.n_components

        # Initialize covariances
        if self.covariance_type == 'full':
            self.covariances_ = np.array([np.eye(n_features) for _ in range(self.n_components)])
        else:  # diag
            self.covariances_ = np.ones((self.n_components, n_features))

    def _estimate_log_gaussian_prob(self, X: np.array) -> np.array:
        """Estimate the log Gaussian probability for each sample in each component."""
        n_samples, n_features = X.shape
        log_prob = np.zeros((n_samples, self.n_components))

        for k in range(self.n_components):
            mean = self.means_[k]
            if self.covariance_type == 'full':
                cov = self.covariances_[k]
                # Add small epsilon to diagonal for numerical stability
                cov = cov + 1e-6 * np.eye(n_features)
                log_det = np.log(np.linalg.det(cov))
                inv_cov = np.linalg.inv(cov)
                diff = X - mean
                exp_term = np.sum(diff @ inv_cov * diff, axis=1)
            else:  # diag
                cov = self.covariances_[k] + 1e-6
                log_det = np.sum(np.log(cov))
                inv_cov = 1.0 / cov
                diff = X - mean
                exp_term = np.sum(diff ** 2 * inv_cov, axis=1)

            log_prob[:, k] = -0.5 * (n_features * np.log(2 * np.pi) + log_det + exp_term)

        return log_prob

    def _e_step(self, X: np.array) -> Tuple[np.array, float]:
        """E step: compute responsibilities."""
        log_prob = self._estimate_log_gaussian_prob(X)
        weighted_log_prob = log_prob + np.log(self.weights_ + 1e-10)
        log_resp = weighted_log_prob - np.logaddexp.reduce(weighted_log_prob, axis=1, keepdims=True)
        resp = np.exp(log_resp)
        lower_bound = np.mean(np.logaddexp.reduce(weighted_log_prob, axis=1))
        return resp, lower_bound

    def _m_step(self, X: np.array, resp: np.array):
        """M step: update parameters."""
        n_samples, n_features = X.shape
        weights = np.sum(resp, axis=0)

        # Update weights
        self.weights_ = weights / n_samples

        # Update means
        self.means_ = (resp.T @ X) / weights[:, np.newaxis]

        # Update covariances
        for k in range(self.n_components):
            diff = X - self.means_[k]
            resp_k = resp[:, k, np.newaxis]
            if self.covariance_type == 'full':
                cov = (resp_k * diff).T @ diff / weights[k]
                self.covariances_[k] = cov
            else:  # diag
                cov = np.sum(resp_k * diff ** 2, axis=0) / weights[k]
                self.covariances_[k] = cov

    def fit(self, X: Union[mx.array, np.array], y: Optional[Union[mx.array, np.array]] = None) -> "GaussianMixture":
        """
        Estimate model parameters with the EM algorithm.

        Parameters:
            X: Training data of shape (n_samples, n_features)
            y: Not used, present for API consistency.

        Returns:
            Fitted estimator
        """
        X, _ = self._validate_data(X, y)
        X_np = np.array(X)

        self._initialize_parameters(X_np)

        lower_bound = -np.inf
        self.converged_ = False

        for iter in range(self.max_iter):
            prev_lower_bound = lower_bound

            # E step
            resp, lower_bound = self._e_step(X_np)

            # Check convergence
            change = lower_bound - prev_lower_bound
            if abs(change) < self.tol:
                self.converged_ = True
                break

            # M step
            self._m_step(X_np, resp)

        self.lower_bound_ = lower_bound

        return self

    def predict(self, X: Union[mx.array, np.array]) -> mx.array:
        """
        Predict the labels for the data samples in X using trained model.

        Parameters:
            X: Samples of shape (n_samples, n_features)

        Returns:
            Component labels of shape (n_samples,)
        """
        if self.means_ is None:
            raise ValueError("Model not fitted yet. Call 'fit' first.")

        X, _ = self._validate_data(X)
        X_np = np.array(X)

        log_prob = self._estimate_log_gaussian_prob(X_np)
        weighted_log_prob = log_prob + np.log(self.weights_ + 1e-10)
        labels = np.argmax(weighted_log_prob, axis=1)

        return mx.array(labels.astype(np.int32))

    def predict_proba(self, X: Union[mx.array, np.array]) -> mx.array:
        """
        Predict posterior probability of each component given the data.

        Parameters:
            X: Samples of shape (n_samples, n_features)

        Returns:
            Posterior probabilities of shape (n_samples, n_components)
        """
        if self.means_ is None:
            raise ValueError("Model not fitted yet. Call 'fit' first.")

        X, _ = self._validate_data(X)
        X_np = np.array(X)

        log_prob = self._estimate_log_gaussian_prob(X_np)
        weighted_log_prob = log_prob + np.log(self.weights_ + 1e-10)
        log_resp = weighted_log_prob - np.logaddexp.reduce(weighted_log_prob, axis=1, keepdims=True)
        resp = np.exp(log_resp)

        return mx.array(resp.astype(np.float32))
