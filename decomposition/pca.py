import mlx.core as mx
import numpy as np
from typing import Optional, Union
from ..base.base_estimator import BaseEstimator


class PCA(BaseEstimator):
    """
    Principal Component Analysis (PCA).

    Linear dimensionality reduction using Singular Value Decomposition of the
    data to project it to a lower dimensional space. The input data is centered
    but not scaled for each feature before applying the SVD.

    Parameters:
        n_components: Number of components to keep. If None, keep all components.
        whiten: When True, the components are scaled to have unit variance
        svd_solver: SVD solver to use. Currently only 'full' is supported.
        random_state: Random seed for reproducibility
    """

    def __init__(
        self,
        n_components: Optional[int] = None,
        whiten: bool = False,
        svd_solver: str = 'full',
        random_state: Optional[int] = None
    ):
        super().__init__()
        self.n_components = n_components
        self.whiten = whiten
        self.svd_solver = svd_solver
        self.random_state = random_state

        self.components_: Optional[mx.array] = None
        self.explained_variance_: Optional[mx.array] = None
        self.explained_variance_ratio_: Optional[mx.array] = None
        self.singular_values_: Optional[mx.array] = None
        self.mean_: Optional[mx.array] = None
        self.n_components_: Optional[int] = None
        self.n_features_in_: Optional[int] = None
        self.n_samples_: Optional[int] = None

        if svd_solver != 'full':
            raise NotImplementedError("Only 'full' SVD solver is currently supported")

    def fit(self, X: Union[mx.array, np.array], y: Optional[Union[mx.array, np.array]] = None) -> "PCA":
        """
        Fit the model with X.

        Parameters:
            X: Training data of shape (n_samples, n_features)
            y: Ignored, present for API compatibility

        Returns:
            Fitted estimator
        """
        X, _ = self._validate_data(X)
        self.n_samples_, self.n_features_in_ = X.shape

        # Determine number of components
        if self.n_components is None:
            self.n_components_ = min(self.n_samples_, self.n_features_in_)
        else:
            self.n_components_ = min(self.n_components, min(self.n_samples_, self.n_features_in_))
            if self.n_components_ < 1 or self.n_components_ > min(self.n_samples_, self.n_features_in_):
                raise ValueError(f"n_components={self.n_components} is invalid for data with "
                                 f"n_samples={self.n_samples_}, n_features={self.n_features_in_}")

        # Center data
        self.mean_ = mx.mean(X, axis=0)
        X_centered = X - self.mean_

        # Convert to numpy for SVD (MLX SVD is not fully supported yet)
        X_np = np.array(X_centered)

        # Compute SVD
        U, S, Vt = np.linalg.svd(X_np, full_matrices=False)

        # Get principal components
        components = Vt[:self.n_components_]
        self.components_ = mx.array(components)

        # Compute explained variance
        explained_variance = (S ** 2) / (self.n_samples_ - 1)
        self.explained_variance_ = mx.array(explained_variance[:self.n_components_])

        total_variance = explained_variance.sum()
        self.explained_variance_ratio_ = mx.array(explained_variance[:self.n_components_] / total_variance)

        # Singular values
        self.singular_values_ = mx.array(S[:self.n_components_])

        # Handle whitening
        if self.whiten:
            # Scale components by 1/sqrt(explained_variance)
            self.components_ = self.components_ / mx.sqrt(self.explained_variance_[:, None] + 1e-8)

        return self

    def transform(self, X: Union[mx.array, np.array]) -> mx.array:
        """
        Apply dimensionality reduction to X.

        Parameters:
            X: Samples of shape (n_samples, n_features)

        Returns:
            X projected to the first n_components dimensions
        """
        if self.components_ is None or self.mean_ is None:
            raise ValueError("Model not fitted yet. Call 'fit' first.")

        X, _ = self._validate_data(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {X.shape[1]}")

        # Center data
        X_centered = X - self.mean_

        # Project to components
        X_projected = X_centered @ self.components_.T

        return X_projected

    def fit_transform(self, X: Union[mx.array, np.array], y: Optional[Union[mx.array, np.array]] = None) -> mx.array:
        """
        Fit the model with X and apply the dimensionality reduction on X.

        Parameters:
            X: Training data of shape (n_samples, n_features)
            y: Ignored, present for API compatibility

        Returns:
            X projected to the first n_components dimensions
        """
        return self.fit(X).transform(X)

    def inverse_transform(self, X: Union[mx.array, np.array]) -> mx.array:
        """
        Transform data back to its original space.

        Parameters:
            X: Samples in the transformed space of shape (n_samples, n_components)

        Returns:
            X reconstructed from the projected data
        """
        if self.components_ is None or self.mean_ is None:
            raise ValueError("Model not fitted yet. Call 'fit' first.")

        X, _ = self._validate_data(X)
        if X.shape[1] != self.n_components_:
            raise ValueError(f"Expected {self.n_components_} features, got {X.shape[1]}")

        # Inverse projection
        if self.whiten:
            # Undo whitening first
            X_scaled = X * mx.sqrt(self.explained_variance_[None, :] + 1e-8)
            X_original = X_scaled @ self.components_ + self.mean_
        else:
            X_original = X @ self.components_ + self.mean_

        return X_original
