import mlx.core as mx
import numpy as np
from typing import Optional, Union
from .base import LinearModel, get_solver
from .solver import make_linear_loss_fn


class LinearRegression(LinearModel):
    """
    Ordinary least squares Linear Regression.

    Parameters:
        fit_intercept: Whether to calculate the intercept for this model
        normalize: If True, the regressors will be normalized before regression
        copy_X: If True, X will be copied; else, it may be overwritten
        n_jobs: Number of jobs to use for computation (unused in MLX implementation)
        solver: Solver to use for optimization: 'lbfgs', 'sgd', or custom solver
        solver_kwargs: Additional keyword arguments passed to the solver
    """

    def __init__(
        self,
        fit_intercept: bool = True,
        normalize: bool = False,
        copy_X: bool = True,
        n_jobs: Optional[int] = None,
        solver: str = 'sgd',
        solver_kwargs: Optional[dict] = None
    ):
        super().__init__(fit_intercept, normalize, copy_X, n_jobs)
        self.solver = solver
        self.solver_kwargs = solver_kwargs or {}

    def fit(self, X: Union[mx.array, np.array], y: Union[mx.array, np.array]) -> "LinearRegression":
        """
        Fit linear model using closed-form solution (OLS).

        Parameters:
            X: Training data of shape (n_samples, n_features)
            y: Target values of shape (n_samples,)

        Returns:
            Fitted estimator
        """
        X, y, X_mean, X_scale, y_mean = self._preprocess_data(X, y)
        n_samples, n_features = X.shape

        # Closed-form solution: beta = (X^T X)^-1 X^T y
        # Convert to numpy for linalg operations not supported on GPU
        X_np = np.array(X)
        y_np = np.array(y)

        if n_samples >= n_features:
            # Use standard OLS
            XTX = X_np.T @ X_np
            XTy = X_np.T @ y_np
            weights = np.linalg.inv(XTX) @ XTy
        else:
            # Use SVD for underdetermined systems
            U, S, Vt = np.linalg.svd(X_np, full_matrices=False)
            weights = Vt.T @ (U.T @ y_np / S)

        weights = mx.array(weights)

        intercept = mx.array(0.0) if self.fit_intercept else None
        self.coef_ = weights
        self._set_intercept(X_mean, X_scale, y_mean, intercept)

        return self


class Ridge(LinearModel):
    """
    Linear least squares with L2 regularization.

    Parameters:
        alpha: Regularization strength (>= 0)
        fit_intercept: Whether to calculate the intercept for this model
        normalize: If True, the regressors will be normalized before regression
        copy_X: If True, X will be copied; else, it may be overwritten
        max_iter: Maximum number of iterations for solver
        tol: Precision of the solution
        solver: Solver to use for optimization: 'lbfgs', 'sgd', or custom solver
        solver_kwargs: Additional keyword arguments passed to the solver
    """

    def __init__(
        self,
        alpha: float = 1.0,
        fit_intercept: bool = True,
        normalize: bool = False,
        copy_X: bool = True,
        max_iter: int = 1000,
        tol: float = 1e-4,
        solver: str = 'sgd',
        solver_kwargs: Optional[dict] = None
    ):
        super().__init__(fit_intercept, normalize, copy_X)
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.solver = solver
        self.solver_kwargs = solver_kwargs or {}

    def fit(self, X: Union[mx.array, np.array], y: Union[mx.array, np.array]) -> "Ridge":
        """
        Fit Ridge regression model using closed-form solution.

        Parameters:
            X: Training data of shape (n_samples, n_features)
            y: Target values of shape (n_samples,)

        Returns:
            Fitted estimator
        """
        X, y, X_mean, X_scale, y_mean = self._preprocess_data(X, y)
        n_samples, n_features = X.shape

        # Closed-form solution: beta = (X^T X + alpha * I)^-1 X^T y
        # Convert to numpy for linalg operations not supported on GPU
        X_np = np.array(X)
        y_np = np.array(y)

        XTX = X_np.T @ X_np
        alpha_I = self.alpha * np.eye(n_features, dtype=X_np.dtype)
        XTy = X_np.T @ y_np
        weights = np.linalg.inv(XTX + alpha_I) @ XTy
        weights = mx.array(weights)

        intercept = mx.array(0.0) if self.fit_intercept else None
        self.coef_ = weights
        self._set_intercept(X_mean, X_scale, y_mean, intercept)

        return self


class Lasso(LinearModel):
    """
    Linear Model trained with L1 prior as regularizer (aka the Lasso).

    Parameters:
        alpha: Regularization strength (>= 0)
        fit_intercept: Whether to calculate the intercept for this model
        normalize: If True, the regressors will be normalized before regression
        copy_X: If True, X will be copied; else, it may be overwritten
        max_iter: Maximum number of iterations for solver
        tol: Precision of the solution
        solver: Solver to use for optimization: 'sgd', 'lbfgs', or custom solver
        solver_kwargs: Additional keyword arguments passed to the solver
    """

    def __init__(
        self,
        alpha: float = 1.0,
        fit_intercept: bool = True,
        normalize: bool = False,
        copy_X: bool = True,
        max_iter: int = 1000,
        tol: float = 1e-4,
        solver: str = 'sgd',
        solver_kwargs: Optional[dict] = None
    ):
        super().__init__(fit_intercept, normalize, copy_X)
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.solver = solver
        self.solver_kwargs = solver_kwargs or {}

    def fit(self, X: Union[mx.array, np.array], y: Union[mx.array, np.array]) -> "Lasso":
        """
        Fit Lasso regression model using coordinate descent.

        Parameters:
            X: Training data of shape (n_samples, n_features)
            y: Target values of shape (n_samples,)

        Returns:
            Fitted estimator
        """
        X, y, X_mean, X_scale, y_mean = self._preprocess_data(X, y)
        n_samples, n_features = X.shape

        # Convert to numpy for coordinate descent
        X_np = np.array(X)
        y_np = np.array(y)

        # Initialize coefficients
        weights = np.zeros(n_features, dtype=X_np.dtype)
        alpha = self.alpha

        # Precompute squared norm of each feature
        norm_cols = np.sum(X_np ** 2, axis=0)
        norm_cols = np.where(norm_cols == 0, 1.0, norm_cols)

        # Coordinate descent
        for _ in range(self.max_iter):
            weights_old = weights.copy()

            for j in range(n_features):
                # Compute partial residual
                residual = y_np - X_np @ weights
                # Add back the contribution of feature j
                residual += X_np[:, j] * weights[j]
                # Compute rho
                rho = np.sum(X_np[:, j] * residual)
                # Soft thresholding
                z = rho / norm_cols[j]
                weights[j] = np.sign(z) * np.maximum(np.abs(z) - alpha / norm_cols[j], 0)

            # Check convergence
            if np.max(np.abs(weights - weights_old)) < self.tol:
                break

        self.coef_ = mx.array(weights)
        self._set_intercept(X_mean, X_scale, y_mean, None)

        return self


class ElasticNet(LinearModel):
    """
    Linear regression with combined L1 and L2 regularizations.

    Parameters:
        alpha: Regularization strength (>= 0)
        l1_ratio: The ElasticNet mixing parameter (0 <= l1_ratio <= 1)
            0 = pure L2 penalty, 1 = pure L1 penalty
        fit_intercept: Whether to calculate the intercept for this model
        normalize: If True, the regressors will be normalized before regression
        copy_X: If True, X will be copied; else, it may be overwritten
        max_iter: Maximum number of iterations for solver
        tol: Precision of the solution
        solver: Solver to use for optimization: 'sgd', 'lbfgs', or custom solver
        solver_kwargs: Additional keyword arguments passed to the solver
    """

    def __init__(
        self,
        alpha: float = 1.0,
        l1_ratio: float = 0.5,
        fit_intercept: bool = True,
        normalize: bool = False,
        copy_X: bool = True,
        max_iter: int = 1000,
        tol: float = 1e-4,
        solver: str = 'sgd',
        solver_kwargs: Optional[dict] = None
    ):
        super().__init__(fit_intercept, normalize, copy_X)
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.tol = tol
        self.solver = solver
        self.solver_kwargs = solver_kwargs or {}

        if l1_ratio < 0 or l1_ratio > 1:
            raise ValueError(f"l1_ratio must be between 0 and 1, got {l1_ratio}")

    def fit(self, X: Union[mx.array, np.array], y: Union[mx.array, np.array]) -> "ElasticNet":
        """
        Fit ElasticNet regression model using coordinate descent.

        Parameters:
            X: Training data of shape (n_samples, n_features)
            y: Target values of shape (n_samples,)

        Returns:
            Fitted estimator
        """
        X, y, X_mean, X_scale, y_mean = self._preprocess_data(X, y)
        n_samples, n_features = X.shape

        # Convert to numpy for coordinate descent
        X_np = np.array(X)
        y_np = np.array(y)

        # Initialize coefficients
        weights = np.zeros(n_features, dtype=X_np.dtype)
        alpha = self.alpha
        l1_ratio = self.l1_ratio

        # Precompute squared norm of each feature + L2 regularization
        norm_cols = np.sum(X_np ** 2, axis=0) + alpha * (1 - l1_ratio)
        norm_cols = np.where(norm_cols == 0, 1.0, norm_cols)
        l1_reg = alpha * l1_ratio

        # Coordinate descent
        for _ in range(self.max_iter):
            weights_old = weights.copy()

            for j in range(n_features):
                # Compute partial residual
                residual = y_np - X_np @ weights
                # Add back the contribution of feature j
                residual += X_np[:, j] * weights[j]
                # Compute rho
                rho = np.sum(X_np[:, j] * residual)
                # Soft thresholding
                z = rho / norm_cols[j]
                weights[j] = np.sign(z) * np.maximum(np.abs(z) - l1_reg / norm_cols[j], 0)

            # Check convergence
            if np.max(np.abs(weights - weights_old)) < self.tol:
                break

        self.coef_ = mx.array(weights)
        self._set_intercept(X_mean, X_scale, y_mean, None)

        return self
