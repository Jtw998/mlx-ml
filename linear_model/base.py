import mlx.core as mx
import numpy as np
from typing import Optional, Union, Callable
from ..base.base_estimator import BaseEstimator
from .solver import SGDSolver, LBFGSSolver


class LinearModel(BaseEstimator):
    """
    Base class for all linear models.
    """

    def __init__(
        self,
        fit_intercept: bool = True,
        normalize: bool = False,
        copy_X: bool = True,
        n_jobs: Optional[int] = None
    ):
        super().__init__()
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.copy_X = copy_X
        self.n_jobs = n_jobs

        self.coef_: Optional[mx.array] = None
        self.intercept_: Optional[mx.array] = None
        self.n_features_in_: Optional[int] = None

    def _preprocess_data(self, X: Union[mx.array, np.array], y: Optional[Union[mx.array, np.array]] = None) -> tuple:
        """
        Preprocess input data: convert to MLX arrays, normalize if needed, add intercept.

        Parameters:
            X: Input features
            y: Target values (optional)

        Returns:
            Processed X, y, X_mean, X_scale
        """
        X, y = self._validate_data(X, y)
        self.n_features_in_ = X.shape[1]

        X_mean = None
        X_scale = None

        if self.normalize or self.fit_intercept:
            # Compute mean and std for centering/scaling
            X_mean = mx.mean(X, axis=0)
            X_scale = mx.std(X, axis=0, ddof=0)
            X_scale = mx.where(X_scale == 0, 1.0, X_scale)

            # Center data
            X = X - X_mean

            if self.normalize:
                X = X / X_scale

        if y is not None and self.fit_intercept:
            # Center target
            y_mean = mx.mean(y)
            y = y - y_mean
        else:
            y_mean = None

        return X, y, X_mean, X_scale, y_mean

    def _set_intercept(self, X_mean: mx.array, X_scale: mx.array, y_mean: mx.array, intercept: Optional[mx.array] = None):
        """
        Compute the intercept term based on preprocessed data statistics.

        Parameters:
            X_mean: Mean of original features
            X_scale: Standard deviation of original features
            y_mean: Mean of original target
            intercept: Intercept from preprocessed model
        """
        if self.fit_intercept:
            if self.normalize:
                self.coef_ = self.coef_ / X_scale

            if intercept is None:
                self.intercept_ = y_mean - mx.sum(self.coef_ * X_mean)
            else:
                self.intercept_ = intercept + y_mean - mx.sum(self.coef_ * X_mean)
        else:
            self.intercept_ = mx.array(0.0) if intercept is None else intercept

    def predict(self, X: Union[mx.array, np.array]) -> mx.array:
        """
        Predict using the linear model.

        Parameters:
            X: Samples of shape (n_samples, n_features)

        Returns:
            Predicted values of shape (n_samples,)
        """
        if self.coef_ is None:
            raise ValueError("Model not fitted yet. Call 'fit' first.")

        X, _ = self._validate_data(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {X.shape[1]}")

        y_pred = X @ self.coef_
        if self.intercept_ is not None:
            y_pred = y_pred + self.intercept_

        return y_pred


def get_solver(solver: Union[str, Callable], **kwargs) -> Callable:
    """
    Get solver instance by name or return custom solver.

    Parameters:
        solver: Name of the solver ('sgd', 'lbfgs') or custom solver instance
        **kwargs: Additional parameters for the solver

    Returns:
        Solver instance
    """
    if isinstance(solver, str):
        solver_map = {
            'sgd': SGDSolver,
            'lbfgs': LBFGSSolver
        }
        if solver not in solver_map:
            raise ValueError(f"Unknown solver '{solver}'. Available solvers: {list(solver_map.keys())}")
        return solver_map[solver](**kwargs)
    elif callable(getattr(solver, 'solve', None)):
        return solver
    else:
        raise ValueError("Solver must be a string or an object with a 'solve' method")
