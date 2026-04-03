import mlx.core as mx
import mlx.nn as nn
from mlx.core import value_and_grad
import numpy as np
from typing import Optional, Union
from ..base.base_estimator import BaseEstimator
from .base import get_solver
from .solver import make_linear_loss_fn


class LogisticRegression(BaseEstimator):
    """
    Logistic Regression (aka logit, MaxEnt) classifier.

    Parameters:
        penalty: Specify the norm of the penalty: 'l1', 'l2', 'elasticnet', or None
        C: Inverse of regularization strength; must be a positive float
        l1_ratio: The ElasticNet mixing parameter, only used if penalty='elasticnet'
        fit_intercept: Specifies if a constant should be added to the decision function
        max_iter: Maximum number of iterations for the solver
        tol: Tolerance for stopping criteria
        solver: Algorithm to use in the optimization problem: 'lbfgs', 'sgd'
        multi_class: Specifies the multi-class strategy: 'auto', 'ovr', 'multinomial'
        class_weight: Weights associated with classes, None for equal weights
        random_state: Random seed for shuffling data
        solver_kwargs: Additional keyword arguments passed to the solver
    """

    def __init__(
        self,
        penalty: Optional[str] = 'l2',
        C: float = 1.0,
        l1_ratio: Optional[float] = None,
        fit_intercept: bool = True,
        max_iter: int = 100,
        tol: float = 1e-4,
        solver: str = 'lbfgs',
        multi_class: str = 'auto',
        class_weight: Optional[Union[str, dict]] = None,
        random_state: Optional[int] = None,
        solver_kwargs: Optional[dict] = None
    ):
        super().__init__()
        self.penalty = penalty
        self.C = C
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.solver = solver
        self.multi_class = multi_class
        self.class_weight = class_weight
        self.random_state = random_state
        self.solver_kwargs = solver_kwargs or {}

        # Validate parameters
        valid_penalties = ['l1', 'l2', 'elasticnet', None]
        if penalty not in valid_penalties:
            raise ValueError(f"Invalid penalty '{penalty}'. Valid options: {valid_penalties}")

        if penalty == 'elasticnet' and l1_ratio is None:
            raise ValueError("l1_ratio must be specified when penalty='elasticnet'")

        if C <= 0:
            raise ValueError(f"C must be positive, got {C}")

        self.coef_: Optional[mx.array] = None
        self.intercept_: Optional[mx.array] = None
        self.classes_: Optional[np.array] = None
        self.n_features_in_: Optional[int] = None
        self.n_classes_: Optional[int] = None

    def _get_loss_function(self, n_classes: int):
        """
        Get the appropriate loss function based on number of classes.
        """
        if n_classes == 2:
            # Binary classification with sigmoid
            def loss_fn(y_pred, y_true):
                logits = y_pred
                # Convert y_true to float
                y_true = y_true.astype(mx.float32)
                # Binary cross entropy with logits
                return mx.mean(mx.maximum(logits, 0) - logits * y_true + mx.log(1 + mx.exp(-mx.abs(logits))))
        else:
            # Multiclass classification with softmax
            def loss_fn(y_pred, y_true):
                logits = y_pred
                # Cross entropy loss
                return mx.mean(nn.losses.cross_entropy(logits, y_true))

        return loss_fn

    def fit(self, X: Union[mx.array, np.array], y: Union[mx.array, np.array]) -> "LogisticRegression":
        """
        Fit the model according to the given training data.

        Parameters:
            X: Training data of shape (n_samples, n_features)
            y: Target values of shape (n_samples,)

        Returns:
            Fitted estimator
        """
        X, y = self._validate_data(X, y)
        self.n_features_in_ = X.shape[1]
        X_np = np.array(X)

        # Convert y to numpy for class processing
        y_np = np.array(y)
        self.classes_ = np.unique(y_np)
        self.n_classes_ = len(self.classes_)

        # Binary classification only for now (simplified stable implementation)
        if self.n_classes_ != 2:
            raise NotImplementedError("Multiclass Logistic Regression is coming soon. Currently only binary classification is supported.")

        # Encode labels to 0/1
        y_encoded = (y_np == self.classes_[1]).astype(np.float32)

        # Add bias term if fit_intercept
        if self.fit_intercept:
            X_bias = np.hstack([X_np, np.ones((X_np.shape[0], 1))])
        else:
            X_bias = X_np

        n_features = X_bias.shape[1]
        weights = np.zeros(n_features, dtype=np.float32)

        # Gradient descent optimization
        learning_rate = 0.01
        for _ in range(self.max_iter):
            # Compute predictions
            z = X_bias @ weights
            y_pred = 1 / (1 + np.exp(-z))  # Sigmoid

            # Compute gradient
            grad = X_bias.T @ (y_pred - y_encoded) / X_bias.shape[0]

            # Add L2 regularization
            if self.penalty == 'l2' or self.penalty == 'elasticnet':
                alpha = 1.0 / self.C
                grad[:-1] += alpha * weights[:-1] / X_bias.shape[0]  # Don't regularize intercept

            # Update weights
            weights -= learning_rate * grad

            # Check convergence
            if np.linalg.norm(grad) < self.tol:
                break

        # Split weights and intercept
        if self.fit_intercept:
            self.coef_ = mx.array(weights[:-1].reshape(1, -1))
            self.intercept_ = mx.array(weights[-1:])
        else:
            self.coef_ = mx.array(weights.reshape(1, -1))
            self.intercept_ = mx.array(0.0)

        return self

    def predict_proba(self, X: Union[mx.array, np.array]) -> mx.array:
        """
        Probability estimates.

        Parameters:
            X: Samples of shape (n_samples, n_features)

        Returns:
            Probability of the sample for each class in the model, of shape (n_samples, n_classes)
        """
        if self.coef_ is None or self.intercept_ is None:
            raise ValueError("Model not fitted yet. Call 'fit' first.")

        X, _ = self._validate_data(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {X.shape[1]}")

        logits = X @ self.coef_.T + self.intercept_

        if self.n_classes_ == 2:
            # Binary classification
            prob_1 = mx.sigmoid(logits)
            prob_0 = 1 - prob_1
            return mx.concatenate([prob_0, prob_1], axis=1)
        else:
            # Multiclass classification
            return mx.softmax(logits, axis=1)

    def predict(self, X: Union[mx.array, np.array]) -> mx.array:
        """
        Predict class labels for samples in X.

        Parameters:
            X: Samples of shape (n_samples, n_features)

        Returns:
            Predicted class labels of shape (n_samples,)
        """
        proba = self.predict_proba(X)
        class_indices = mx.argmax(proba, axis=1)
        # Map indices back to original classes
        return mx.array(self.classes_[np.array(class_indices)])
