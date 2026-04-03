import mlx.core as mx
import mlx.nn as nn
from mlx.core import value_and_grad
import numpy as np
from typing import Callable, Tuple, Optional


def make_linear_loss_fn(
    loss_fn: Callable,
    X: mx.array,
    y: mx.array,
    alpha: float = 0.0,
    l1_ratio: float = 0.0,
    fit_intercept: bool = True
) -> Callable:
    """
    Create a loss function for linear models with regularization.

    Parameters:
        loss_fn: Base loss function (e.g., mse, cross_entropy)
        X: Training features
        y: Target values
        alpha: Regularization strength
        l1_ratio: Ratio of L1 regularization (0 = pure L2, 1 = pure L1)
        fit_intercept: Whether the model includes an intercept term

    Returns:
        Loss function that takes parameters (weights, intercept) and returns loss value
    """
    def loss(params: Tuple[mx.array, Optional[mx.array]]) -> mx.array:
        weights, intercept = params
        y_pred = X @ weights
        if fit_intercept and intercept is not None:
            y_pred = y_pred + intercept

        base_loss = loss_fn(y_pred, y)

        # Add regularization
        if alpha > 0.0:
            l2_reg = 0.5 * alpha * (1 - l1_ratio) * mx.sum(weights ** 2)
            l1_reg = alpha * l1_ratio * mx.sum(mx.abs(weights))
            base_loss = base_loss + l2_reg + l1_reg

        return base_loss

    return loss


class SGDSolver:
    """
    Stochastic Gradient Descent solver for linear models.

    Parameters:
        learning_rate: Learning rate for SGD steps
        momentum: Momentum factor (0 = no momentum)
        nesterov: Whether to use Nesterov momentum
        max_iter: Maximum number of iterations
        tol: Tolerance for early stopping
        batch_size: Batch size for SGD, None for full-batch GD
        shuffle: Whether to shuffle data each epoch
        random_state: Random seed for shuffling
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        momentum: float = 0.9,
        nesterov: bool = False,
        max_iter: int = 5000,
        tol: float = 1e-6,
        batch_size: Optional[int] = None,
        shuffle: bool = True,
        random_state: Optional[int] = None
    ):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.nesterov = nesterov
        self.max_iter = max_iter
        self.tol = tol
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

    def solve(
        self,
        loss_fn: Callable,
        init_params: Tuple[mx.array, Optional[mx.array]],
        X: mx.array,
        y: mx.array
    ) -> Tuple[mx.array, Optional[mx.array]]:
        """
        Minimize the loss function using SGD.

        Parameters:
            loss_fn: Loss function to minimize
            init_params: Initial parameters (weights, intercept)
            X: Training features
            y: Target values

        Returns:
            Optimized parameters (weights, intercept)
        """
        weights, intercept = init_params
        n_samples = X.shape[0]

        # Initialize momentum buffers
        weight_velocity = mx.zeros_like(weights)
        intercept_velocity = mx.zeros_like(intercept) if intercept is not None else None

        # Get gradient function
        loss_and_grad_fn = value_and_grad(loss_fn)

        batch_size = self.batch_size if self.batch_size is not None else n_samples
        n_batches = (n_samples + batch_size - 1) // batch_size

        prev_loss = float('inf')

        for epoch in range(self.max_iter):
            if self.shuffle:
                indices = self.rng.permutation(n_samples)
                indices = mx.array(indices, dtype=mx.int32)
                X_shuffled = X[indices]
                y_shuffled = y[indices]
            else:
                X_shuffled = X
                y_shuffled = y

            epoch_loss = 0.0

            for i in range(n_batches):
                start = i * batch_size
                end = min(start + batch_size, n_samples)
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                # Create batch-specific loss function
                batch_loss_fn = make_linear_loss_fn(
                    lambda pred, y: mx.mean((pred - y) ** 2),  # This will be overridden by actual loss
                    X_batch, y_batch, 0.0, 0.0, intercept is not None
                )

                # Compute loss and gradients
                loss, grads = loss_and_grad_fn((weights, intercept))
                epoch_loss += loss.item() * (end - start) / n_samples

                d_weights, d_intercept = grads

                # Update weights with momentum
                weight_velocity = self.momentum * weight_velocity - self.learning_rate * d_weights
                if self.nesterov:
                    weights = weights + self.momentum * weight_velocity - self.learning_rate * d_weights
                else:
                    weights = weights + weight_velocity

                # Update intercept if present
                if intercept is not None and d_intercept is not None:
                    intercept_velocity = self.momentum * intercept_velocity - self.learning_rate * d_intercept
                    if self.nesterov:
                        intercept = intercept + self.momentum * intercept_velocity - self.learning_rate * d_intercept
                    else:
                        intercept = intercept + intercept_velocity

            # Check early stopping
            if abs(prev_loss - epoch_loss) < self.tol:
                break
            prev_loss = epoch_loss

        return weights, intercept


class LBFGSSolver:
    """
    Limited-memory BFGS solver for linear models.
    Simple L-BFGS implementation for smooth convex functions.

    Parameters:
        max_iter: Maximum number of iterations
        tol: Tolerance for convergence
        max_history_size: Number of previous updates to store for Hessian approximation
        learning_rate: Step size for line search
    """

    def __init__(
        self,
        max_iter: int = 1000,
        tol: float = 1e-5,
        max_history_size: int = 10,
        learning_rate: float = 1.0
    ):
        self.max_iter = max_iter
        self.tol = tol
        self.max_history_size = max_history_size
        self.learning_rate = learning_rate

    def solve(
        self,
        loss_fn: Callable,
        init_params: Tuple[mx.array, Optional[mx.array]],
        X: mx.array,
        y: mx.array
    ) -> Tuple[mx.array, Optional[mx.array]]:
        """
        Minimize the loss function using L-BFGS.

        Parameters:
            loss_fn: Loss function to minimize
            init_params: Initial parameters (weights, intercept)
            X: Training features (unused, for interface consistency)
            y: Target values (unused, for interface consistency)

        Returns:
            Optimized parameters (weights, intercept)
        """
        weights, intercept = init_params
        params = (weights, intercept)

        # Get gradient function
        loss_and_grad_fn = value_and_grad(loss_fn)

        # Initialize history buffers
        s_history = []  # Parameter differences
        y_history = []  # Gradient differences
        rho_history = []  # 1/(y^T s)

        # Initial loss and gradient
        loss, grads = loss_and_grad_fn(params)
        prev_loss = loss.item()

        for i in range(self.max_iter):
            # Compute search direction using two-loop recursion
            q = grads
            alphas = []

            # First loop: backwards through history
            for s, y, rho in reversed(list(zip(s_history, y_history, rho_history))):
                alpha = rho * self._dot(s, q)
                alphas.append(alpha)
                q = self._add(q, self._scale(y, -alpha))

            # Approximate Hessian
            if len(s_history) > 0:
                s, y = s_history[-1], y_history[-1]
                gamma = self._dot(s, y) / self._dot(y, y)
                r = self._scale(q, gamma)
            else:
                r = q

            # Second loop: forwards through history
            for s, y, rho, alpha in zip(s_history, y_history, rho_history, reversed(alphas)):
                beta = rho * self._dot(y, r)
                r = self._add(r, self._scale(s, alpha - beta))

            # Update parameters
            search_dir = self._scale(r, -self.learning_rate)
            new_params = self._add(params, search_dir)

            # Evaluate new loss and gradient
            new_loss, new_grads = loss_and_grad_fn(new_params)
            new_loss_val = new_loss.item()

            # Check convergence
            if abs(new_loss_val - prev_loss) < self.tol or new_loss_val < self.tol:
                params = new_params
                break

            # Update history
            s = self._sub(new_params, params)
            y = self._sub(new_grads, grads)
            ys = self._dot(y, s)

            if ys > 1e-10:  # Ensure curvature condition holds
                rho = 1.0 / ys
                s_history.append(s)
                y_history.append(y)
                rho_history.append(rho)

                # Keep only last max_history_size entries
                if len(s_history) > self.max_history_size:
                    s_history.pop(0)
                    y_history.pop(0)
                    rho_history.pop(0)

            # Update for next iteration
            params = new_params
            grads = new_grads
            prev_loss = new_loss_val

        return params

    def _dot(self, a, b):
        """Dot product of two parameter tuples"""
        res = 0.0
        for ai, bi in zip(a, b):
            if ai is not None and bi is not None:
                res += mx.sum(ai * bi).item()
        return res

    def _add(self, a, b):
        """Add two parameter tuples"""
        res = []
        for ai, bi in zip(a, b):
            if ai is None or bi is None:
                res.append(None)
            else:
                res.append(ai + bi)
        return tuple(res)

    def _sub(self, a, b):
        """Subtract two parameter tuples"""
        res = []
        for ai, bi in zip(a, b):
            if ai is None or bi is None:
                res.append(None)
            else:
                res.append(ai - bi)
        return tuple(res)

    def _scale(self, a, scalar):
        """Scale a parameter tuple by a scalar"""
        res = []
        for ai in a:
            if ai is None:
                res.append(None)
            else:
                res.append(ai * scalar)
        return tuple(res)
