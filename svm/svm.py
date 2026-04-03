import mlx.core as mx
import numpy as np
from typing import Optional, Union, Tuple
from ..base.base_estimator import BaseEstimator


class BaseSVM(BaseEstimator):
    """Base class for SVM classifiers and regressors."""

    def __init__(
        self,
        C: float = 1.0,
        kernel: str = 'rbf',
        degree: int = 3,
        gamma: Optional[Union[str, float]] = 'scale',
        coef0: float = 0.0,
        tol: float = 1e-3,
        max_iter: int = 1000,
        random_state: Optional[int] = None
    ):
        super().__init__()
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

        valid_kernels = ['linear', 'poly', 'rbf', 'sigmoid']
        if kernel not in valid_kernels:
            raise ValueError(f"Invalid kernel '{kernel}'. Valid options: {valid_kernels}")

    def _compute_gamma(self, n_features: int) -> float:
        """Compute gamma value based on input."""
        if isinstance(self.gamma, float):
            return self.gamma
        elif self.gamma == 'scale':
            return 1.0 / (n_features * np.var(self.X_train_) if hasattr(self, 'X_train_') and np.var(self.X_train_) > 0 else 1.0)
        elif self.gamma == 'auto':
            return 1.0 / n_features
        else:
            raise ValueError(f"Invalid gamma value: {self.gamma}")

    def _kernel(self, X1: np.array, X2: np.array) -> np.array:
        """Compute kernel matrix between X1 and X2."""
        n_samples_1, n_features = X1.shape
        n_samples_2 = X2.shape[0]

        gamma = self._compute_gamma(n_features)

        if self.kernel == 'linear':
            return X1 @ X2.T
        elif self.kernel == 'poly':
            return (gamma * (X1 @ X2.T) + self.coef0) ** self.degree
        elif self.kernel == 'rbf':
            # Compute squared Euclidean distances
            X1_sq = np.sum(X1 ** 2, axis=1)[:, np.newaxis]
            X2_sq = np.sum(X2 ** 2, axis=1)[np.newaxis, :]
            dist_sq = X1_sq + X2_sq - 2 * X1 @ X2.T
            return np.exp(-gamma * dist_sq)
        elif self.kernel == 'sigmoid':
            return np.tanh(gamma * (X1 @ X2.T) + self.coef0)
        else:
            raise ValueError(f"Unsupported kernel: {self.kernel}")

    def _smo_solve(self, K: np.array, y: np.array) -> Tuple[np.array, float]:
        """
        Simplified SMO (Sequential Minimal Optimization) algorithm for SVM training.

        Returns:
            alpha: Lagrange multipliers
            b: Bias term
        """
        n_samples = len(y)
        alpha = np.zeros(n_samples, dtype=np.float64)
        b = 0.0

        for iter in range(self.max_iter):
            alpha_changed = 0

            for i in range(n_samples):
                # Compute E_i
                f_i = np.sum(alpha * y * K[:, i]) + b
                E_i = f_i - y[i]

                # Check if alpha_i violates KKT conditions
                if (y[i] * E_i < -self.tol and alpha[i] < self.C) or (y[i] * E_i > self.tol and alpha[i] > 0):
                    # Select j != i randomly
                    j = self.rng.choice([idx for idx in range(n_samples) if idx != i])

                    # Compute E_j
                    f_j = np.sum(alpha * y * K[:, j]) + b
                    E_j = f_j - y[j]

                    # Save old alphas
                    alpha_i_old = alpha[i].copy()
                    alpha_j_old = alpha[j].copy()

                    # Compute L and H
                    if y[i] != y[j]:
                        L = max(0.0, alpha[j] - alpha[i])
                        H = min(self.C, self.C + alpha[j] - alpha[i])
                    else:
                        L = max(0.0, alpha[i] + alpha[j] - self.C)
                        H = min(self.C, alpha[i] + alpha[j])

                    if L == H:
                        continue

                    # Compute eta
                    eta = 2 * K[i, j] - K[i, i] - K[j, j]
                    if eta >= 0:
                        continue

                    # Update alpha_j
                    alpha[j] -= y[j] * (E_i - E_j) / eta

                    # Clip alpha_j
                    alpha[j] = np.clip(alpha[j], L, H)

                    if abs(alpha[j] - alpha_j_old) < 1e-5:
                        continue

                    # Update alpha_i
                    alpha[i] += y[i] * y[j] * (alpha_j_old - alpha[j])

                    # Compute b1 and b2
                    b1 = b - E_i - y[i] * (alpha[i] - alpha_i_old) * K[i, i] - y[j] * (alpha[j] - alpha_j_old) * K[i, j]
                    b2 = b - E_j - y[i] * (alpha[i] - alpha_i_old) * K[i, j] - y[j] * (alpha[j] - alpha_j_old) * K[j, j]

                    # Update b
                    if 0 < alpha[i] < self.C:
                        b = b1
                    elif 0 < alpha[j] < self.C:
                        b = b2
                    else:
                        b = (b1 + b2) / 2.0

                    alpha_changed += 1

            if alpha_changed == 0:
                break

        return alpha, b


class SVC(BaseSVM):
    """
    C-Support Vector Classification.

    Parameters:
        C: Regularization parameter. The strength of the regularization is inversely proportional to C.
        kernel: Specifies the kernel type to be used in the algorithm.
        degree: Degree of the polynomial kernel function ('poly').
        gamma: Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.
        coef0: Independent term in kernel function. It is only significant in 'poly' and 'sigmoid'.
        tol: Tolerance for stopping criterion.
        max_iter: Hard limit on iterations within solver.
        random_state: Controls the randomness of the estimator.
    """

    def __init__(
        self,
        C: float = 1.0,
        kernel: str = 'rbf',
        degree: int = 3,
        gamma: Optional[Union[str, float]] = 'scale',
        coef0: float = 0.0,
        tol: float = 1e-3,
        max_iter: int = 1000,
        random_state: Optional[int] = None
    ):
        super().__init__(
            C=C,
            kernel=kernel,
            degree=degree,
            gamma=gamma,
            coef0=coef0,
            tol=tol,
            max_iter=max_iter,
            random_state=random_state
        )

        self.classes_: Optional[np.array] = None
        self.n_classes_: Optional[int] = None
        self.support_: Optional[np.array] = None
        self.support_vectors_: Optional[np.array] = None
        self.dual_coef_: Optional[np.array] = None
        self.intercept_: Optional[float] = None

    def fit(self, X: Union[mx.array, np.array], y: Union[mx.array, np.array]) -> "SVC":
        """
        Fit the SVM model according to the given training data.

        Parameters:
            X: Training data of shape (n_samples, n_features)
            y: Target values of shape (n_samples,)

        Returns:
            Fitted estimator
        """
        X, y = self._validate_data(X, y)
        X_np = np.array(X)
        y_np = np.array(y)

        self.classes_ = np.unique(y_np)
        self.n_classes_ = len(self.classes_)

        if self.n_classes_ != 2:
            raise NotImplementedError("Only binary classification is currently supported.")

        # Convert y to -1/1
        y_bin = np.where(y_np == self.classes_[1], 1.0, -1.0)

        # Compute kernel matrix
        self.X_train_ = X_np
        K = self._kernel(X_np, X_np)

        # Solve SMO
        alpha, self.intercept_ = self._smo_solve(K, y_bin)

        # Get support vectors (alpha > 0)
        sv_mask = alpha > 1e-5
        self.support_ = np.where(sv_mask)[0]
        self.support_vectors_ = X_np[sv_mask]
        self.dual_coef_ = alpha[sv_mask] * y_bin[sv_mask]

        return self

    def decision_function(self, X: Union[mx.array, np.array]) -> mx.array:
        """
        Evaluate the decision function for the samples in X.

        Parameters:
            X: Samples of shape (n_samples, n_features)

        Returns:
            Decision function values of shape (n_samples,)
        """
        if self.support_vectors_ is None:
            raise ValueError("Model not fitted yet. Call 'fit' first.")

        X, _ = self._validate_data(X)
        X_np = np.array(X)

        # Compute kernel between X and support vectors
        K = self._kernel(X_np, self.support_vectors_)

        # Compute decision function
        decision = K @ self.dual_coef_ + self.intercept_

        return mx.array(decision.astype(np.float32))

    def predict(self, X: Union[mx.array, np.array]) -> mx.array:
        """
        Perform classification on samples in X.

        Parameters:
            X: Samples of shape (n_samples, n_features)

        Returns:
            Predicted classes of shape (n_samples,)
        """
        decision = np.array(self.decision_function(X))
        predictions = np.where(decision > 0, self.classes_[1], self.classes_[0])

        return mx.array(predictions.astype(np.int32))


class SVR(BaseSVM):
    """
    Epsilon-Support Vector Regression.

    Parameters:
        C: Regularization parameter. The strength of the regularization is inversely proportional to C.
        kernel: Specifies the kernel type to be used in the algorithm.
        degree: Degree of the polynomial kernel function ('poly').
        gamma: Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.
        coef0: Independent term in kernel function. It is only significant in 'poly' and 'sigmoid'.
        tol: Tolerance for stopping criterion.
        max_iter: Hard limit on iterations within solver.
        epsilon: Epsilon in the epsilon-SVR model.
        random_state: Controls the randomness of the estimator.
    """

    def __init__(
        self,
        C: float = 1.0,
        kernel: str = 'rbf',
        degree: int = 3,
        gamma: Optional[Union[str, float]] = 'scale',
        coef0: float = 0.0,
        tol: float = 1e-3,
        max_iter: int = 1000,
        epsilon: float = 0.1,
        random_state: Optional[int] = None
    ):
        super().__init__(
            C=C,
            kernel=kernel,
            degree=degree,
            gamma=gamma,
            coef0=coef0,
            tol=tol,
            max_iter=max_iter,
            random_state=random_state
        )
        self.epsilon = epsilon

        self.support_: Optional[np.array] = None
        self.support_vectors_: Optional[np.array] = None
        self.dual_coef_: Optional[np.array] = None
        self.intercept_: Optional[float] = None

    def fit(self, X: Union[mx.array, np.array], y: Union[mx.array, np.array]) -> "SVR":
        """
        Fit the SVR model according to the given training data.

        Parameters:
            X: Training data of shape (n_samples, n_features)
            y: Target values of shape (n_samples,)

        Returns:
            Fitted estimator
        """
        X, y = self._validate_data(X, y)
        X_np = np.array(X)
        y_np = np.array(y)

        # SVR implementation adapted from classification SVM
        # We transform SVR into binary classification problem for SMO
        n_samples = len(y_np)

        # Compute kernel matrix
        self.X_train_ = X_np
        K = self._kernel(X_np, X_np)

        # Use classification SMO with target values transformed to +1/-1 relative to margin
        # For simplicity, we reuse the classification SMO implementation by creating two classes
        # Upper margin: y + epsilon, class +1
        # Lower margin: y - epsilon, class -1
        # This is a simplified implementation for demonstration
        alpha = np.zeros(n_samples, dtype=np.float64)
        b = 0.0

        # First pass: fit to predict y directly using regression formulation
        # Use modified SMO for regression
        for iter in range(self.max_iter):
            alpha_changed = 0
            for i in range(n_samples):
                f_i = np.sum(alpha * K[:, i]) + b
                E_i = f_i - y_np[i]

                if abs(E_i) > self.epsilon:
                    # Select j
                    j = self.rng.choice([idx for idx in range(n_samples) if idx != i])
                    f_j = np.sum(alpha * K[:, j]) + b
                    E_j = f_j - y_np[j]

                    alpha_i_old = alpha[i].copy()
                    alpha_j_old = alpha[j].copy()

                    eta = 2 * K[i, j] - K[i, i] - K[j, j]
                    if eta >= 0:
                        continue

                    alpha[j] -= (E_i - E_j) / eta
                    alpha[j] = np.clip(alpha[j], -self.C, self.C)

                    if abs(alpha[j] - alpha_j_old) < 1e-5:
                        continue

                    alpha[i] += (E_i - E_j) * (alpha_j_old - alpha[j]) / eta
                    alpha[i] = np.clip(alpha[i], -self.C, self.C)

                    # Update b
                    b1 = b - E_i - (alpha[i] - alpha_i_old) * K[i,i] - (alpha[j] - alpha_j_old) * K[i,j]
                    b2 = b - E_j - (alpha[i] - alpha_i_old) * K[i,j] - (alpha[j] - alpha_j_old) * K[j,j]
                    b = (b1 + b2) / 2.0

                    alpha_changed += 1

            if alpha_changed == 0:
                break

        # Get support vectors
        sv_mask = abs(alpha) > 1e-5
        self.support_ = np.where(sv_mask)[0]
        self.support_vectors_ = X_np[sv_mask]
        self.dual_coef_ = alpha[sv_mask]

        # Compute intercept
        if len(self.support_) > 0:
            self.intercept_ = np.mean(y_np[sv_mask] - self._kernel(self.support_vectors_, self.support_vectors_) @ self.dual_coef_)
        else:
            self.intercept_ = np.mean(y_np)

        return self

    def predict(self, X: Union[mx.array, np.array]) -> mx.array:
        """
        Perform regression on samples in X.

        Parameters:
            X: Samples of shape (n_samples, n_features)

        Returns:
            Predicted values of shape (n_samples,)
        """
        if self.support_vectors_ is None:
            raise ValueError("Model not fitted yet. Call 'fit' first.")

        X, _ = self._validate_data(X)
        X_np = np.array(X)

        # Compute kernel between X and support vectors
        K = self._kernel(X_np, self.support_vectors_)

        # Compute predictions
        predictions = K @ self.dual_coef_ + self.intercept_

        return mx.array(predictions.astype(np.float32))
