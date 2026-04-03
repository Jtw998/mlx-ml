import mlx.core as mx
import numpy as np
from typing import Optional, Union, List, Tuple
from ..base.base_estimator import BaseEstimator
from ..tree.decision_tree import DecisionTreeRegressor


class GradientBoostingClassifier(BaseEstimator):
    """
    Gradient Boosting for classification.

    Parameters:
        n_estimators: The number of boosting stages to perform.
        learning_rate: Learning rate shrinks the contribution of each tree by learning_rate.
        max_depth: The maximum depth of the individual regression estimators.
        min_samples_split: The minimum number of samples required to split an internal node.
        min_samples_leaf: The minimum number of samples required to be at a leaf node.
        subsample: The fraction of samples to be used for fitting the individual base learners.
        max_features: The number of features to consider when looking for the best split.
        loss: The loss function to be optimized. 'deviance' refers to deviance (= logistic regression) for classification.
        random_state: Controls the randomness of the estimator.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        subsample: float = 1.0,
        max_features: Optional[Union[str, int, float]] = None,
        loss: str = 'deviance',
        random_state: Optional[int] = None
    ):
        super().__init__()
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.max_features = max_features
        self.loss = loss
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

        self.estimators_: List[DecisionTreeRegressor] = []
        self.classes_: Optional[np.array] = None
        self.n_classes_: Optional[int] = None
        self.n_features_in_: Optional[int] = None
        self.init_: Optional[float] = None

        valid_losses = ['deviance', 'log_loss']
        if loss not in valid_losses:
            raise ValueError(f"Invalid loss '{loss}'. Valid options: {valid_losses}")

    def _get_max_features(self, n_features: int) -> int:
        """Compute the number of features to consider for each split."""
        if self.max_features is None:
            return n_features
        elif isinstance(self.max_features, int):
            return min(self.max_features, n_features)
        elif isinstance(self.max_features, float):
            return int(self.max_features * n_features)
        elif self.max_features == 'sqrt':
            return int(np.sqrt(n_features))
        elif self.max_features == 'log2':
            return int(np.log2(n_features)) if n_features > 0 else 0
        else:
            raise ValueError(f"Invalid max_features value: {self.max_features}")

    def _sigmoid(self, x: np.array) -> np.array:
        """Compute sigmoid function."""
        return 1.0 / (1.0 + np.exp(-np.clip(x, -100, 100)))

    def _log_loss_gradient(self, y: np.array, y_pred: np.array) -> np.array:
        """Compute gradient of logistic loss."""
        y_proba = self._sigmoid(y_pred)
        return y - y_proba

    def fit(self, X: Union[mx.array, np.array], y: Union[mx.array, np.array]) -> "GradientBoostingClassifier":
        """
        Build a gradient boosting classifier from the training set.

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

        # Convert y to 0/1
        y_bin = (y_np == self.classes_[1]).astype(np.float64)
        self.n_features_in_ = X.shape[1]
        n_samples = X_np.shape[0]

        max_features = self._get_max_features(self.n_features_in_)

        # Initial prediction (log odds)
        pos = np.sum(y_bin)
        neg = n_samples - pos
        self.init_ = np.log(pos / neg) if pos > 0 and neg > 0 else 0.0
        y_pred = np.full(n_samples, self.init_, dtype=np.float64)

        self.estimators_ = []

        for i in range(self.n_estimators):
            # Compute negative gradient (pseudo-residuals)
            residuals = self._log_loss_gradient(y_bin, y_pred)

            # Subsample if needed
            if self.subsample < 1.0:
                sample_indices = self.rng.choice(n_samples, size=int(self.subsample * n_samples), replace=False)
                X_sample = X_np[sample_indices]
                residuals_sample = residuals[sample_indices]
            else:
                X_sample = X_np
                residuals_sample = residuals
                sample_indices = np.arange(n_samples)

            # Create and fit regression tree on residuals
            tree_random_state = self.rng.randint(0, np.iinfo(np.int32).max) if self.random_state is not None else None
            tree = DecisionTreeRegressor(
                criterion='squared_error',
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                random_state=tree_random_state
            )

            # Monkey patch for max_features
            original_best_split = tree._best_split

            def _patched_best_split(X_tree: np.array, y_tree: np.array) -> Tuple[int, float, float]:
                """Patched best split that only considers a random subset of features."""
                best_gain = -np.inf
                best_feature = None
                best_threshold = None
                current_error = tree._criterion(y_tree)
                n_samples_tree, n_features_tree = X_tree.shape

                feature_indices = self.rng.choice(n_features_tree, size=max_features, replace=False) if max_features < n_features_tree else np.arange(n_features_tree)

                for feature_idx in feature_indices:
                    feature_values = X_tree[:, feature_idx]
                    thresholds = np.unique(feature_values)

                    for threshold in thresholds:
                        left_mask = feature_values <= threshold
                        right_mask = ~left_mask

                        if np.sum(left_mask) < tree.min_samples_leaf or np.sum(right_mask) < tree.min_samples_leaf:
                            continue

                        left_error = tree._criterion(y_tree[left_mask])
                        right_error = tree._criterion(y_tree[right_mask])

                        n_left = np.sum(left_mask)
                        n_right = np.sum(right_mask)
                        weighted_error = (n_left / n_samples_tree) * left_error + (n_right / n_samples_tree) * right_error
                        gain = current_error - weighted_error

                        if gain > best_gain:
                            best_gain = gain
                            best_feature = feature_idx
                            best_threshold = threshold

                return best_feature, best_threshold, best_gain

            tree._best_split = _patched_best_split

            tree.fit(X_sample, residuals_sample)

            # Update predictions
            tree_pred = np.array(tree.predict(X_np))
            y_pred += self.learning_rate * tree_pred

            self.estimators_.append(tree)

        return self

    def decision_function(self, X: Union[mx.array, np.array]) -> mx.array:
        """
        Compute the decision function for X.

        Parameters:
            X: Samples of shape (n_samples, n_features)

        Returns:
            Decision function values of shape (n_samples,)
        """
        if not self.estimators_:
            raise ValueError("Model not fitted yet. Call 'fit' first.")

        X, _ = self._validate_data(X)
        X_np = np.array(X)

        y_pred = np.full(len(X_np), self.init_, dtype=np.float64)
        for tree in self.estimators_:
            y_pred += self.learning_rate * np.array(tree.predict(X_np))

        return mx.array(y_pred.astype(np.float32))

    def predict_proba(self, X: Union[mx.array, np.array]) -> mx.array:
        """
        Predict class probabilities for X.

        Parameters:
            X: Samples of shape (n_samples, n_features)

        Returns:
            Predicted class probabilities of shape (n_samples, n_classes)
        """
        decision = np.array(self.decision_function(X))
        proba_pos = self._sigmoid(decision)
        proba = np.column_stack((1 - proba_pos, proba_pos))
        return mx.array(proba.astype(np.float32))

    def predict(self, X: Union[mx.array, np.array]) -> mx.array:
        """
        Predict class for X.

        Parameters:
            X: Samples of shape (n_samples, n_features)

        Returns:
            Predicted classes of shape (n_samples,)
        """
        proba = np.array(self.predict_proba(X))
        predictions = self.classes_[np.argmax(proba, axis=1)]
        return mx.array(predictions.astype(np.int32))


class GradientBoostingRegressor(BaseEstimator):
    """
    Gradient Boosting for regression.

    Parameters:
        n_estimators: The number of boosting stages to perform.
        learning_rate: Learning rate shrinks the contribution of each tree by learning_rate.
        max_depth: The maximum depth of the individual regression estimators.
        min_samples_split: The minimum number of samples required to split an internal node.
        min_samples_leaf: The minimum number of samples required to be at a leaf node.
        subsample: The fraction of samples to be used for fitting the individual base learners.
        max_features: The number of features to consider when looking for the best split.
        loss: The loss function to be optimized.
        random_state: Controls the randomness of the estimator.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        subsample: float = 1.0,
        max_features: Optional[Union[str, int, float]] = None,
        loss: str = 'squared_error',
        random_state: Optional[int] = None
    ):
        super().__init__()
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.max_features = max_features
        self.loss = loss
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

        self.estimators_: List[DecisionTreeRegressor] = []
        self.n_features_in_: Optional[int] = None
        self.init_: Optional[float] = None

        valid_losses = ['squared_error', 'absolute_error', 'huber']
        if loss not in valid_losses:
            raise ValueError(f"Invalid loss '{loss}'. Valid options: {valid_losses}")

    def _get_max_features(self, n_features: int) -> int:
        """Compute the number of features to consider for each split."""
        if self.max_features is None:
            return n_features
        elif isinstance(self.max_features, int):
            return min(self.max_features, n_features)
        elif isinstance(self.max_features, float):
            return int(self.max_features * n_features)
        elif self.max_features == 'sqrt':
            return int(np.sqrt(n_features))
        elif self.max_features == 'log2':
            return int(np.log2(n_features)) if n_features > 0 else 0
        else:
            raise ValueError(f"Invalid max_features value: {self.max_features}")

    def _compute_gradient(self, y: np.array, y_pred: np.array) -> np.array:
        """Compute gradient of the loss function."""
        if self.loss == 'squared_error':
            return y - y_pred
        elif self.loss == 'absolute_error':
            return np.sign(y - y_pred)
        elif self.loss == 'huber':
            delta = 1.0
            diff = y - y_pred
            is_small = np.abs(diff) <= delta
            grad = np.zeros_like(diff)
            grad[is_small] = diff[is_small]
            grad[~is_small] = delta * np.sign(diff[~is_small])
            return grad
        else:
            raise ValueError(f"Unsupported loss: {self.loss}")

    def fit(self, X: Union[mx.array, np.array], y: Union[mx.array, np.array]) -> "GradientBoostingRegressor":
        """
        Build a gradient boosting regressor from the training set.

        Parameters:
            X: Training data of shape (n_samples, n_features)
            y: Target values of shape (n_samples,)

        Returns:
            Fitted estimator
        """
        X, y = self._validate_data(X, y)
        X_np = np.array(X)
        y_np = np.array(y)

        self.n_features_in_ = X.shape[1]
        n_samples = X_np.shape[0]

        max_features = self._get_max_features(self.n_features_in_)

        # Initial prediction
        if self.loss in ['squared_error', 'huber']:
            self.init_ = np.mean(y_np)
        else:  # absolute_error
            self.init_ = np.median(y_np)

        y_pred = np.full(n_samples, self.init_, dtype=np.float64)

        self.estimators_ = []

        for i in range(self.n_estimators):
            # Compute negative gradient (pseudo-residuals)
            residuals = self._compute_gradient(y_np, y_pred)

            # Subsample if needed
            if self.subsample < 1.0:
                sample_indices = self.rng.choice(n_samples, size=int(self.subsample * n_samples), replace=False)
                X_sample = X_np[sample_indices]
                residuals_sample = residuals[sample_indices]
            else:
                X_sample = X_np
                residuals_sample = residuals
                sample_indices = np.arange(n_samples)

            # Create and fit regression tree on residuals
            tree_random_state = self.rng.randint(0, np.iinfo(np.int32).max) if self.random_state is not None else None
            tree = DecisionTreeRegressor(
                criterion='squared_error',
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                random_state=tree_random_state
            )

            # Monkey patch for max_features
            original_best_split = tree._best_split

            def _patched_best_split(X_tree: np.array, y_tree: np.array) -> Tuple[int, float, float]:
                """Patched best split that only considers a random subset of features."""
                best_gain = -np.inf
                best_feature = None
                best_threshold = None
                current_error = tree._criterion(y_tree)
                n_samples_tree, n_features_tree = X_tree.shape

                feature_indices = self.rng.choice(n_features_tree, size=max_features, replace=False) if max_features < n_features_tree else np.arange(n_features_tree)

                for feature_idx in feature_indices:
                    feature_values = X_tree[:, feature_idx]
                    thresholds = np.unique(feature_values)

                    for threshold in thresholds:
                        left_mask = feature_values <= threshold
                        right_mask = ~left_mask

                        if np.sum(left_mask) < tree.min_samples_leaf or np.sum(right_mask) < tree.min_samples_leaf:
                            continue

                        left_error = tree._criterion(y_tree[left_mask])
                        right_error = tree._criterion(y_tree[right_mask])

                        n_left = np.sum(left_mask)
                        n_right = np.sum(right_mask)
                        weighted_error = (n_left / n_samples_tree) * left_error + (n_right / n_samples_tree) * right_error
                        gain = current_error - weighted_error

                        if gain > best_gain:
                            best_gain = gain
                            best_feature = feature_idx
                            best_threshold = threshold

                return best_feature, best_threshold, best_gain

            tree._best_split = _patched_best_split

            tree.fit(X_sample, residuals_sample)

            # Update predictions
            tree_pred = np.array(tree.predict(X_np))
            y_pred += self.learning_rate * tree_pred

            self.estimators_.append(tree)

        return self

    def predict(self, X: Union[mx.array, np.array]) -> mx.array:
        """
        Predict regression value for X.

        Parameters:
            X: Samples of shape (n_samples, n_features)

        Returns:
            Predicted values of shape (n_samples,)
        """
        if not self.estimators_:
            raise ValueError("Model not fitted yet. Call 'fit' first.")

        X, _ = self._validate_data(X)
        X_np = np.array(X)

        y_pred = np.full(len(X_np), self.init_, dtype=np.float64)
        for tree in self.estimators_:
            y_pred += self.learning_rate * np.array(tree.predict(X_np))

        return mx.array(y_pred.astype(np.float32))
