import mlx.core as mx
import numpy as np
from typing import Optional, Union, List, Tuple
from ..base.base_estimator import BaseEstimator


class Node:
    """
    Decision tree node structure.
    """
    def __init__(self):
        self.feature_idx: Optional[int] = None
        self.threshold: Optional[float] = None
        self.left: Optional[Node] = None
        self.right: Optional[Node] = None
        self.value: Optional[Union[float, int, mx.array]] = None
        self.is_leaf: bool = False


class DecisionTreeClassifier(BaseEstimator):
    """
    A decision tree classifier.

    Parameters:
        criterion: The function to measure the quality of a split.
            Supported criteria are "gini" for the Gini impurity and "entropy" for the information gain.
        max_depth: The maximum depth of the tree. If None, nodes are expanded until all leaves are pure.
        min_samples_split: The minimum number of samples required to split an internal node.
        min_samples_leaf: The minimum number of samples required to be at a leaf node.
        random_state: Controls the randomness of the estimator.
    """

    def __init__(
        self,
        criterion: str = 'gini',
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        random_state: Optional[int] = None
    ):
        super().__init__()
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

        self.root: Optional[Node] = None
        self.classes_: Optional[np.array] = None
        self.n_classes_: Optional[int] = None
        self.n_features_in_: Optional[int] = None

        valid_criteria = ['gini', 'entropy']
        if criterion not in valid_criteria:
            raise ValueError(f"Invalid criterion '{criterion}'. Valid options: {valid_criteria}")

    def _gini(self, y: np.array) -> float:
        """Compute Gini impurity."""
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return 1.0 - np.sum(probabilities ** 2)

    def _entropy(self, y: np.array) -> float:
        """Compute entropy."""
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return -np.sum(probabilities * np.log2(probabilities + 1e-10))

    def _criterion(self, y: np.array) -> float:
        """Compute the selected criterion."""
        if self.criterion == 'gini':
            return self._gini(y)
        else:  # entropy
            return self._entropy(y)

    def _best_split(self, X: np.array, y: np.array) -> Tuple[int, float, float]:
        """Find the best split feature and threshold."""
        best_gain = -np.inf
        best_feature = None
        best_threshold = None
        current_impurity = self._criterion(y)
        n_samples, n_features = X.shape

        for feature_idx in range(n_features):
            feature_values = X[:, feature_idx]
            thresholds = np.unique(feature_values)

            for threshold in thresholds:
                left_mask = feature_values <= threshold
                right_mask = ~left_mask

                if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
                    continue

                left_impurity = self._criterion(y[left_mask])
                right_impurity = self._criterion(y[right_mask])

                # Compute information gain
                n_left = np.sum(left_mask)
                n_right = np.sum(right_mask)
                weighted_impurity = (n_left / n_samples) * left_impurity + (n_right / n_samples) * right_impurity
                gain = current_impurity - weighted_impurity

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold

        return best_feature, best_threshold, best_gain

    def _build_tree(self, X: np.array, y: np.array, depth: int = 0) -> Node:
        """Recursively build the decision tree."""
        node = Node()
        n_samples = len(y)
        unique_classes, counts = np.unique(y, return_counts=True)

        # Check leaf node conditions
        if (self.max_depth is not None and depth >= self.max_depth) or \
           (n_samples < self.min_samples_split) or \
           (len(unique_classes) == 1) or \
           (n_samples < 2 * self.min_samples_leaf):
            node.is_leaf = True
            # Majority class
            node.value = unique_classes[np.argmax(counts)]
            return node

        # Find best split
        feature_idx, threshold, gain = self._best_split(X, y)

        # If no good split found, make leaf node
        if feature_idx is None or gain <= 0:
            node.is_leaf = True
            node.value = unique_classes[np.argmax(counts)]
            return node

        # Split data
        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask

        # Recursively build children
        node.feature_idx = feature_idx
        node.threshold = threshold
        node.left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return node

    def _predict_sample(self, x: np.array, node: Node) -> Union[int, float]:
        """Predict a single sample."""
        if node.is_leaf:
            return node.value

        if x[node.feature_idx] <= node.threshold:
            return self._predict_sample(x, node.left)
        else:
            return self._predict_sample(x, node.right)

    def fit(self, X: Union[mx.array, np.array], y: Union[mx.array, np.array]) -> "DecisionTreeClassifier":
        """
        Build a decision tree classifier from the training set.

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
        self.n_features_in_ = X.shape[1]

        self.root = self._build_tree(X_np, y_np)
        return self

    def predict(self, X: Union[mx.array, np.array]) -> mx.array:
        """
        Predict class for X.

        Parameters:
            X: Samples of shape (n_samples, n_features)

        Returns:
            Predicted classes of shape (n_samples,)
        """
        if self.root is None:
            raise ValueError("Model not fitted yet. Call 'fit' first.")

        X, _ = self._validate_data(X)
        X_np = np.array(X)

        predictions = [self._predict_sample(x, self.root) for x in X_np]
        return mx.array(np.array(predictions, dtype=np.int32))

    def predict_proba(self, X: Union[mx.array, np.array]) -> mx.array:
        """
        Predict class probabilities for X.

        Parameters:
            X: Samples of shape (n_samples, n_features)

        Returns:
            Predicted class probabilities of shape (n_samples, n_classes)
        """
        if self.root is None:
            raise ValueError("Model not fitted yet. Call 'fit' first.")

        X, _ = self._validate_data(X)
        X_np = np.array(X)

        probas = np.zeros((len(X_np), self.n_classes_), dtype=np.float32)
        for i, x in enumerate(X_np):
            pred = self._predict_sample(x, self.root)
            class_idx = np.where(self.classes_ == pred)[0][0]
            probas[i, class_idx] = 1.0

        return mx.array(probas)


class DecisionTreeRegressor(BaseEstimator):
    """
    A decision tree regressor.

    Parameters:
        criterion: The function to measure the quality of a split.
            Supported criteria are "squared_error" for the mean squared error and "absolute_error" for the mean absolute error.
        max_depth: The maximum depth of the tree. If None, nodes are expanded until all leaves are pure.
        min_samples_split: The minimum number of samples required to split an internal node.
        min_samples_leaf: The minimum number of samples required to be at a leaf node.
        random_state: Controls the randomness of the estimator.
    """

    def __init__(
        self,
        criterion: str = 'squared_error',
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        random_state: Optional[int] = None
    ):
        super().__init__()
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

        self.root: Optional[Node] = None
        self.n_features_in_: Optional[int] = None

        valid_criteria = ['squared_error', 'absolute_error']
        if criterion not in valid_criteria:
            raise ValueError(f"Invalid criterion '{criterion}'. Valid options: {valid_criteria}")

    def _criterion(self, y: np.array) -> float:
        """Compute the selected criterion."""
        if self.criterion == 'squared_error':
            return np.mean((y - np.mean(y)) ** 2)
        else:  # absolute_error
            return np.mean(np.abs(y - np.median(y)))

    def _best_split(self, X: np.array, y: np.array) -> Tuple[int, float, float]:
        """Find the best split feature and threshold."""
        best_gain = -np.inf
        best_feature = None
        best_threshold = None
        current_error = self._criterion(y)
        n_samples, n_features = X.shape

        for feature_idx in range(n_features):
            feature_values = X[:, feature_idx]
            thresholds = np.unique(feature_values)

            for threshold in thresholds:
                left_mask = feature_values <= threshold
                right_mask = ~left_mask

                if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
                    continue

                left_error = self._criterion(y[left_mask])
                right_error = self._criterion(y[right_mask])

                # Compute error reduction
                n_left = np.sum(left_mask)
                n_right = np.sum(right_mask)
                weighted_error = (n_left / n_samples) * left_error + (n_right / n_samples) * right_error
                gain = current_error - weighted_error

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold

        return best_feature, best_threshold, best_gain

    def _build_tree(self, X: np.array, y: np.array, depth: int = 0) -> Node:
        """Recursively build the decision tree."""
        node = Node()
        n_samples = len(y)

        # Check leaf node conditions
        if (self.max_depth is not None and depth >= self.max_depth) or \
           (n_samples < self.min_samples_split) or \
           (n_samples < 2 * self.min_samples_leaf):
            node.is_leaf = True
            # Mean or median of the node
            if self.criterion == 'squared_error':
                node.value = np.mean(y)
            else:
                node.value = np.median(y)
            return node

        # Find best split
        feature_idx, threshold, gain = self._best_split(X, y)

        # If no good split found, make leaf node
        if feature_idx is None or gain <= 0:
            node.is_leaf = True
            if self.criterion == 'squared_error':
                node.value = np.mean(y)
            else:
                node.value = np.median(y)
            return node

        # Split data
        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask

        # Recursively build children
        node.feature_idx = feature_idx
        node.threshold = threshold
        node.left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return node

    def _predict_sample(self, x: np.array, node: Node) -> float:
        """Predict a single sample."""
        if node.is_leaf:
            return node.value

        if x[node.feature_idx] <= node.threshold:
            return self._predict_sample(x, node.left)
        else:
            return self._predict_sample(x, node.right)

    def fit(self, X: Union[mx.array, np.array], y: Union[mx.array, np.array]) -> "DecisionTreeRegressor":
        """
        Build a decision tree regressor from the training set.

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
        self.root = self._build_tree(X_np, y_np)
        return self

    def predict(self, X: Union[mx.array, np.array]) -> mx.array:
        """
        Predict regression value for X.

        Parameters:
            X: Samples of shape (n_samples, n_features)

        Returns:
            Predicted values of shape (n_samples,)
        """
        if self.root is None:
            raise ValueError("Model not fitted yet. Call 'fit' first.")

        X, _ = self._validate_data(X)
        X_np = np.array(X)

        predictions = [self._predict_sample(x, self.root) for x in X_np]
        return mx.array(np.array(predictions, dtype=np.float32))
