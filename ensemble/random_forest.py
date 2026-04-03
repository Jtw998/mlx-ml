import mlx.core as mx
import numpy as np
from typing import Optional, Union, List, Tuple
from ..base.base_estimator import BaseEstimator
from ..tree.decision_tree import DecisionTreeClassifier, DecisionTreeRegressor


class RandomForestClassifier(BaseEstimator):
    """
    A random forest classifier.

    Parameters:
        n_estimators: The number of trees in the forest.
        criterion: The function to measure the quality of a split.
            Supported criteria are "gini" for the Gini impurity and "entropy" for the information gain.
        max_depth: The maximum depth of the tree. If None, nodes are expanded until all leaves are pure.
        min_samples_split: The minimum number of samples required to split an internal node.
        min_samples_leaf: The minimum number of samples required to be at a leaf node.
        max_features: The number of features to consider when looking for the best split.
            If "sqrt", then max_features = sqrt(n_features).
            If "log2", then max_features = log2(n_features).
            If int, then consider max_features features at each split.
            If float, then max_features is a fraction and int(max_features * n_features) features are considered at each split.
            If None, then max_features = n_features.
        bootstrap: Whether bootstrap samples are used when building trees. If False, the whole dataset is used to build each tree.
        oob_score: Whether to use out-of-bag samples to estimate the generalization accuracy.
        n_jobs: The number of jobs to run in parallel for fit and predict. Currently not used, reserved for future implementation.
        random_state: Controls the randomness of the estimator.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        criterion: str = 'gini',
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Optional[Union[str, int, float]] = 'sqrt',
        bootstrap: bool = True,
        oob_score: bool = False,
        n_jobs: Optional[int] = None,
        random_state: Optional[int] = None
    ):
        super().__init__()
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

        self.estimators_: List[DecisionTreeClassifier] = []
        self.classes_: Optional[np.array] = None
        self.n_classes_: Optional[int] = None
        self.n_features_in_: Optional[int] = None
        self.oob_score_: Optional[float] = None
        self.oob_decision_function_: Optional[np.array] = None

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

    def fit(self, X: Union[mx.array, np.array], y: Union[mx.array, np.array]) -> "RandomForestClassifier":
        """
        Build a random forest classifier from the training set.

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
        n_samples = X_np.shape[0]

        max_features = self._get_max_features(self.n_features_in_)

        # Initialize estimators
        self.estimators_ = []
        oob_predictions = np.zeros((n_samples, self.n_classes_), dtype=np.float64)
        oob_counts = np.zeros(n_samples, dtype=np.int32)

        for i in range(self.n_estimators):
            # Create tree with random state
            tree_random_state = self.rng.randint(0, np.iinfo(np.int32).max) if self.random_state is not None else None
            tree = DecisionTreeClassifier(
                criterion=self.criterion,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                random_state=tree_random_state
            )

            # Bootstrap sample
            if self.bootstrap:
                indices = self.rng.choice(n_samples, size=n_samples, replace=True)
                X_sample = X_np[indices]
                y_sample = y_np[indices]

                # Track out-of-bag indices
                if self.oob_score:
                    oob_indices = np.setdiff1d(np.arange(n_samples), indices)
            else:
                X_sample = X_np
                y_sample = y_np
                oob_indices = np.array([], dtype=np.int32)

            # Monkey patch the tree's _best_split method to use max_features
            original_best_split = tree._best_split

            def _patched_best_split(X_tree: np.array, y_tree: np.array) -> Tuple[int, float, float]:
                """Patched best split that only considers a random subset of features."""
                best_gain = -np.inf
                best_feature = None
                best_threshold = None
                current_impurity = tree._criterion(y_tree)
                n_samples_tree, n_features_tree = X_tree.shape

                # Randomly select max_features features
                feature_indices = self.rng.choice(n_features_tree, size=max_features, replace=False) if max_features < n_features_tree else np.arange(n_features_tree)

                for feature_idx in feature_indices:
                    feature_values = X_tree[:, feature_idx]
                    thresholds = np.unique(feature_values)

                    for threshold in thresholds:
                        left_mask = feature_values <= threshold
                        right_mask = ~left_mask

                        if np.sum(left_mask) < tree.min_samples_leaf or np.sum(right_mask) < tree.min_samples_leaf:
                            continue

                        left_impurity = tree._criterion(y_tree[left_mask])
                        right_impurity = tree._criterion(y_tree[right_mask])

                        n_left = np.sum(left_mask)
                        n_right = np.sum(right_mask)
                        weighted_impurity = (n_left / n_samples_tree) * left_impurity + (n_right / n_samples_tree) * right_impurity
                        gain = current_impurity - weighted_impurity

                        if gain > best_gain:
                            best_gain = gain
                            best_feature = feature_idx
                            best_threshold = threshold

                return best_feature, best_threshold, best_gain

            tree._best_split = _patched_best_split

            # Fit the tree
            tree.fit(X_sample, y_sample)
            self.estimators_.append(tree)

            # Compute OOB predictions if needed
            if self.oob_score and len(oob_indices) > 0:
                oob_pred = np.array(tree.predict(X_np[oob_indices]))
                if len(oob_pred.shape) == 1:
                    # For classification, count votes
                    for j, pred in enumerate(oob_pred):
                        class_idx = np.where(self.classes_ == pred)[0][0]
                        oob_predictions[oob_indices[j], class_idx] += 1.0
                else:
                    oob_predictions[oob_indices] += np.array(oob_pred)
                oob_counts[oob_indices] += 1

        # Compute OOB score
        if self.oob_score:
            # Normalize OOB predictions
            mask = oob_counts > 0
            oob_decision = oob_predictions[mask] / oob_counts[mask, np.newaxis]
            oob_pred_classes = self.classes_[np.argmax(oob_decision, axis=1)]
            self.oob_score_ = np.mean(oob_pred_classes == y_np[mask])
            self.oob_decision_function_ = np.full((n_samples, self.n_classes_), np.nan)
            self.oob_decision_function_[mask] = oob_decision

        return self

    def predict(self, X: Union[mx.array, np.array]) -> mx.array:
        """
        Predict class for X.

        Parameters:
            X: Samples of shape (n_samples, n_features)

        Returns:
            Predicted classes of shape (n_samples,)
        """
        if not self.estimators_:
            raise ValueError("Model not fitted yet. Call 'fit' first.")

        X, _ = self._validate_data(X)
        X_np = np.array(X)

        # Collect predictions from all trees
        all_predictions = np.array([np.array(tree.predict(X_np)) for tree in self.estimators_])

        # Majority vote
        predictions = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=all_predictions)

        return mx.array(predictions.astype(np.int32))

    def predict_proba(self, X: Union[mx.array, np.array]) -> mx.array:
        """
        Predict class probabilities for X.

        Parameters:
            X: Samples of shape (n_samples, n_features)

        Returns:
            Predicted class probabilities of shape (n_samples, n_classes)
        """
        if not self.estimators_:
            raise ValueError("Model not fitted yet. Call 'fit' first.")

        X, _ = self._validate_data(X)
        X_np = np.array(X)

        # Collect probabilities from all trees
        all_probas = []
        for tree in self.estimators_:
            if hasattr(tree, 'predict_proba'):
                probas = np.array(tree.predict_proba(X_np))
            else:
                # Convert class predictions to one-hot probabilities
                preds = np.array(tree.predict(X_np))
                probas = np.zeros((len(preds), self.n_classes_))
                for i, pred in enumerate(preds):
                    class_idx = np.where(self.classes_ == pred)[0][0]
                    probas[i, class_idx] = 1.0
            all_probas.append(probas)

        # Average probabilities
        avg_probas = np.mean(all_probas, axis=0)

        return mx.array(avg_probas.astype(np.float32))


class RandomForestRegressor(BaseEstimator):
    """
    A random forest regressor.

    Parameters:
        n_estimators: The number of trees in the forest.
        criterion: The function to measure the quality of a split.
            Supported criteria are "squared_error" for the mean squared error and "absolute_error" for the mean absolute error.
        max_depth: The maximum depth of the tree. If None, nodes are expanded until all leaves are pure.
        min_samples_split: The minimum number of samples required to split an internal node.
        min_samples_leaf: The minimum number of samples required to be at a leaf node.
        max_features: The number of features to consider when looking for the best split.
            If "sqrt", then max_features = sqrt(n_features).
            If "log2", then max_features = log2(n_features).
            If int, then consider max_features features at each split.
            If float, then max_features is a fraction and int(max_features * n_features) features are considered at each split.
            If None, then max_features = n_features.
        bootstrap: Whether bootstrap samples are used when building trees. If False, the whole dataset is used to build each tree.
        oob_score: Whether to use out-of-bag samples to estimate the generalization score.
        n_jobs: The number of jobs to run in parallel for fit and predict. Currently not used, reserved for future implementation.
        random_state: Controls the randomness of the estimator.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        criterion: str = 'squared_error',
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Optional[Union[str, int, float]] = 1.0,
        bootstrap: bool = True,
        oob_score: bool = False,
        n_jobs: Optional[int] = None,
        random_state: Optional[int] = None
    ):
        super().__init__()
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

        self.estimators_: List[DecisionTreeRegressor] = []
        self.n_features_in_: Optional[int] = None
        self.oob_score_: Optional[float] = None
        self.oob_prediction_: Optional[np.array] = None

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

    def fit(self, X: Union[mx.array, np.array], y: Union[mx.array, np.array]) -> "RandomForestRegressor":
        """
        Build a random forest regressor from the training set.

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

        # Initialize estimators
        self.estimators_ = []
        oob_predictions = np.zeros(n_samples, dtype=np.float64)
        oob_counts = np.zeros(n_samples, dtype=np.int32)

        for i in range(self.n_estimators):
            # Create tree with random state
            tree_random_state = self.rng.randint(0, np.iinfo(np.int32).max) if self.random_state is not None else None
            tree = DecisionTreeRegressor(
                criterion=self.criterion,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                random_state=tree_random_state
            )

            # Bootstrap sample
            if self.bootstrap:
                indices = self.rng.choice(n_samples, size=n_samples, replace=True)
                X_sample = X_np[indices]
                y_sample = y_np[indices]

                # Track out-of-bag indices
                if self.oob_score:
                    oob_indices = np.setdiff1d(np.arange(n_samples), indices)
            else:
                X_sample = X_np
                y_sample = y_np
                oob_indices = np.array([], dtype=np.int32)

            # Monkey patch the tree's _best_split method to use max_features
            original_best_split = tree._best_split

            def _patched_best_split(X_tree: np.array, y_tree: np.array) -> Tuple[int, float, float]:
                """Patched best split that only considers a random subset of features."""
                best_gain = -np.inf
                best_feature = None
                best_threshold = None
                current_error = tree._criterion(y_tree)
                n_samples_tree, n_features_tree = X_tree.shape

                # Randomly select max_features features
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

            # Fit the tree
            tree.fit(X_sample, y_sample)
            self.estimators_.append(tree)

            # Compute OOB predictions if needed
            if self.oob_score and len(oob_indices) > 0:
                oob_pred = np.array(tree.predict(X_np[oob_indices]))
                oob_predictions[oob_indices] += oob_pred
                oob_counts[oob_indices] += 1

        # Compute OOB score
        if self.oob_score:
            mask = oob_counts > 0
            oob_avg = oob_predictions[mask] / oob_counts[mask]
            if self.criterion == 'squared_error':
                self.oob_score_ = 1 - np.sum((y_np[mask] - oob_avg) ** 2) / np.sum((y_np[mask] - np.mean(y_np[mask])) ** 2)
            else:
                self.oob_score_ = 1 - np.sum(np.abs(y_np[mask] - oob_avg)) / np.sum(np.abs(y_np[mask] - np.median(y_np[mask])))
            self.oob_prediction_ = np.full(n_samples, np.nan)
            self.oob_prediction_[mask] = oob_avg

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

        # Collect predictions from all trees
        all_predictions = np.array([np.array(tree.predict(X_np)) for tree in self.estimators_])

        # Average predictions
        predictions = np.mean(all_predictions, axis=0)

        return mx.array(predictions.astype(np.float32))
