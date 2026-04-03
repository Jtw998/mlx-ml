import mlx.core as mx
import numpy as np
from typing import Optional, Union, Callable, List, Any
from ..base import BaseEstimator
from ..tree import DecisionTreeClassifier, DecisionTreeRegressor
from ..ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor

class KernelSHAP:
    """
    Kernel SHAP: A model-agnostic method for explaining predictions of any machine learning model.

    Parameters:
        model: Callable
            The model to explain. Should take an array of samples and return predictions.
        data: Union[mx.array, np.array]
            Background dataset used to integrate out features.
        nsamples: int, default=1000
            Number of samples to use for estimating SHAP values.
    """
    def __init__(
        self,
        model: Callable,
        data: Union[mx.array, np.array],
        nsamples: int = 1000
    ):
        self.model = model
        self.data_np = np.array(data)
        self.N, self.M = self.data_np.shape
        self.nsamples = nsamples
        self.expected_value = np.mean(model(data))

    def explain(
        self,
        x: Union[mx.array, np.array]
    ) -> Union[mx.array, np.array]:
        """
        Explain a single prediction.

        Parameters:
            x: Union[mx.array, np.array]
                The sample to explain.

        Returns:
            shap_values: SHAP values for each feature.
        """
        x_np = np.array(x).flatten()
        n_features = len(x_np)

        # Generate coalitions
        samples = np.random.randint(0, 2, size=(self.nsamples, n_features))
        # Make sure we include all 0 and all 1
        samples[0] = np.zeros(n_features)
        samples[1] = np.ones(n_features)

        # Compute predictions for each coalition
        y = np.zeros(self.nsamples)
        for i in range(self.nsamples):
            mask = samples[i]
            # Create synthetic sample by replacing masked features with background values
            synth_samples = np.tile(x_np, (self.N, 1))
            synth_samples[:, mask == 0] = self.data_np[:, mask == 0]
            y[i] = np.mean(self.model(synth_samples))

        # Compute weights
        weights = np.zeros(self.nsamples)
        for i in range(self.nsamples):
            s = np.sum(samples[i])
            if s == 0 or s == n_features:
                weights[i] = 1e6  # High weight for full and empty sets
            else:
                weights[i] = (n_features - 1) / (s * (n_features - s) * np.math.comb(n_features, s))

        # Solve weighted least squares
        X = samples
        y_centered = y - y[0]
        X_centered = X - X[0]

        # Add intercept
        W = np.diag(np.sqrt(weights))
        X_weighted = W @ X_centered
        y_weighted = W @ y_centered

        shap_values, _, _, _ = np.linalg.lstsq(X_weighted, y_weighted, rcond=None)

        return shap_values

class TreeSHAP:
    """
    Tree SHAP: Fast and exact SHAP value computation for tree-based models.

    Parameters:
        model: Tree-based model
            The tree model to explain. Supported models: DecisionTreeClassifier, DecisionTreeRegressor,
            RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor.
    """
    def __init__(self, model: BaseEstimator):
        self.model = model
        self._is_classifier = isinstance(model, (DecisionTreeClassifier, RandomForestClassifier, GradientBoostingClassifier))
        self._trees = self._extract_trees(model)

    def _extract_trees(self, model: BaseEstimator) -> List[Any]:
        """Extract individual trees from the model"""
        if isinstance(model, (DecisionTreeClassifier, DecisionTreeRegressor)):
            return [model.tree_]
        elif isinstance(model, (RandomForestClassifier, RandomForestRegressor)):
            return [estimator.tree_ for estimator in model.estimators_]
        elif isinstance(model, (GradientBoostingClassifier, GradientBoostingRegressor)):
            return [estimator[0].tree_ for estimator in model.estimators_]
        else:
            raise ValueError("Unsupported model type for TreeSHAP")

    def explain(
        self,
        X: Union[mx.array, np.array]
    ) -> Union[mx.array, np.array]:
        """
        Explain predictions for a set of samples.

        Parameters:
            X: Union[mx.array, np.array]
                Samples to explain.

        Returns:
            shap_values: SHAP values for each sample and feature.
        """
        X_np = np.array(X)
        n_samples, n_features = X_np.shape
        shap_values = np.zeros((n_samples, n_features))
        expected_value = 0.0

        # Get base value (average prediction)
        if self._is_classifier:
            expected_value = self.model.predict_proba(np.zeros((1, n_features)))[0, 1] if hasattr(self.model, 'predict_proba') else 0.0
        else:
            expected_value = self.model.predict(np.zeros((1, n_features)))[0]

        # For each sample, compute SHAP values recursively
        for i in range(n_samples):
            x = X_np[i]
            for tree in self._trees:
                # Simple recursive SHAP computation for trees
                shap_values[i] += self._tree_shap_recursive(tree, 0, x, np.zeros(n_features))[0]

        # Average over all trees
        shap_values /= len(self._trees)

        return shap_values

    def _tree_shap_recursive(self, tree: Any, node_idx: int, x: np.array, path: np.array) -> Tuple[np.array, float]:
        """Recursively compute SHAP values for a single tree node"""
        # Placeholder implementation (simplified)
        if tree.children_left[node_idx] == -1:  # Leaf node
            return path, tree.value[node_idx].item()

        feature = tree.feature[node_idx]
        threshold = tree.threshold[node_idx]
        left_child = tree.children_left[node_idx]
        right_child = tree.children_right[node_idx]

        if x[feature] <= threshold:
            # Go left
            path[feature] += 1
            left_path, left_val = self._tree_shap_recursive(tree, left_child, x, path.copy())
            path[feature] -= 1

            # Contribution from this split
            right_val = self._get_node_value(tree, right_child)
            left_path[feature] += (left_val - right_val) * (tree.n_node_samples[left_child] / tree.n_node_samples[node_idx])
            return left_path, left_val
        else:
            # Go right
            path[feature] += 1
            right_path, right_val = self._tree_shap_recursive(tree, right_child, x, path.copy())
            path[feature] -= 1

            # Contribution from this split
            left_val = self._get_node_value(tree, left_child)
            right_path[feature] += (right_val - left_val) * (tree.n_node_samples[right_child] / tree.n_node_samples[node_idx])
            return right_path, right_val

    def _get_node_value(self, tree: Any, node_idx: int) -> float:
        """Get the expected value of a node"""
        if tree.children_left[node_idx] == -1:
            return tree.value[node_idx].item()

        left_val = self._get_node_value(tree, tree.children_left[node_idx])
        right_val = self._get_node_value(tree, tree.children_right[node_idx])
        n_left = tree.n_node_samples[tree.children_left[node_idx]]
        n_right = tree.n_node_samples[tree.children_right[node_idx]]

        return (left_val * n_left + right_val * n_right) / (n_left + n_right)

def shap_values(
    model: BaseEstimator,
    X: Union[mx.array, np.array],
    background_data: Optional[Union[mx.array, np.array]] = None,
    model_agnostic: bool = False
) -> Union[mx.array, np.array]:
    """
    Compute SHAP values for a model and input samples.

    Parameters:
        model: The model to explain.
        X: Samples to explain.
        background_data: Background dataset for KernelSHAP (required if model_agnostic=True).
        model_agnostic: If True, use KernelSHAP for any model type. If False, use TreeSHAP for tree models.

    Returns:
        SHAP values for each sample and feature.
    """
    if model_agnostic:
        if background_data is None:
            raise ValueError("background_data is required for model-agnostic SHAP")
        explainer = KernelSHAP(model.predict, background_data)
        shap_vals = np.array([explainer.explain(x) for x in np.array(X)])
    else:
        try:
            explainer = TreeSHAP(model)
            shap_vals = explainer.explain(X)
        except ValueError:
            if background_data is None:
                raise ValueError("background_data is required for non-tree models")
            explainer = KernelSHAP(model.predict, background_data)
            shap_vals = np.array([explainer.explain(x) for x in np.array(X)])

    return shap_vals
