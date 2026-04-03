import mlx.core as mx
import numpy as np
from typing import Optional, Union, List, Dict
from ..base.base_estimator import BaseEstimator


class LabelEncoder(BaseEstimator):
    """
    Encode target labels with value between 0 and n_classes-1.

    This transformer should be used to encode target values, i.e. y,
    and not the input X.
    """

    def __init__(self):
        super().__init__()
        self.classes_: Optional[mx.array] = None
        self.class_to_idx_: Optional[Dict[any, int]] = None

    def fit(self, y: Union[mx.array, np.array, List]) -> "LabelEncoder":
        """
        Fit label encoder.

        Parameters:
            y: Target values of shape (n_samples,)

        Returns:
            Fitted label encoder instance
        """
        if not isinstance(y, np.ndarray):
            y = np.array(y)

        # Get unique classes
        self.classes_ = np.unique(y)
        self.class_to_idx_ = {cls: idx for idx, cls in enumerate(self.classes_)}
        return self

    def transform(self, y: Union[mx.array, np.array, List]) -> mx.array:
        """
        Transform labels to normalized encoding.

        Parameters:
            y: Target values of shape (n_samples,)

        Returns:
            Encoded labels
        """
        if self.classes_ is None or self.class_to_idx_ is None:
            raise ValueError("LabelEncoder is not fitted yet. Call 'fit' first.")

        if not isinstance(y, np.ndarray):
            y = np.array(y)

        # Map values to indices
        encoded = np.array([self.class_to_idx_[val] for val in y])
        return mx.array(encoded, dtype=mx.int32)

    def fit_transform(self, y: Union[mx.array, np.array, List]) -> mx.array:
        """
        Fit label encoder and return encoded labels.

        Parameters:
            y: Target values of shape (n_samples,)

        Returns:
            Encoded labels
        """
        return self.fit(y).transform(y)

    def inverse_transform(self, y: Union[mx.array, np.array]) -> np.array:
        """
        Transform labels back to original encoding.

        Parameters:
            y: Encoded labels of shape (n_samples,)

        Returns:
            Original labels
        """
        if self.classes_ is None:
            raise ValueError("LabelEncoder is not fitted yet. Call 'fit' first.")

        if isinstance(y, mx.array):
            y = np.array(y)

        return self.classes_[y.astype(int)]


class OneHotEncoder(BaseEstimator):
    """
    Encode categorical features as a one-hot numeric array.

    The input to this transformer should be an array-like of integers or strings,
    denoting the values taken on by categorical (discrete) features.

    Parameters:
        categories: 'auto' or list of list/array, default='auto'
            Categories (unique values) per feature:
            - 'auto': Determine categories automatically from the training data
            - list: categories[i] holds the categories expected in the ith column
        drop: None, 'first', or array of shape (n_features,), default=None
            Specifies a methodology to use to drop one of the categories per feature
        sparse_output: Ignored in MLX implementation, always returns dense array
        dtype: Data type of output, default=float32
        handle_unknown: {'error', 'ignore'}, default='error'
            Whether to raise an error or ignore if an unknown category is present during transform
    """

    def __init__(self, categories: Union[str, List] = 'auto', drop: Optional[Union[str, List]] = None,
                 sparse_output: bool = False, dtype: Union[mx.Dtype, np.dtype] = mx.float32, handle_unknown: str = 'error'):
        super().__init__()
        self.categories = categories
        self.drop = drop
        self.sparse_output = sparse_output  # Ignored, always dense
        self.dtype = dtype
        self.handle_unknown = handle_unknown
        self.categories_: Optional[List[np.array]] = None
        self.drop_idx_: Optional[List[Optional[int]]] = None
        self.n_features_in_: Optional[int] = None

        # Convert mlx dtype to numpy dtype for internal use
        if isinstance(dtype, mx.Dtype):
            dtype_map = {
                mx.float32: np.float32,
                mx.float64: np.float64,
                mx.int32: np.int32,
                mx.int64: np.int64,
                mx.uint32: np.uint32,
                mx.uint64: np.uint64,
                mx.bool_: np.bool_
            }
            self._np_dtype = dtype_map.get(dtype, np.float32)
        else:
            self._np_dtype = dtype

    def fit(self, X: Union[mx.array, np.array], y: Optional[Union[mx.array, np.array]] = None) -> "OneHotEncoder":
        """
        Fit OneHotEncoder to X.

        Parameters:
            X: Data of shape (n_samples, n_features)
            y: Ignored, present for API compatibility

        Returns:
            Fitted encoder instance
        """
        X_np = np.array(X) if isinstance(X, mx.array) else X
        self.n_features_in_ = X_np.shape[1] if X_np.ndim > 1 else 1
        if X_np.ndim == 1:
            X_np = X_np.reshape(-1, 1)

        # Determine categories
        if self.categories == 'auto':
            self.categories_ = [np.unique(X_np[:, i]) for i in range(self.n_features_in_)]
        else:
            if len(self.categories) != self.n_features_in_:
                raise ValueError(f"Expected {self.n_features_in_} categories lists, got {len(self.categories)}")
            self.categories_ = [np.array(cats) for cats in self.categories]

        # Determine drop indices
        if self.drop is None:
            self.drop_idx_ = [None] * self.n_features_in_
        elif self.drop == 'first':
            self.drop_idx_ = [0] * self.n_features_in_
        else:
            if len(self.drop) != self.n_features_in_:
                raise ValueError(f"Expected {self.n_features_in_} drop values, got {len(self.drop)}")
            self.drop_idx_ = [
                np.where(cats == drop_val)[0][0] if drop_val is not None else None
                for cats, drop_val in zip(self.categories_, self.drop)
            ]

        return self

    def transform(self, X: Union[mx.array, np.array]) -> mx.array:
        """
        Transform X using one-hot encoding.

        Parameters:
            X: Data of shape (n_samples, n_features)

        Returns:
            Encoded data as dense array
        """
        if self.categories_ is None or self.drop_idx_ is None:
            raise ValueError("OneHotEncoder is not fitted yet. Call 'fit' first.")

        X_np = np.array(X) if isinstance(X, mx.array) else X
        if X_np.ndim == 1:
            X_np = X_np.reshape(-1, 1)

        n_samples = X_np.shape[0]
        if X_np.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {X_np.shape[1]}")

        # Calculate total output dimensions
        total_dims = 0
        feature_dims = []
        for i in range(self.n_features_in_):
            n_cats = len(self.categories_[i])
            if self.drop_idx_[i] is not None:
                n_cats -= 1
            feature_dims.append(n_cats)
            total_dims += n_cats

        # Initialize output
        output = np.zeros((n_samples, total_dims), dtype=self._np_dtype)
        col_idx = 0

        for feature_idx in range(self.n_features_in_):
            cats = self.categories_[feature_idx]
            cat_to_idx = {cat: idx for idx, cat in enumerate(cats)}
            drop_idx = self.drop_idx_[feature_idx]
            n_dims = feature_dims[feature_idx]

            for sample_idx in range(n_samples):
                val = X_np[sample_idx, feature_idx]
                if val not in cat_to_idx:
                    if self.handle_unknown == 'error':
                        raise ValueError(f"Unknown category '{val}' for feature {feature_idx}")
                    else:
                        # Ignore unknown, leave all zeros
                        continue

                idx = cat_to_idx[val]
                if drop_idx is not None:
                    if idx == drop_idx:
                        # Skip dropped category
                        continue
                    elif idx > drop_idx:
                        idx -= 1

                output[sample_idx, col_idx + idx] = 1.0

            col_idx += n_dims

        return mx.array(output)

    def fit_transform(self, X: Union[mx.array, np.array], y: Optional[Union[mx.array, np.array]] = None) -> mx.array:
        """
        Fit to data, then transform it.

        Parameters:
            X: Data of shape (n_samples, n_features)
            y: Ignored, present for API compatibility

        Returns:
            Encoded data
        """
        return self.fit(X).transform(X)

    def inverse_transform(self, X: Union[mx.array, np.array]) -> np.array:
        """
        Convert the encoded data back to original features.

        Parameters:
            X: Encoded data of shape (n_samples, n_encoded_features)

        Returns:
            Original categorical data
        """
        if self.categories_ is None or self.drop_idx_ is None:
            raise ValueError("OneHotEncoder is not fitted yet. Call 'fit' first.")

        X_np = np.array(X) if isinstance(X, mx.array) else X
        n_samples = X_np.shape[0]

        # Calculate feature dimensions
        feature_dims = []
        for i in range(self.n_features_in_):
            n_cats = len(self.categories_[i])
            if self.drop_idx_[i] is not None:
                n_cats -= 1
            feature_dims.append(n_cats)

        # Initialize output
        output = np.zeros((n_samples, self.n_features_in_), dtype=object)
        col_idx = 0

        for feature_idx in range(self.n_features_in_):
            cats = self.categories_[feature_idx]
            drop_idx = self.drop_idx_[feature_idx]
            n_dims = feature_dims[feature_idx]
            feature_data = X_np[:, col_idx:col_idx + n_dims]

            for sample_idx in range(n_samples):
                row = feature_data[sample_idx]
                if np.all(row == 0):
                    if drop_idx is not None:
                        # All zeros implies it's the dropped category
                        output[sample_idx, feature_idx] = cats[drop_idx]
                    else:
                        output[sample_idx, feature_idx] = None
                else:
                    idx = np.argmax(row)
                    if drop_idx is not None and idx >= drop_idx:
                        idx += 1
                    output[sample_idx, feature_idx] = cats[idx]

            col_idx += n_dims

        return output


class OrdinalEncoder(BaseEstimator):
    """
    Encode categorical features as an integer array.

    The input to this transformer should be an array-like of integers or strings,
    denoting the values taken on by categorical (discrete) features.

    Parameters:
        categories: 'auto' or list of list/array, default='auto'
            Categories (unique values) per feature
        dtype: Data type of output, default=float32
        handle_unknown: {'error', 'use_encoded_value'}, default='error'
            Whether to raise an error or use a encoded value if an unknown category is present
        unknown_value: int or None, default=None
            Value to use for unknown categories when handle_unknown='use_encoded_value'
    """

    def __init__(self, categories: Union[str, List] = 'auto', dtype: Union[mx.Dtype, np.dtype] = mx.float32,
                 handle_unknown: str = 'error', unknown_value: Optional[int] = None):
        super().__init__()
        self.categories = categories
        self.dtype = dtype
        self.handle_unknown = handle_unknown
        self.unknown_value = unknown_value
        self.categories_: Optional[List[np.array]] = None
        self.n_features_in_: Optional[int] = None

        if handle_unknown == 'use_encoded_value' and unknown_value is None:
            raise ValueError("unknown_value must be set when handle_unknown='use_encoded_value'")

        # Convert mlx dtype to numpy dtype for internal use
        if isinstance(dtype, mx.Dtype):
            dtype_map = {
                mx.float32: np.float32,
                mx.float64: np.float64,
                mx.int32: np.int32,
                mx.int64: np.int64,
                mx.uint32: np.uint32,
                mx.uint64: np.uint64,
                mx.bool_: np.bool_
            }
            self._np_dtype = dtype_map.get(dtype, np.float32)
        else:
            self._np_dtype = dtype

    def fit(self, X: Union[mx.array, np.array], y: Optional[Union[mx.array, np.array]] = None) -> "OrdinalEncoder":
        """
        Fit OrdinalEncoder to X.

        Parameters:
            X: Data of shape (n_samples, n_features)
            y: Ignored, present for API compatibility

        Returns:
            Fitted encoder instance
        """
        X_np = np.array(X) if isinstance(X, mx.array) else X
        self.n_features_in_ = X_np.shape[1] if X_np.ndim > 1 else 1
        if X_np.ndim == 1:
            X_np = X_np.reshape(-1, 1)

        # Determine categories
        if self.categories == 'auto':
            self.categories_ = [np.unique(X_np[:, i]) for i in range(self.n_features_in_)]
        else:
            if len(self.categories) != self.n_features_in_:
                raise ValueError(f"Expected {self.n_features_in_} categories lists, got {len(self.categories)}")
            self.categories_ = [np.array(cats) for cats in self.categories]

        return self

    def transform(self, X: Union[mx.array, np.array]) -> mx.array:
        """
        Transform X to ordinal codes.

        Parameters:
            X: Data of shape (n_samples, n_features)

        Returns:
            Encoded data
        """
        if self.categories_ is None:
            raise ValueError("OrdinalEncoder is not fitted yet. Call 'fit' first.")

        X_np = np.array(X) if isinstance(X, mx.array) else X
        if X_np.ndim == 1:
            X_np = X_np.reshape(-1, 1)

        n_samples = X_np.shape[0]
        if X_np.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {X_np.shape[1]}")

        # Initialize output
        output = np.zeros((n_samples, self.n_features_in_), dtype=self._np_dtype)

        for feature_idx in range(self.n_features_in_):
            cats = self.categories_[feature_idx]
            cat_to_idx = {cat: idx for idx, cat in enumerate(cats)}

            for sample_idx in range(n_samples):
                val = X_np[sample_idx, feature_idx]
                if val not in cat_to_idx:
                    if self.handle_unknown == 'error':
                        raise ValueError(f"Unknown category '{val}' for feature {feature_idx}")
                    else:
                        output[sample_idx, feature_idx] = self.unknown_value
                else:
                    output[sample_idx, feature_idx] = cat_to_idx[val]

        return mx.array(output)

    def fit_transform(self, X: Union[mx.array, np.array], y: Optional[Union[mx.array, np.array]] = None) -> mx.array:
        """
        Fit to data, then transform it.

        Parameters:
            X: Data of shape (n_samples, n_features)
            y: Ignored, present for API compatibility

        Returns:
            Encoded data
        """
        return self.fit(X).transform(X)

    def inverse_transform(self, X: Union[mx.array, np.array]) -> np.array:
        """
        Convert the encoded data back to original features.

        Parameters:
            X: Encoded data of shape (n_samples, n_features)

        Returns:
            Original categorical data
        """
        if self.categories_ is None:
            raise ValueError("OrdinalEncoder is not fitted yet. Call 'fit' first.")

        X_np = np.array(X) if isinstance(X, mx.array) else X
        if X_np.ndim == 1:
            X_np = X_np.reshape(-1, 1)

        n_samples = X_np.shape[0]
        if X_np.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {X_np.shape[1]}")

        # Initialize output
        output = np.zeros((n_samples, self.n_features_in_), dtype=object)

        for feature_idx in range(self.n_features_in_):
            cats = self.categories_[feature_idx]
            for sample_idx in range(n_samples):
                idx = int(X_np[sample_idx, feature_idx])
                if idx < 0 or idx >= len(cats):
                    output[sample_idx, feature_idx] = None
                else:
                    output[sample_idx, feature_idx] = cats[idx]

        return output
