import mlx.core as mx
import numpy as np
from typing import Optional, Union, Any
from ..base.base_estimator import BaseEstimator


class SimpleImputer(BaseEstimator):
    """
    Imputation transformer for completing missing values.

    Parameters:
        missing_values: The value to treat as missing, default=np.nan
        strategy: Imputation strategy:
            - 'mean': replace missing values using the mean along each column
            - 'median': replace missing values using the median along each column
            - 'most_frequent': replace missing values using the mode along each column
            - 'constant': replace missing values with fill_value
        fill_value: Value to use when strategy='constant', default=None
        add_indicator: If True, add a binary missing indicator feature, default=False
    """

    def __init__(self, missing_values: Any = np.nan, strategy: str = 'mean',
                 fill_value: Any = None, add_indicator: bool = False):
        super().__init__()
        self.missing_values = missing_values
        self.strategy = strategy
        self.fill_value = fill_value
        self.add_indicator = add_indicator
        self.statistics_: Optional[mx.array] = None
        self.indicator_mask_: Optional[np.array] = None
        self.n_features_in_: Optional[int] = None

        valid_strategies = ['mean', 'median', 'most_frequent', 'constant']
        if strategy not in valid_strategies:
            raise ValueError(f"Invalid strategy '{strategy}'. Valid options: {valid_strategies}")

        if strategy == 'constant' and fill_value is None:
            raise ValueError("fill_value must be specified when strategy='constant'")

    def fit(self, X: Union[mx.array, np.array], y: Optional[Union[mx.array, np.array]] = None) -> "SimpleImputer":
        """
        Fit the imputer on X.

        Parameters:
            X: Training data of shape (n_samples, n_features)
            y: Ignored, present for API compatibility

        Returns:
            Fitted imputer instance
        """
        X_np = np.array(X) if isinstance(X, mx.array) else X
        self.n_features_in_ = X_np.shape[1] if X_np.ndim > 1 else 1
        if X_np.ndim == 1:
            X_np = X_np.reshape(-1, 1)

        # Create mask for missing values
        if np.isnan(self.missing_values):
            mask = np.isnan(X_np)
        else:
            mask = (X_np == self.missing_values)

        self.indicator_mask_ = mask.any(axis=0) if self.add_indicator else None

        # Compute statistics
        stats = []
        for col in range(self.n_features_in_):
            col_data = X_np[:, col][~mask[:, col]]

            if len(col_data) == 0:
                if self.strategy == 'constant':
                    stat = self.fill_value
                else:
                    raise ValueError(f"Feature {col} has only missing values, cannot compute {self.strategy}")
            else:
                if self.strategy == 'mean':
                    stat = np.mean(col_data)
                elif self.strategy == 'median':
                    stat = np.median(col_data)
                elif self.strategy == 'most_frequent':
                    vals, counts = np.unique(col_data, return_counts=True)
                    stat = vals[np.argmax(counts)]
                elif self.strategy == 'constant':
                    stat = self.fill_value

            stats.append(stat)

        self.statistics_ = mx.array(stats)
        return self

    def transform(self, X: Union[mx.array, np.array]) -> mx.array:
        """
        Impute all missing values in X.

        Parameters:
            X: Data to impute of shape (n_samples, n_features)

        Returns:
            Imputed data with optional missing indicators
        """
        if self.statistics_ is None:
            raise ValueError("SimpleImputer is not fitted yet. Call 'fit' first.")

        X_np = np.array(X) if isinstance(X, mx.array) else X
        orig_ndim = X_np.ndim
        if orig_ndim == 1:
            X_np = X_np.reshape(-1, 1)

        n_samples = X_np.shape[0]
        if X_np.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {X_np.shape[1]}")

        # Create mask for missing values
        if np.isnan(self.missing_values):
            mask = np.isnan(X_np)
        else:
            mask = (X_np == self.missing_values)

        # Fill missing values
        X_imputed = X_np.copy()
        stats_np = np.array(self.statistics_)
        for col in range(self.n_features_in_):
            X_imputed[mask[:, col], col] = stats_np[col]

        # Add missing indicators if requested
        if self.add_indicator and self.indicator_mask_ is not None:
            indicator_cols = mask[:, self.indicator_mask_].astype(np.float32)
            X_imputed = np.hstack([X_imputed, indicator_cols])

        if orig_ndim == 1:
            X_imputed = X_imputed.ravel()

        return mx.array(X_imputed)

    def fit_transform(self, X: Union[mx.array, np.array], y: Optional[Union[mx.array, np.array]] = None) -> mx.array:
        """
        Fit the imputer on X and transform X.

        Parameters:
            X: Data to impute of shape (n_samples, n_features)
            y: Ignored, present for API compatibility

        Returns:
            Imputed data
        """
        return self.fit(X).transform(X)
