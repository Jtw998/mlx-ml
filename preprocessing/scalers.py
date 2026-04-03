import mlx.core as mx
import numpy as np
from typing import Optional, Union
from ..base.base_estimator import BaseEstimator


class StandardScaler(BaseEstimator):
    """
    Standardize features by removing the mean and scaling to unit variance.

    The standard score of a sample x is calculated as:
        z = (x - u) / s
    where u is the mean of the training samples, and s is the standard deviation of the training samples.

    Parameters:
        with_mean: If True, center the data before scaling
        with_std: If True, scale the data to unit variance
    """

    def __init__(self, with_mean: bool = True, with_std: bool = True):
        super().__init__()
        self.with_mean = with_mean
        self.with_std = with_std
        self.mean_: Optional[mx.array] = None
        self.scale_: Optional[mx.array] = None
        self.n_features_in_: Optional[int] = None

    def fit(self, X: Union[mx.array, np.array], y: Optional[Union[mx.array, np.array]] = None) -> "StandardScaler":
        """
        Compute the mean and std to be used for later scaling.

        Parameters:
            X: Training data of shape (n_samples, n_features)
            y: Ignored, present for API compatibility

        Returns:
            Fitted scaler instance
        """
        X, _ = self._validate_data(X)
        self.n_features_in_ = X.shape[1]

        if self.with_mean:
            self.mean_ = mx.mean(X, axis=0)
        else:
            self.mean_ = mx.zeros(self.n_features_in_, dtype=X.dtype)

        if self.with_std:
            self.scale_ = mx.std(X, axis=0, ddof=0)
            # Avoid division by zero for constant features
            self.scale_ = mx.where(self.scale_ == 0, 1.0, self.scale_)
        else:
            self.scale_ = mx.ones(self.n_features_in_, dtype=X.dtype)

        return self

    def transform(self, X: Union[mx.array, np.array]) -> mx.array:
        """
        Perform standardization by centering and scaling.

        Parameters:
            X: Data to transform of shape (n_samples, n_features)

        Returns:
            Transformed data
        """
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("StandardScaler is not fitted yet. Call 'fit' first.")

        X, _ = self._validate_data(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {X.shape[1]}")

        X_scaled = X
        if self.with_mean:
            X_scaled = X_scaled - self.mean_
        if self.with_std:
            X_scaled = X_scaled / self.scale_

        return X_scaled

    def inverse_transform(self, X: Union[mx.array, np.array]) -> mx.array:
        """
        Scale back the data to the original representation.

        Parameters:
            X: Data to inverse transform of shape (n_samples, n_features)

        Returns:
            Data in original scale
        """
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("StandardScaler is not fitted yet. Call 'fit' first.")

        X, _ = self._validate_data(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {X.shape[1]}")

        X_original = X * self.scale_ + self.mean_
        return X_original


class MinMaxScaler(BaseEstimator):
    """
    Transform features by scaling each feature to a given range.

    This estimator scales and translates each feature individually such
    that it is in the given range on the training set, e.g. between zero and one.

    The transformation is given by:
        X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
        X_scaled = X_std * (max - min) + min
    where min, max = feature_range.

    Parameters:
        feature_range: Desired range of transformed data, default (0, 1)
        clip: If True, clip values beyond the feature_range during transform
    """

    def __init__(self, feature_range: tuple = (0, 1), clip: bool = False):
        super().__init__()
        self.feature_range = feature_range
        self.clip = clip
        self.min_: Optional[mx.array] = None
        self.scale_: Optional[mx.array] = None
        self.data_min_: Optional[mx.array] = None
        self.data_max_: Optional[mx.array] = None
        self.data_range_: Optional[mx.array] = None
        self.n_features_in_: Optional[int] = None

    def fit(self, X: Union[mx.array, np.array], y: Optional[Union[mx.array, np.array]] = None) -> "MinMaxScaler":
        """
        Compute the minimum and maximum to be used for later scaling.

        Parameters:
            X: Training data of shape (n_samples, n_features)
            y: Ignored, present for API compatibility

        Returns:
            Fitted scaler instance
        """
        X, _ = self._validate_data(X)
        self.n_features_in_ = X.shape[1]
        min_range, max_range = self.feature_range

        self.data_min_ = mx.min(X, axis=0)
        self.data_max_ = mx.max(X, axis=0)
        self.data_range_ = self.data_max_ - self.data_min_

        # Handle constant features
        self.data_range_ = mx.where(self.data_range_ == 0, 1.0, self.data_range_)

        self.scale_ = (max_range - min_range) / self.data_range_
        self.min_ = min_range - self.data_min_ * self.scale_

        return self

    def transform(self, X: Union[mx.array, np.array]) -> mx.array:
        """
        Scale features according to feature_range.

        Parameters:
            X: Data to transform of shape (n_samples, n_features)

        Returns:
            Transformed data
        """
        if self.min_ is None or self.scale_ is None:
            raise ValueError("MinMaxScaler is not fitted yet. Call 'fit' first.")

        X, _ = self._validate_data(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {X.shape[1]}")

        X_scaled = X * self.scale_ + self.min_

        if self.clip:
            min_range, max_range = self.feature_range
            X_scaled = mx.clip(X_scaled, min_range, max_range)

        return X_scaled

    def inverse_transform(self, X: Union[mx.array, np.array]) -> mx.array:
        """
        Undo the scaling of X according to feature_range.

        Parameters:
            X: Data to inverse transform of shape (n_samples, n_features)

        Returns:
            Data in original scale
        """
        if self.min_ is None or self.scale_ is None:
            raise ValueError("MinMaxScaler is not fitted yet. Call 'fit' first.")

        X, _ = self._validate_data(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {X.shape[1]}")

        X_original = (X - self.min_) / self.scale_
        return X_original


class RobustScaler(BaseEstimator):
    """
    Scale features using statistics that are robust to outliers.

    This Scaler removes the median and scales the data according to
    the quantile range (defaults to IQR: Interquartile Range).
    The IQR is the range between the 1st quartile (25th quantile)
    and the 3rd quartile (75th quantile).

    Parameters:
        with_centering: If True, center the data before scaling
        with_scaling: If True, scale the data to interquartile range
        quantile_range: tuple (q_min, q_max), default (25.0, 75.0) = IQR
        unit_variance: If True, scale data to have unit variance
    """

    def __init__(self, with_centering: bool = True, with_scaling: bool = True,
                 quantile_range: tuple = (25.0, 75.0), unit_variance: bool = False):
        super().__init__()
        self.with_centering = with_centering
        self.with_scaling = with_scaling
        self.quantile_range = quantile_range
        self.unit_variance = unit_variance
        self.center_: Optional[mx.array] = None
        self.scale_: Optional[mx.array] = None
        self.n_features_in_: Optional[int] = None

    def fit(self, X: Union[mx.array, np.array], y: Optional[Union[mx.array, np.array]] = None) -> "RobustScaler":
        """
        Compute the median and quantiles to be used for later scaling.

        Parameters:
            X: Training data of shape (n_samples, n_features)
            y: Ignored, present for API compatibility

        Returns:
            Fitted scaler instance
        """
        X, _ = self._validate_data(X)
        self.n_features_in_ = X.shape[1]
        q_min, q_max = self.quantile_range

        if self.with_centering:
            # Compute median
            sorted_X = mx.sort(X, axis=0)
            n = X.shape[0]
            median_idx = n // 2
            if n % 2 == 1:
                self.center_ = sorted_X[median_idx]
            else:
                self.center_ = (sorted_X[median_idx - 1] + sorted_X[median_idx]) / 2
        else:
            self.center_ = mx.zeros(self.n_features_in_, dtype=X.dtype)

        if self.with_scaling:
            # Compute quantiles using numpy temporarily until MLX has quantile function
            X_np = np.array(X)
            q1 = np.percentile(X_np, q_min, axis=0)
            q3 = np.percentile(X_np, q_max, axis=0)
            iqr = q3 - q1

            # Handle constant features
            iqr = np.where(iqr == 0, 1.0, iqr)

            if self.unit_variance:
                # Scale to have unit variance (assuming normal distribution)
                iqr /= 1.34898

            self.scale_ = mx.array(iqr)
        else:
            self.scale_ = mx.ones(self.n_features_in_, dtype=X.dtype)

        return self

    def transform(self, X: Union[mx.array, np.array]) -> mx.array:
        """
        Center and scale the data.

        Parameters:
            X: Data to transform of shape (n_samples, n_features)

        Returns:
            Transformed data
        """
        if self.center_ is None or self.scale_ is None:
            raise ValueError("RobustScaler is not fitted yet. Call 'fit' first.")

        X, _ = self._validate_data(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {X.shape[1]}")

        X_scaled = X
        if self.with_centering:
            X_scaled = X_scaled - self.center_
        if self.with_scaling:
            X_scaled = X_scaled / self.scale_

        return X_scaled

    def inverse_transform(self, X: Union[mx.array, np.array]) -> mx.array:
        """
        Scale back the data to the original representation.

        Parameters:
            X: Data to inverse transform of shape (n_samples, n_features)

        Returns:
            Data in original scale
        """
        if self.center_ is None or self.scale_ is None:
            raise ValueError("RobustScaler is not fitted yet. Call 'fit' first.")

        X, _ = self._validate_data(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {X.shape[1]}")

        X_original = X * self.scale_ + self.center_
        return X_original
