import mlx.core as mx
import numpy as np
from typing import Any, Dict, List, Optional, Union
import inspect


class BaseEstimator:
    """
    Base class for all estimators in mlx-ml.
    Follows scikit-learn API conventions for compatibility.
    """

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """
        Get parameters for this estimator.

        Parameters:
            deep: If True, will return the parameters for this estimator and contained subobjects that are estimators.

        Returns:
            Dictionary of parameter names mapped to their values.
        """
        params = {}
        for name in self._get_param_names():
            try:
                value = getattr(self, name)
            except AttributeError:
                continue

            if deep and hasattr(value, 'get_params'):
                deep_items = value.get_params().items()
                params.update((f"{name}__{k}", v) for k, v in deep_items)
            params[name] = value
        return params

    def set_params(self, **params: Dict[str, Any]) -> "BaseEstimator":
        """
        Set the parameters of this estimator.

        Parameters:
            **params: Estimator parameters.

        Returns:
            Estimator instance.
        """
        if not params:
            return self

        valid_params = self.get_params(deep=True)

        for key, value in params.items():
            if '__' not in key:
                if key not in valid_params:
                    raise ValueError(f"Invalid parameter {key} for estimator {self.__class__.__name__}. "
                                     f"Check the list of available parameters with `estimator.get_params().keys()`.")
                setattr(self, key, value)
            else:
                # Nested parameter
                parent_key, child_key = key.split('__', 1)
                if parent_key not in valid_params:
                    raise ValueError(f"Invalid parameter {parent_key} for estimator {self.__class__.__name__}.")
                parent = getattr(self, parent_key)
                if not hasattr(parent, 'set_params'):
                    raise ValueError(f"Parameter {parent_key} is of type {type(parent).__name__}, "
                                     "which does not implement set_params().")
                parent.set_params(**{child_key: value})

        return self

    def _get_param_names(self) -> List[str]:
        """
        Get parameter names for the estimator.

        Returns:
            List of parameter names.
        """
        init_signature = inspect.signature(self.__init__)
        return [p.name for p in init_signature.parameters.values() if p.name != 'self']

    def _validate_data(self, X: Union[mx.array, np.array], y: Optional[Union[mx.array, np.array]] = None,
                      ensure_2d: bool = True, allow_nd: bool = False) -> tuple:
        """
        Validate input data.

        Parameters:
            X: Input features
            y: Target values (optional)
            ensure_2d: If True, ensure X is 2D
            allow_nd: If True, allow X to have more than 2 dimensions

        Returns:
            Validated X and y as mlx arrays
        """
        # Convert to mlx array
        if not isinstance(X, mx.array):
            X = mx.array(X)

        # Validate X dimensions
        if X.ndim == 1 and ensure_2d:
            X = X.reshape(-1, 1)
        if X.ndim > 2 and not allow_nd:
            raise ValueError(f"Found array with dim {X.ndim}. Expected <= 2.")

        if y is not None:
            if not isinstance(y, mx.array):
                y = mx.array(y)
            if y.ndim == 2 and y.shape[1] == 1:
                y = y.ravel()
            if X.shape[0] != y.shape[0]:
                raise ValueError(f"X has {X.shape[0]} samples, y has {y.shape[0]} samples.")

        return X, y

    def fit(self, X: Union[mx.array, np.array], y: Optional[Union[mx.array, np.array]] = None, **kwargs) -> "BaseEstimator":
        """
        Fit the estimator to the training data.
        Should be implemented by subclasses.
        """
        raise NotImplementedError("fit() method not implemented for this estimator")

    def predict(self, X: Union[mx.array, np.array]) -> mx.array:
        """
        Predict target values for samples in X.
        Should be implemented by subclasses for estimators that support prediction.
        """
        raise NotImplementedError("predict() method not implemented for this estimator")

    def transform(self, X: Union[mx.array, np.array]) -> mx.array:
        """
        Transform X.
        Should be implemented by subclasses for estimators that support transformation.
        """
        raise NotImplementedError("transform() method not implemented for this estimator")

    def fit_transform(self, X: Union[mx.array, np.array], y: Optional[Union[mx.array, np.array]] = None, **kwargs) -> mx.array:
        """
        Fit to data, then transform it.
        """
        self.fit(X, y, **kwargs)
        return self.transform(X)

    def __repr__(self) -> str:
        """
        Return a string representation of the estimator.
        """
        params = self.get_params(deep=False)
        params_str = ", ".join([f"{k}={v}" for k, v in params.items() if not k.startswith('_')])
        return f"{self.__class__.__name__}({params_str})"
