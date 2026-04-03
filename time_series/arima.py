import mlx.core as mx
import numpy as np
from typing import Optional, Union, Tuple
from ..base import BaseEstimator
from ..stats import acf, pacf

class ARIMA(BaseEstimator):
    """
    Autoregressive Integrated Moving Average (ARIMA) model.

    Parameters:
        order: tuple (p, d, q)
            The (p,d,q) order of the model:
            - p: number of autoregressive (AR) parameters
            - d: number of non-seasonal differences
            - q: number of moving average (MA) parameters
            All values must be non-negative integers.
        seasonal_order: tuple (P, D, Q, s), default=None
            The (P,D,Q,s) order of the seasonal component:
            - P: number of seasonal AR parameters
            - D: number of seasonal differences
            - Q: number of seasonal MA parameters
            - s: number of time steps in a seasonal period
            All values must be non-negative integers.
        trend: {'n', 'c', 't', 'ct'}, default='c'
            Parameter controlling the deterministic trend:
            - 'n': no trend
            - 'c': constant term (default)
            - 't': linear trend
            - 'ct': constant + linear trend
    """
    def __init__(
        self,
        order: Tuple[int, int, int],
        seasonal_order: Optional[Tuple[int, int, int, int]] = None,
        trend: str = 'c'
    ):
        # 参数校验
        if len(order) != 3:
            raise ValueError("order must be a tuple of 3 non-negative integers (p, d, q)")
        if any(not isinstance(x, int) or x < 0 for x in order):
            raise ValueError("p, d, q must be non-negative integers")
        self.p, self.d, self.q = order

        if seasonal_order is not None:
            if len(seasonal_order) != 4:
                raise ValueError("seasonal_order must be a tuple of 4 non-negative integers (P, D, Q, s)")
            if any(not isinstance(x, int) or x < 0 for x in seasonal_order):
                raise ValueError("P, D, Q, s must be non-negative integers")
            if seasonal_order[3] < 2 and (seasonal_order[0] > 0 or seasonal_order[1] > 0 or seasonal_order[2] > 0):
                raise ValueError("seasonal period s must be >= 2 when seasonal terms are non-zero")
            self.P, self.D, self.Q, self.s = seasonal_order
        else:
            self.P = self.D = self.Q = self.s = 0

        if trend not in ['n', 'c', 't', 'ct']:
            raise ValueError("trend must be one of 'n', 'c', 't', 'ct'")
        self.trend = trend

        # 模型状态
        self.ar_params = None
        self.ma_params = None
        self.mu = None
        self.residuals = None
        self.fitted_values = None
        self.y_np = None
        self.n = None
        self.y_diff = None
        self.n_diff = None

    def fit(self, y: Union[mx.array, np.array]) -> 'ARIMA':
        """
        Fit the ARIMA model to the given time series.

        Parameters:
            y: Time series array. Must be 1-dimensional.

        Returns:
            self: Fitted model.
        """
        # 输入校验
        y_np = np.array(y)
        if y_np.ndim > 1 and not (y_np.ndim == 2 and min(y_np.shape) == 1):
            raise ValueError("y must be a 1-dimensional time series array")
        self.y_np = y_np.flatten()
        self.n = len(self.y_np)

        # 校验样本量
        min_obs = self.d + self.D * (self.s if self.s > 0 else 1) + max(self.p, self.q, self.P, self.Q) + 1
        if self.n < min_obs:
            raise ValueError(f"ARIMA({self.p},{self.d},{self.q}) requires at least {min_obs} observations, got {self.n}")
        if self.n < 2:
            raise ValueError("Time series must have at least 2 observations")

        # Step 1: Apply differencing
        self.y_diff = self._difference(self.y_np, self.d)
        if self.D > 0:
            self.y_diff = self._difference(self.y_diff, self.D, period=self.s)

        self.n_diff = len(self.y_diff)
        if self.n_diff < max(self.p, self.q) + 1:
            raise ValueError(f"After differencing, only {self.n_diff} observations remain, need at least {max(self.p, self.q) + 1}")

        # Step 2: Estimate AR parameters using Yule-Walker
        if self.p > 0:
            ar_params = self._estimate_ar_params(self.y_diff, self.p)
        else:
            ar_params = np.array([])

        # Step 3: Estimate MA parameters using Hannan-Rissanen
        if self.q > 0:
            ma_params = self._estimate_ma_params(self.y_diff, ar_params, self.q)
        else:
            ma_params = np.array([])

        # Step 4: Estimate trend parameter
        if self.trend == 'c' or self.trend == 'ct':
            self.mu = np.mean(self.y_diff)
        else:
            self.mu = 0.0

        self.ar_params = ar_params
        self.ma_params = ma_params

        # Compute fitted values and residuals
        self._compute_fitted_values()

        return self

    def _difference(self, y: np.array, d: int, period: int = 1) -> np.array:
        """Apply d-th order differencing"""
        for _ in range(d):
            y = y[period:] - y[:-period]
        return y

    def _inverse_difference(self, diff: np.array, original: np.array, d: int, period: int = 1) -> np.array:
        """Inverse of differencing operation"""
        result = diff.copy()
        for _ in range(d):
            last_vals = original[-period:]
            result = np.concatenate([last_vals, result])
            for i in range(period, len(result)):
                result[i] = result[i] + result[i - period]
            original = result
        return result

    def _estimate_ar_params(self, y: np.array, p: int) -> np.array:
        """Estimate AR parameters using Yule-Walker equations"""
        acf_vals = acf(y, nlags=p)
        R = np.zeros((p, p))
        for i in range(p):
            for j in range(p):
                R[i, j] = acf_vals[abs(i - j)]
        r = acf_vals[1:p+1]
        return np.linalg.solve(R, r)

    def _estimate_ma_params(self, y: np.array, ar_params: np.array, q: int) -> np.array:
        """Estimate MA parameters using Hannan-Rissanen algorithm"""
        n = len(y)
        max_ar = len(ar_params)
        max_lag = max(max_ar, q)

        # Compute residuals from AR model
        residuals = np.zeros(n)
        for t in range(max_ar, n):
            pred = np.sum(ar_params * y[t - max_ar : t][::-1])
            residuals[t] = y[t] - pred

        # Estimate MA parameters by regression
        X = np.zeros((n - max_lag, q))
        y_reg = residuals[max_lag:]
        for t in range(max_lag, n):
            X[t - max_lag] = residuals[t - q : t][::-1]

        if q > 0:
            ma_params = np.linalg.lstsq(X, y_reg, rcond=None)[0]
        else:
            ma_params = np.array([])

        return ma_params

    def _compute_fitted_values(self):
        """Compute fitted values and residuals"""
        n = self.n_diff
        fitted = np.zeros(n)
        residuals = np.zeros(n)

        max_lag = max(self.p, self.q)

        for t in range(max_lag, n):
            # AR part
            ar_part = 0.0
            if self.p > 0:
                ar_part = np.sum(self.ar_params * self.y_diff[t - self.p : t][::-1])

            # MA part
            ma_part = 0.0
            if self.q > 0:
                ma_part = np.sum(self.ma_params * residuals[t - self.q : t][::-1])

            fitted[t] = self.mu + ar_part + ma_part
            residuals[t] = self.y_diff[t] - fitted[t]

        # Inverse differencing to get fitted values on original scale
        self.fitted_values = self._inverse_difference(fitted, self.y_np, self.d)
        if self.D > 0:
            self.fitted_values = self._inverse_difference(self.fitted_values, self.y_np, self.D, period=self.s)

        self.residuals = self.y_np - self.fitted_values

    def predict(self, steps: int = 1) -> Union[mx.array, np.array]:
        """
        Predict future values.

        Parameters:
            steps: Number of steps to forecast. Must be >= 1.

        Returns:
            Forecasted values. 1D array of length steps.
        """
        # 校验参数
        if steps < 1:
            raise ValueError("steps must be >= 1")
        if self.ar_params is None or self.y_np is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")

        # Forecast on differenced series first
        forecast_diff = np.zeros(steps)
        last_values = self.y_diff[-max(self.p, 1):] if self.p > 0 else np.array([])
        last_residuals = self.residuals[-max(self.q, 1):] if self.q > 0 else np.array([])

        for h in range(steps):
            # AR part
            ar_part = 0.0
            if self.p > 0:
                ar_part = np.sum(self.ar_params * last_values[-self.p:][::-1])

            # MA part
            ma_part = 0.0
            if self.q > 0:
                # For future steps, residuals are zero
                if h < self.q:
                    ma_part = np.sum(self.ma_params[:self.q - h] * last_residuals[-self.q + h:][::-1])

            forecast_diff[h] = self.mu + ar_part + ma_part

            # Update last values for next step
            if self.p > 0:
                last_values = np.append(last_values, forecast_diff[h])

        # Inverse differencing to get forecast on original scale
        forecast = self._inverse_difference(forecast_diff, self.y_np, self.d)
        if self.D > 0:
            forecast = self._inverse_difference(forecast, self.y_np, self.D, period=self.s)

        # Return only the forecast steps
        return forecast[-steps:]

def arima(
    y: Union[mx.array, np.array],
    order: Tuple[int, int, int],
    steps: int = 1,
    trend: str = 'c'
) -> Union[mx.array, np.array]:
    """
    Simple ARIMA forecasting function.

    Parameters:
        y: Input time series.
        order: (p, d, q) order of the ARIMA model.
        steps: Number of steps to forecast.
        trend: Trend parameter.

    Returns:
        Forecasted values.
    """
    model = ARIMA(order=order, trend=trend)
    model.fit(y)
    return model.predict(steps=steps)
