import mlx.core as mx
import numpy as np
from typing import Optional, Union, Tuple
from ..base import BaseEstimator

class ExponentialSmoothing(BaseEstimator):
    """
    Holt-Winters Exponential Smoothing.

    Parameters:
        trend: {'add', 'mul', None}, default='add'
            Type of trend component. 'add' for additive trend, 'mul' for multiplicative trend.
        seasonal: {'add', 'mul', None}, default=None
            Type of seasonal component. 'add' for additive seasonality, 'mul' for multiplicative.
        seasonal_periods: int, default=None
            The number of time steps in a seasonal period. Required if seasonal is not None.
        initialization_method: {'estimated', 'heuristic', 'legacy-heuristic'}, default='heuristic'
            Method for initializing the level, trend, and seasonal components.
    """
    def __init__(
        self,
        trend: Optional[str] = 'add',
        seasonal: Optional[str] = None,
        seasonal_periods: Optional[int] = None,
        initialization_method: str = 'heuristic'
    ):
        # 参数校验
        if trend not in ['add', 'mul', None]:
            raise ValueError("trend must be one of 'add', 'mul', or None")
        if seasonal not in ['add', 'mul', None]:
            raise ValueError("seasonal must be one of 'add', 'mul', or None")
        if seasonal is not None and (seasonal_periods is None or seasonal_periods < 2):
            raise ValueError("seasonal_periods must be >= 2 when seasonal is specified")
        if initialization_method not in ['estimated', 'heuristic', 'legacy-heuristic']:
            raise ValueError("initialization_method must be 'estimated', 'heuristic', or 'legacy-heuristic'")

        self.trend = trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self.initialization_method = initialization_method
        self.alpha = None
        self.beta = None
        self.gamma = None
        self.init_level = None
        self.init_trend = None
        self.init_seasonal = None
        self.level = None
        self.trend_component = None
        self.seasonal_component = None
        self.n = None
        self.y_np = None

    def fit(
        self,
        y: Union[mx.array, np.array],
        smoothing_level: Optional[float] = None,
        smoothing_trend: Optional[float] = None,
        smoothing_seasonal: Optional[float] = None,
        optimized: bool = True
    ) -> 'ExponentialSmoothing':
        """
        Fit the model to the given time series.

        Parameters:
            y: Time series array. Must be 1-dimensional.
            smoothing_level: Alpha parameter for level smoothing (0 <= alpha <= 1).
            smoothing_trend: Beta parameter for trend smoothing (0 <= beta <= 1).
            smoothing_seasonal: Gamma parameter for seasonal smoothing (0 <= gamma <= 1).
            optimized: If True, optimize the smoothing parameters using grid search.

        Returns:
            self: Fitted model.
        """
        # 输入校验
        y_np = np.array(y)
        if y_np.ndim > 1 and not (y_np.ndim == 2 and min(y_np.shape) == 1):
            raise ValueError("y must be a 1-dimensional time series array")
        self.y_np = y_np.flatten()
        self.n = len(self.y_np)

        if self.n < 2:
            raise ValueError("Time series must have at least 2 observations")

        # 校验平滑参数
        def check_param(p, name):
            if p is not None and not (0 <= p <= 1):
                raise ValueError(f"{name} must be between 0 and 1")

        check_param(smoothing_level, "smoothing_level")
        check_param(smoothing_trend, "smoothing_trend")
        check_param(smoothing_seasonal, "smoothing_seasonal")

        # 校验季节性周期长度
        if self.seasonal is not None and self.n < 2 * self.seasonal_periods:
            raise ValueError(f"Time series must have at least 2 * seasonal_periods = {2 * self.seasonal_periods} observations for seasonal models")

        # Initialize components
        self._initialize_components()

        # If parameters are given, use them, else optimize
        required_params = [smoothing_level]
        if self.trend is not None:
            required_params.append(smoothing_trend)
        if self.seasonal is not None:
            required_params.append(smoothing_seasonal)

        if not optimized and all(p is not None for p in required_params):
            self.alpha = smoothing_level
            self.beta = smoothing_trend if self.trend is not None else None
            self.gamma = smoothing_seasonal if self.seasonal is not None else None
        else:
            # Simple grid search for parameters (placeholder for full optimization)
            best_mse = float('inf')
            best_params = None

            for alpha in [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]:
                for beta in ([0.001, 0.01, 0.05, 0.1, 0.2] if self.trend is not None else [None]):
                    for gamma in ([0.01, 0.05, 0.1, 0.2, 0.3] if self.seasonal is not None else [None]):
                        mse = self._compute_mse(alpha, beta, gamma)
                        if mse < best_mse:
                            best_mse = mse
                            best_params = (alpha, beta, gamma)

            self.alpha, self.beta, self.gamma = best_params

        # Compute final components with optimal parameters
        self._fit_components()

        return self

    def _initialize_components(self):
        """Initialize level, trend and seasonal components"""
        # Simple heuristic initialization
        if self.seasonal is not None:
            # Compute average of first m periods
            m = self.seasonal_periods
            # 取完整周期的数据
            n_full_periods = self.n // m
            y_full = self.y_np[:n_full_periods * m]
            period_means = np.array([np.mean(y_full[i*m : (i+1)*m]) for i in range(n_full_periods)])

            # Initial level
            self.init_level = period_means[0]

            # Initial trend
            if self.trend is not None and n_full_periods > 1:
                self.init_trend = (period_means[-1] - period_means[0]) / ((n_full_periods - 1) * m)
            else:
                self.init_trend = 0.0

            # Initial seasonal components
            if self.seasonal == 'add':
                # 计算每个季节位置的平均偏差
                seasonal = np.zeros(m)
                for i in range(m):
                    seasonal[i] = np.mean(y_full[i::m] - np.repeat(period_means, m)[i::m])
                # 季节性分量中心化，保证和为0
                seasonal -= seasonal.mean()
                self.init_seasonal = seasonal
            else: # mul
                seasonal = np.zeros(m)
                for i in range(m):
                    seasonal[i] = np.mean(y_full[i::m] / np.repeat(period_means, m)[i::m])
                # 季节性分量中心化，保证乘积为1
                seasonal /= seasonal.prod() ** (1/m)
                self.init_seasonal = seasonal
        else:
            # No seasonal component
            self.init_level = self.y_np[0]
            if self.trend is not None:
                self.init_trend = (self.y_np[-1] - self.y_np[0]) / (self.n - 1)
            else:
                self.init_trend = 0.0
            self.init_seasonal = None

    def _compute_mse(self, alpha: float, beta: Optional[float], gamma: Optional[float]) -> float:
        """Compute MSE for given parameters"""
        level = self.init_level
        trend = self.init_trend
        seasonal = self.init_seasonal.copy() if self.seasonal is not None else None

        errors = []
        m = self.seasonal_periods if self.seasonal is not None else 0

        for t in range(self.n):
            # Predict step
            if self.trend is None and self.seasonal is None:
                y_pred = level
            elif self.trend is not None and self.seasonal is None:
                y_pred = level + trend
            elif self.trend is None and self.seasonal is not None:
                y_pred = level + seasonal[t % m] if self.seasonal == 'add' else level * seasonal[t % m]
            else:
                if self.seasonal == 'add':
                    y_pred = level + trend + seasonal[t % m]
                else:
                    y_pred = (level + trend) * seasonal[t % m]

            errors.append(self.y_np[t] - y_pred)

            # Update step
            if self.seasonal is None:
                prev_level = level
                level = alpha * self.y_np[t] + (1 - alpha) * (prev_level + (trend if self.trend is not None else 0))
                if self.trend is not None:
                    trend = beta * (level - prev_level) + (1 - beta) * trend
            else:
                prev_level = level
                s_idx = t % m
                if self.seasonal == 'add':
                    level = alpha * (self.y_np[t] - seasonal[s_idx]) + (1 - alpha) * (prev_level + (trend if self.trend is not None else 0))
                else:
                    level = alpha * (self.y_np[t] / seasonal[s_idx]) + (1 - alpha) * (prev_level + (trend if self.trend is not None else 0))

                if self.trend is not None:
                    trend = beta * (level - prev_level) + (1 - beta) * trend

                if self.seasonal == 'add':
                    seasonal[s_idx] = gamma * (self.y_np[t] - level) + (1 - gamma) * seasonal[s_idx]
                else:
                    seasonal[s_idx] = gamma * (self.y_np[t] / level) + (1 - gamma) * seasonal[s_idx]

        return np.mean(np.array(errors) ** 2)

    def _fit_components(self):
        """Compute all components with optimal parameters"""
        self.level = np.zeros(self.n)
        self.level[0] = self.init_level

        if self.trend is not None:
            self.trend_component = np.zeros(self.n)
            self.trend_component[0] = self.init_trend

        if self.seasonal is not None:
            m = self.seasonal_periods
            self.seasonal_component = np.zeros(self.n + m)
            self.seasonal_component[:m] = self.init_seasonal

        for t in range(1, self.n):
            prev_level = self.level[t-1]
            prev_trend = self.trend_component[t-1] if self.trend is not None else 0
            s_idx = t % m if self.seasonal is not None else 0

            # Update level
            if self.seasonal is None:
                self.level[t] = self.alpha * self.y_np[t] + (1 - self.alpha) * (prev_level + prev_trend)
            else:
                if self.seasonal == 'add':
                    self.level[t] = self.alpha * (self.y_np[t] - self.seasonal_component[s_idx]) + (1 - self.alpha) * (prev_level + prev_trend)
                else:
                    self.level[t] = self.alpha * (self.y_np[t] / self.seasonal_component[s_idx]) + (1 - self.alpha) * (prev_level + prev_trend)

            # Update trend
            if self.trend is not None:
                self.trend_component[t] = self.beta * (self.level[t] - prev_level) + (1 - self.beta) * prev_trend

            # Update seasonal
            if self.seasonal is not None:
                if self.seasonal == 'add':
                    self.seasonal_component[t + m] = self.gamma * (self.y_np[t] - self.level[t]) + (1 - self.gamma) * self.seasonal_component[s_idx]
                else:
                    self.seasonal_component[t + m] = self.gamma * (self.y_np[t] / self.level[t]) + (1 - self.gamma) * self.seasonal_component[s_idx]

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
        if self.level is None or self.n is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")

        last_level = self.level[-1]
        last_trend = self.trend_component[-1] if self.trend is not None else 0.0

        forecast = np.zeros(steps)

        if self.seasonal is None:
            for h in range(1, steps + 1):
                if self.trend is None:
                    forecast[h-1] = last_level
                else:
                    forecast[h-1] = last_level + h * last_trend
        else:
            m = self.seasonal_periods
            last_seasonal = self.seasonal_component[self.n : self.n + m]
            for h in range(1, steps + 1):
                s_idx = (h - 1) % m
                if self.trend is None and self.seasonal == 'add':
                    forecast[h-1] = last_level + last_seasonal[s_idx]
                elif self.trend is None and self.seasonal == 'mul':
                    forecast[h-1] = last_level * last_seasonal[s_idx]
                elif self.trend is not None and self.seasonal == 'add':
                    forecast[h-1] = last_level + h * last_trend + last_seasonal[s_idx]
                else: # mul trend and seasonal
                    forecast[h-1] = (last_level + h * last_trend) * last_seasonal[s_idx]

        return forecast

def holt_winters(
    y: Union[mx.array, np.array],
    steps: int = 1,
    trend: Optional[str] = 'add',
    seasonal: Optional[str] = None,
    seasonal_periods: Optional[int] = None
) -> Union[mx.array, np.array]:
    """
    Simple Holt-Winters exponential smoothing forecasting function.

    Parameters:
        y: Input time series.
        steps: Number of steps to forecast.
        trend: Type of trend component.
        seasonal: Type of seasonal component.
        seasonal_periods: Number of time steps in a seasonal period.

    Returns:
        Forecasted values.
    """
    model = ExponentialSmoothing(trend=trend, seasonal=seasonal, seasonal_periods=seasonal_periods)
    model.fit(y)
    return model.predict(steps=steps)
