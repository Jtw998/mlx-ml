import mlx.core as mx
import numpy as np
from typing import Optional, Union, Tuple

def _to_mx_array(x: Union[mx.array, np.array]) -> mx.array:
    """Convert input to MLX array if it's numpy array"""
    return mx.array(x) if isinstance(x, np.ndarray) else x

# ===================== 描述性统计 =====================
def quantile(
    a: Union[mx.array, np.array],
    q: Union[float, list, np.array, mx.array],
    axis: Optional[int] = None,
    keepdims: bool = False
) -> Union[float, mx.array, np.array]:
    """
    Compute the q-th quantile of the data along the specified axis.

    Parameters:
        a: Input array.
        q: Quantile or sequence of quantiles to compute, between 0 and 1.
        axis: Axis along which to compute quantiles.
        keepdims: If True, the output has the same number of dimensions as input.

    Returns:
        Quantile(s) computed.
    """
    # 用numpy实现，因为mlx没有内置quantile
    a_np = np.array(a)
    result = np.quantile(a_np, q, axis=axis, keepdims=keepdims)

    if isinstance(a, np.ndarray):
        if result.size == 1 and result.ndim == 0:
            return float(result)
        return result
    return float(result) if result.size == 1 and result.ndim == 0 else mx.array(result)

def percentile(
    a: Union[mx.array, np.array],
    q: Union[float, list, np.array, mx.array],
    axis: Optional[int] = None,
    keepdims: bool = False
) -> Union[float, mx.array, np.array]:
    """
    Compute the q-th percentile of the data along the specified axis.

    Parameters:
        a: Input array.
        q: Percentile or sequence of percentiles to compute, between 0 and 100.
        axis: Axis along which to compute percentiles.
        keepdims: If True, the output has the same number of dimensions as input.

    Returns:
        Percentile(s) computed.
    """
    # 转换q为numpy数组处理
    if isinstance(q, (list, np.ndarray, mx.array)):
        q_np = np.array(q) / 100
    else:
        q_np = q / 100
    return quantile(a, q_np, axis=axis, keepdims=keepdims)

def mode(
    a: Union[mx.array, np.array],
    axis: Optional[int] = None,
    keepdims: bool = False
) -> Tuple[Union[float, mx.array, np.array], Union[int, mx.array, np.array]]:
    """
    Compute the mode (most frequent value) of the data along the specified axis.

    Parameters:
        a: Input array.
        axis: Axis along which to compute mode.
        keepdims: If True, the output has the same number of dimensions as input.

    Returns:
        mode: The modal value(s).
        count: The count of the modal value(s).
    """
    a_np = np.array(a)
    if axis is None:
        a_np = a_np.flatten()
        axis = 0

    vals, counts = np.unique(a_np, return_counts=True)
    mode_idx = np.argmax(counts)
    mode_val = vals[mode_idx]
    mode_count = counts[mode_idx]

    if isinstance(a, np.ndarray):
        return mode_val, mode_count
    return mx.array(mode_val), mx.array(mode_count)

def skew(
    a: Union[mx.array, np.array],
    axis: Optional[int] = 0,
    bias: bool = True
) -> Union[float, mx.array, np.array]:
    """
    Compute the skewness of a data set.

    Parameters:
        a: Input array.
        axis: Axis along which to compute skewness.
        bias: If False, calculations are corrected for statistical bias.

    Returns:
        Skewness value(s).
    """
    a_mx = _to_mx_array(a)
    mean = mx.mean(a_mx, axis=axis, keepdims=True)
    std = mx.std(a_mx, axis=axis, keepdims=True, ddof=0 if bias else 1)

    # Avoid division by zero
    std = mx.where(std == 0, 1.0, std)

    # Compute skewness
    m3 = mx.mean(((a_mx - mean) / std) ** 3, axis=axis)

    if not bias:
        n = a_mx.shape[axis] if axis is not None else a_mx.size
        m3 = m3 * (n ** 2) / ((n - 1) * (n - 2)) if n > 2 else m3

    if isinstance(a, np.ndarray):
        return float(m3) if m3.size == 1 and m3.ndim == 0 else np.array(m3)
    return float(m3) if m3.size == 1 and m3.ndim == 0 else m3

def kurtosis(
    a: Union[mx.array, np.array],
    axis: Optional[int] = 0,
    fisher: bool = True,
    bias: bool = True
) -> Union[float, mx.array, np.array]:
    """
    Compute the kurtosis of a data set.

    Parameters:
        a: Input array.
        axis: Axis along which to compute kurtosis.
        fisher: If True, Fisher's definition is used (normal ==> 0.0). If False, Pearson's definition is used (normal ==> 3.0).
        bias: If False, calculations are corrected for statistical bias.

    Returns:
        Kurtosis value(s).
    """
    a_mx = _to_mx_array(a)
    mean = mx.mean(a_mx, axis=axis, keepdims=True)
    std = mx.std(a_mx, axis=axis, keepdims=True, ddof=0 if bias else 1)

    # Avoid division by zero
    std = mx.where(std == 0, 1.0, std)

    # Compute kurtosis
    m4 = mx.mean(((a_mx - mean) / std) ** 4, axis=axis)

    if not bias:
        n = a_mx.shape[axis] if axis is not None else a_mx.size
        if n > 3:
            m4 = ((n + 1) * (n ** 2) * m4 - 3 * (n - 1) ** 3) / ((n - 2) * (n - 3))
            if fisher:
                m4 = m4 - 3 * ((n - 1) ** 2) / ((n - 2) * (n - 3))
        else:
            if fisher:
                m4 = m4 - 3.0
    else:
        if fisher:
            m4 = m4 - 3.0

    if isinstance(a, np.ndarray):
        return float(m4) if m4.size == 1 and m4.ndim == 0 else np.array(m4)
    return float(m4) if m4.size == 1 and m4.ndim == 0 else m4

# ===================== 相关性计算 =====================
def cov(
    m: Union[mx.array, np.array],
    y: Optional[Union[mx.array, np.array]] = None,
    rowvar: bool = True,
    bias: bool = False,
    ddof: Optional[int] = None
) -> Union[mx.array, np.array]:
    """
    Estimate a covariance matrix.

    Parameters:
        m: A 1-D or 2-D array containing multiple variables and observations.
        y: An additional set of variables and observations.
        rowvar: If True, each row represents a variable, with observations in the columns.
        bias: Default normalization (False) is by (N-1), where N is the number of observations.
        ddof: If not None, the default value implied by bias is overridden.

    Returns:
        Covariance matrix.
    """
    m_np = np.array(m)
    y_np = np.array(y) if y is not None else None
    result = np.cov(m_np, y=y_np, rowvar=rowvar, bias=bias, ddof=ddof)

    if isinstance(m, np.ndarray):
        return result
    return mx.array(result)

def corrcoef(
    x: Union[mx.array, np.array],
    y: Optional[Union[mx.array, np.array]] = None,
    rowvar: bool = True
) -> Union[mx.array, np.array]:
    """
    Compute Pearson product-moment correlation coefficients.

    Parameters:
        x: A 1-D or 2-D array containing multiple variables and observations.
        y: An additional set of variables and observations.
        rowvar: If True, each row represents a variable, with observations in the columns.

    Returns:
        Correlation coefficient matrix.
    """
    x_np = np.array(x)
    y_np = np.array(y) if y is not None else None
    result = np.corrcoef(x_np, y=y_np, rowvar=rowvar)

    if isinstance(x, np.ndarray):
        return result
    return mx.array(result)

def pearsonr(
    x: Union[mx.array, np.array],
    y: Union[mx.array, np.array]
) -> Tuple[float, float]:
    """
    Calculate a Pearson correlation coefficient and the p-value for testing non-correlation.

    Parameters:
        x: 1-D array of observations.
        y: 1-D array of observations of the same length as x.

    Returns:
        r: Pearson correlation coefficient.
        p-value: Two-tailed p-value.
    """
    from scipy.stats import pearsonr as sp_pearsonr
    x_np = np.array(x).flatten()
    y_np = np.array(y).flatten()
    r, p = sp_pearsonr(x_np, y_np)
    return float(r), float(p)

def spearmanr(
    x: Union[mx.array, np.array],
    y: Optional[Union[mx.array, np.array]] = None,
    axis: int = 0,
    nan_policy: str = 'propagate'
) -> Tuple[float, float]:
    """
    Calculate a Spearman rank-order correlation coefficient and the p-value.

    Parameters:
        x: 1-D or 2-D array of observations.
        y: Optional second 1-D or 2-D array of observations.
        axis: Axis along which to compute correlations.
        nan_policy: Defines how to handle nan values.

    Returns:
        correlation: Spearman correlation coefficient.
        p-value: Two-tailed p-value.
    """
    from scipy.stats import spearmanr as sp_spearmanr
    x_np = np.array(x)
    if y is None:
        r, p = sp_spearmanr(x_np, axis=axis, nan_policy=nan_policy)
    else:
        y_np = np.array(y)
        r, p = sp_spearmanr(x_np, y_np, axis=axis, nan_policy=nan_policy)
    return float(r), float(p)

# ===================== 概率分布 =====================
def norm_pdf(x: Union[mx.array, np.array], loc: float = 0.0, scale: float = 1.0) -> Union[mx.array, np.array]:
    """
    Probability density function of the normal distribution.

    Parameters:
        x: Quantiles.
        loc: Mean of the distribution.
        scale: Standard deviation of the distribution.

    Returns:
        PDF evaluated at x.
    """
    x_mx = _to_mx_array(x)
    result = mx.exp(-0.5 * ((x_mx - loc) / scale) ** 2) / (scale * mx.sqrt(2 * mx.pi))
    return np.array(result) if isinstance(x, np.ndarray) else result

def norm_cdf(x: Union[mx.array, np.array], loc: float = 0.0, scale: float = 1.0) -> Union[mx.array, np.array]:
    """
    Cumulative distribution function of the normal distribution.

    Parameters:
        x: Quantiles.
        loc: Mean of the distribution.
        scale: Standard deviation of the distribution.

    Returns:
        CDF evaluated at x.
    """
    x_mx = _to_mx_array(x)
    z = (x_mx - loc) / scale
    result = 0.5 * (1 + mx.erf(z / mx.sqrt(2)))
    return np.array(result) if isinstance(x, np.ndarray) else result

def norm_rvs(loc: float = 0.0, scale: float = 1.0, size: Optional[Union[int, Tuple[int, ...]]] = None) -> Union[mx.array, np.array]:
    """
    Random variates from the normal distribution.

    Parameters:
        loc: Mean of the distribution.
        scale: Standard deviation of the distribution.
        size: Output shape.

    Returns:
        Random variates.
    """
    result = mx.random.normal(loc=loc, scale=scale, shape=size if size is not None else ())
    return np.array(result) if size is not None else float(result)

def bernoulli_pmf(k: Union[mx.array, np.array], p: float = 0.5) -> Union[mx.array, np.array]:
    """
    Probability mass function of the Bernoulli distribution.

    Parameters:
        k: Quantiles.
        p: Probability of success.

    Returns:
        PMF evaluated at k.
    """
    k_mx = _to_mx_array(k)
    result = (p ** k_mx) * ((1 - p) ** (1 - k_mx))
    return np.array(result) if isinstance(k, np.ndarray) else result

def bernoulli_rvs(p: float = 0.5, size: Optional[Union[int, Tuple[int, ...]]] = None) -> Union[mx.array, np.array]:
    """
    Random variates from the Bernoulli distribution.

    Parameters:
        p: Probability of success.
        size: Output shape.

    Returns:
        Random variates (0 or 1).
    """
    result = mx.random.bernoulli(p=p, shape=size if size is not None else ())
    return np.array(result) if size is not None else int(result)

def multinomial_pmf(x: Union[mx.array, np.array], n: int, p: Union[list, np.array, mx.array]) -> Union[mx.array, np.array]:
    """
    Probability mass function of the multinomial distribution.

    Parameters:
        x: Quantiles. Each row is a vector of counts.
        n: Number of trials.
        p: Probabilities of each outcome.

    Returns:
        PMF evaluated at x.
    """
    x_mx = _to_mx_array(x)
    p_mx = _to_mx_array(p)

    # Compute factorial part (log for numerical stability)
    log_fact_n = mx.lgamma(n + 1)
    log_fact_x = mx.sum(mx.lgamma(x_mx + 1), axis=-1)
    log_pow_p = mx.sum(x_mx * mx.log(p_mx), axis=-1)

    result = mx.exp(log_fact_n - log_fact_x + log_pow_p)
    return np.array(result) if isinstance(x, np.ndarray) else result

def multinomial_rvs(n: int, p: Union[list, np.array, mx.array], size: Optional[Union[int, Tuple[int, ...]]] = None) -> Union[mx.array, np.array]:
    """
    Random variates from the multinomial distribution.

    Parameters:
        n: Number of trials.
        p: Probabilities of each outcome.
        size: Output shape.

    Returns:
        Random variates, each row is a vector of counts.
    """
    p_mx = _to_mx_array(p)
    result = mx.random.categorical(mx.log(p_mx), shape=(size if size is not None else 1, n))
    counts = mx.zeros((size if size is not None else 1, len(p_mx)), dtype=mx.int32)
    for i in range(size if size is not None else 1):
        counts[i] = mx.bincount(result[i], length=len(p_mx))
    result = counts[0] if size is None else counts
    return np.array(result) if isinstance(p, (list, np.ndarray)) else result

# ===================== 时序相关统计 =====================
def acf(
    x: Union[mx.array, np.array],
    nlags: Optional[int] = None,
    adjusted: bool = False,
    fft: bool = True
) -> Union[mx.array, np.array]:
    """
    Compute the autocorrelation function of a 1D array.

    Parameters:
        x: Input time series array. Must be 1-dimensional.
        nlags: Number of lags to compute ACF for. If None, defaults to min(10*log10(n), n//2 -1).
        adjusted: If True, denominators for ACF are n-k instead of n.
        fft: If True, use FFT to compute ACF (faster for large n).

    Returns:
        Autocorrelation values at lags 0, 1, ..., nlags.
    """
    # 输入校验
    x_np = np.array(x)
    if x_np.ndim > 1 and not (x_np.ndim == 2 and min(x_np.shape) == 1):
        raise ValueError("x must be a 1-dimensional array")
    x_np = x_np.flatten()
    n = len(x_np)

    if n < 2:
        raise ValueError("x must have at least 2 observations")
    if nlags is not None and nlags < 0:
        raise ValueError("nlags must be non-negative")

    if nlags is None:
        nlags = min(int(np.floor(10 * np.log10(n))), n // 2 - 1)
    nlags = min(nlags, n - 1)
    if nlags < 0:
        nlags = 0

    if fft:
        # Compute ACF via FFT
        n_pad = 2 ** int(np.ceil(np.log2(2 * n - 1)))
        f = np.fft.fft(x_np - x_np.mean(), n_pad)
        acf = np.fft.ifft(f * np.conj(f))[:n].real
        acf /= acf[0]
    else:
        # Compute ACF directly
        mean_x = x_np.mean()
        acov = np.zeros(nlags + 1)
        acov[0] = np.sum((x_np - mean_x) ** 2) / n
        for k in range(1, nlags + 1):
            acov[k] = np.sum((x_np[k:] - mean_x) * (x_np[:-k] - mean_x)) / (n - k if adjusted else n)
        acf = acov / acov[0]

    return acf[:nlags + 1]

def pacf(
    x: Union[mx.array, np.array],
    nlags: Optional[int] = None,
    method: str = 'ywadjusted'
) -> Union[mx.array, np.array]:
    """
    Compute the partial autocorrelation function of a 1D array.

    Parameters:
        x: Input time series array. Must be 1-dimensional.
        nlags: Number of lags to compute PACF for. If None, defaults to min(10*log10(n), n//2 -1).
        method: Method to use for PACF computation. Currently only 'ywadjusted' (Yule-Walker) is supported.

    Returns:
        Partial autocorrelation values at lags 0, 1, ..., nlags.
    """
    # 输入校验
    x_np = np.array(x)
    if x_np.ndim > 1 and not (x_np.ndim == 2 and min(x_np.shape) == 1):
        raise ValueError("x must be a 1-dimensional array")
    x_np = x_np.flatten()
    n = len(x_np)

    if n < 2:
        raise ValueError("x must have at least 2 observations")
    if nlags is not None and nlags < 0:
        raise ValueError("nlags must be non-negative")
    if method not in ['ywadjusted', 'yule-walker']:
        raise ValueError("Only 'ywadjusted' (Yule-Walker) method is currently supported")

    if nlags is None:
        nlags = min(int(np.floor(10 * np.log10(n))), n // 2 - 1)
    nlags = min(nlags, n - 1)
    if nlags < 0:
        nlags = 0

    # Compute using Yule-Walker equations
    acf_vals = acf(x_np, nlags=nlags)
    pacf_vals = np.zeros(nlags + 1)
    pacf_vals[0] = 1.0

    for k in range(1, nlags + 1):
        R = np.zeros((k, k))
        for i in range(k):
            for j in range(k):
                R[i, j] = acf_vals[abs(i - j)]
        r = acf_vals[1:k+1]
        pacf_vals[k] = np.linalg.solve(R, r)[-1]

    return pacf_vals
