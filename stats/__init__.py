"""
Statistics module for mlx-ml, compatible with scipy.stats API.
"""

from .stats import (
    # 描述性统计
    quantile,
    percentile,
    mode,
    skew,
    kurtosis,

    # 相关性
    cov,
    corrcoef,
    pearsonr,
    spearmanr,

    # 分布函数
    norm_pdf,
    norm_cdf,
    norm_rvs,
    bernoulli_pmf,
    bernoulli_rvs,
    multinomial_pmf,
    multinomial_rvs,

    # 时序相关统计
    acf,
    pacf
)

__all__ = [
    # 描述性统计
    'quantile',
    'percentile',
    'mode',
    'skew',
    'kurtosis',

    # 相关性
    'cov',
    'corrcoef',
    'pearsonr',
    'spearmanr',

    # 分布函数
    'norm_pdf',
    'norm_cdf',
    'norm_rvs',
    'bernoulli_pmf',
    'bernoulli_rvs',
    'multinomial_pmf',
    'multinomial_rvs',

    # 时序相关统计
    'acf',
    'pacf'
]
