"""
Linear algebra module for mlx-ml, compatible with scipy.linalg API.
"""

from .linalg import (
    inv,
    pinv,
    matrix_power,
    det,
    trace,
    eig,
    eigh,
    svd,
    qr,
    cholesky,
    solve,
    lstsq
)

__all__ = [
    'inv',
    'pinv',
    'matrix_power',
    'det',
    'trace',
    'eig',
    'eigh',
    'svd',
    'qr',
    'cholesky',
    'solve',
    'lstsq'
]
