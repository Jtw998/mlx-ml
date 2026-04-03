import mlx.core as mx
import numpy as np
from typing import Optional, Union, Tuple

def _to_mx_array(x: Union[mx.array, np.array]) -> mx.array:
    """Convert input to MLX array if it's numpy array"""
    return mx.array(x) if isinstance(x, np.ndarray) else x

def inv(a: Union[mx.array, np.array]) -> Union[mx.array, np.array]:
    """
    Compute the (multiplicative) inverse of a matrix.

    Parameters:
        a: Square matrix to be inverted.

    Returns:
        Inverse of the matrix `a`.
    """
    a_mx = _to_mx_array(a)
    if a_mx.ndim < 2 or a_mx.shape[-1] != a_mx.shape[-2]:
        raise ValueError("Last 2 dimensions of the array must be square")

    result = mx.inv(a_mx)
    return np.array(result) if isinstance(a, np.ndarray) else result

def pinv(a: Union[mx.array, np.array], rcond: float = 1e-15) -> Union[mx.array, np.array]:
    """
    Compute the (Moore-Penrose) pseudo-inverse of a matrix.

    Parameters:
        a: Matrix to be pseudo-inverted.
        rcond: Cutoff for small singular values.

    Returns:
        Pseudo-inverse of the matrix `a`.
    """
    a_mx = _to_mx_array(a)
    result = mx.pinv(a_mx, rcond=rcond)
    return np.array(result) if isinstance(a, np.ndarray) else result

def matrix_power(a: Union[mx.array, np.array], power: int) -> Union[mx.array, np.array]:
    """
    Raise a square matrix to the (integer) power `power`.

    Parameters:
        a: Square matrix to be raised to power.
        power: The exponent can be any integer, positive, negative, or zero.

    Returns:
        The matrix `a` raised to the power `power`.
    """
    a_mx = _to_mx_array(a)
    if a_mx.ndim < 2 or a_mx.shape[-1] != a_mx.shape[-2]:
        raise ValueError("Last 2 dimensions of the array must be square")

    if power == 0:
        result = mx.eye(a_mx.shape[-1], dtype=a_mx.dtype)
    elif power < 0:
        a_mx = inv(a_mx)
        power = -power
        result = a_mx
        for _ in range(power - 1):
            result = result @ a_mx
    else:
        result = a_mx
        for _ in range(power - 1):
            result = result @ a_mx

    return np.array(result) if isinstance(a, np.ndarray) else result

def det(a: Union[mx.array, np.array]) -> Union[float, mx.array, np.array]:
    """
    Compute the determinant of a matrix.

    Parameters:
        a: Square matrix to compute the determinant for.

    Returns:
        Determinant of `a`.
    """
    a_mx = _to_mx_array(a)
    if a_mx.ndim < 2 or a_mx.shape[-1] != a_mx.shape[-2]:
        raise ValueError("Last 2 dimensions of the array must be square")

    result = mx.det(a_mx)
    return float(result) if a.ndim == 2 else (np.array(result) if isinstance(a, np.ndarray) else result)

def trace(a: Union[mx.array, np.array], offset: int = 0) -> Union[float, mx.array, np.array]:
    """
    Compute the sum along diagonals of the array.

    Parameters:
        a: Array from which the diagonals are taken.
        offset: Offset of the diagonal from the main diagonal.

    Returns:
        Sum of the diagonal elements.
    """
    a_mx = _to_mx_array(a)
    result = mx.trace(a_mx, offset=offset)
    return float(result) if a.ndim == 2 else (np.array(result) if isinstance(a, np.ndarray) else result)

def eig(a: Union[mx.array, np.array]) -> Tuple[Union[mx.array, np.array], Union[mx.array, np.array]]:
    """
    Compute the eigenvalues and right eigenvectors of a square array.

    Parameters:
        a: Square matrix to compute eigenvalues and eigenvectors for.

    Returns:
        w: The eigenvalues, each repeated according to its multiplicity.
        v: The normalized (unit "length") eigenvectors.
    """
    a_mx = _to_mx_array(a)
    if a_mx.ndim < 2 or a_mx.shape[-1] != a_mx.shape[-2]:
        raise ValueError("Last 2 dimensions of the array must be square")

    w, v = mx.eig(a_mx)
    if isinstance(a, np.ndarray):
        return np.array(w), np.array(v)
    return w, v

def eigh(a: Union[mx.array, np.array], UPLO: str = 'L') -> Tuple[Union[mx.array, np.array], Union[mx.array, np.array]]:
    """
    Compute the eigenvalues and eigenvectors of a complex Hermitian or real symmetric matrix.

    Parameters:
        a: Hermitian or symmetric matrix whose eigenvalues and eigenvectors are to be computed.
        UPLO: Specifies whether the calculation is done with the lower ('L') or upper ('U') triangular part of `a`.

    Returns:
        w: The eigenvalues, in ascending order.
        v: The normalized eigenvectors.
    """
    a_mx = _to_mx_array(a)
    if a_mx.ndim < 2 or a_mx.shape[-1] != a_mx.shape[-2]:
        raise ValueError("Last 2 dimensions of the array must be square")

    w, v = mx.eigh(a_mx, UPLO=UPLO)
    if isinstance(a, np.ndarray):
        return np.array(w), np.array(v)
    return w, v

def svd(a: Union[mx.array, np.array], full_matrices: bool = True, compute_uv: bool = True) -> Union[
    Union[mx.array, np.array],
    Tuple[Union[mx.array, np.array], Union[mx.array, np.array], Union[mx.array, np.array]]
]:
    """
    Singular Value Decomposition.

    Parameters:
        a: Matrix to decompose.
        full_matrices: If True, U and Vh have full size, otherwise they are truncated.
        compute_uv: Whether to compute U and Vh in addition to s.

    Returns:
        U: Unitary matrix.
        s: Singular values, sorted in descending order.
        Vh: Unitary matrix, conjugate transpose of V.
    """
    a_mx = _to_mx_array(a)
    result = mx.svd(a_mx, full_matrices=full_matrices, compute_uv=compute_uv)

    if isinstance(a, np.ndarray):
        if compute_uv:
            return np.array(result[0]), np.array(result[1]), np.array(result[2])
        return np.array(result)
    return result

def qr(a: Union[mx.array, np.array], mode: str = 'reduced') -> Tuple[Union[mx.array, np.array], Union[mx.array, np.array]]:
    """
    Compute the QR decomposition of a matrix.

    Parameters:
        a: Matrix to decompose.
        mode: 'reduced' (default), 'complete', 'r', 'raw'.

    Returns:
        Q: Orthogonal matrix.
        R: Upper triangular matrix.
    """
    a_mx = _to_mx_array(a)
    Q, R = mx.qr(a_mx, mode=mode)
    if isinstance(a, np.ndarray):
        return np.array(Q), np.array(R)
    return Q, R

def cholesky(a: Union[mx.array, np.array], lower: bool = True) -> Union[mx.array, np.array]:
    """
    Compute the Cholesky decomposition of a Hermitian positive-definite matrix.

    Parameters:
        a: Hermitian positive-definite matrix.
        lower: If True, return lower triangular matrix, else upper.

    Returns:
        L: Lower or upper triangular Cholesky factor.
    """
    a_mx = _to_mx_array(a)
    if a_mx.ndim < 2 or a_mx.shape[-1] != a_mx.shape[-2]:
        raise ValueError("Last 2 dimensions of the array must be square")

    result = mx.cholesky(a_mx, upper=not lower)
    return np.array(result) if isinstance(a, np.ndarray) else result

def solve(a: Union[mx.array, np.array], b: Union[mx.array, np.array]) -> Union[mx.array, np.array]:
    """
    Solve the linear equation set a * x = b for the unknown x.

    Parameters:
        a: Square coefficient matrix.
        b: Ordinate or "dependent variable" values.

    Returns:
        x: Solution to the system a x = b.
    """
    a_mx = _to_mx_array(a)
    b_mx = _to_mx_array(b)
    if a_mx.ndim < 2 or a_mx.shape[-1] != a_mx.shape[-2]:
        raise ValueError("Last 2 dimensions of `a` must be square")

    result = mx.solve(a_mx, b_mx)
    return np.array(result) if isinstance(a, np.ndarray) else result

def lstsq(a: Union[mx.array, np.array], b: Union[mx.array, np.array], rcond: Optional[float] = None) -> Tuple[
    Union[mx.array, np.array],
    Union[mx.array, np.array],
    Union[mx.array, np.array],
    Union[mx.array, np.array]
]:
    """
    Compute the least-squares solution to a linear matrix equation.

    Parameters:
        a: Coefficient matrix.
        b: Ordinate values.
        rcond: Cutoff for small singular values.

    Returns:
        x: Least-squares solution.
        residuals: Sums of squared residuals.
        rank: Rank of matrix `a`.
        s: Singular values of `a`.
    """
    a_mx = _to_mx_array(a)
    b_mx = _to_mx_array(b)

    if rcond is None:
        rcond = -1.0

    # Compute SVD
    U, s, Vh = svd(a_mx, full_matrices=False)

    # Compute pseudoinverse
    cutoff = rcond * mx.max(s)
    s_inv = mx.where(s > cutoff, 1 / s, 0.0)

    # Compute solution
    x = Vh.T @ (mx.diag(s_inv) @ (U.T @ b_mx))

    # Compute residuals
    residuals = mx.sum((a_mx @ x - b_mx) ** 2, axis=0) if b_mx.ndim > 1 else mx.sum((a_mx @ x - b_mx) ** 2)

    # Compute rank
    rank = mx.sum(s > cutoff)

    if isinstance(a, np.ndarray):
        return np.array(x), np.array(residuals), int(rank), np.array(s)
    return x, residuals, rank, s
