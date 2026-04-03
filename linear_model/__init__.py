"""
Linear models module for mlx-ml, compatible with scikit-learn API.
"""

# Regression models
from .linear_regression import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet
)

# Classification models
from .logistic_regression import LogisticRegression

# Solvers
from .solver import SGDSolver, LBFGSSolver
from .base import get_solver

__all__ = [
    # Regression
    'LinearRegression',
    'Ridge',
    'Lasso',
    'ElasticNet',

    # Classification
    'LogisticRegression',

    # Solvers
    'SGDSolver',
    'LBFGSSolver',
    'get_solver'
]
