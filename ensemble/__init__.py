"""
Ensemble learning module for mlx-ml, compatible with scikit-learn API.
"""

from .random_forest import RandomForestClassifier, RandomForestRegressor
from .gradient_boosting import GradientBoostingClassifier, GradientBoostingRegressor

__all__ = ['RandomForestClassifier', 'RandomForestRegressor', 'GradientBoostingClassifier', 'GradientBoostingRegressor']
