"""
Metrics module for mlx-ml, compatible with scikit-learn API.
"""

# Classification metrics
from .classification import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# Regression metrics
from .regression import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
    root_mean_squared_error
)

# Clustering metrics
from .cluster import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score
)

__all__ = [
    # Classification
    'accuracy_score',
    'confusion_matrix',
    'precision_recall_fscore_support',
    'precision_score',
    'recall_score',
    'f1_score',
    'roc_auc_score',

    # Regression
    'mean_squared_error',
    'mean_absolute_error',
    'mean_absolute_percentage_error',
    'r2_score',
    'root_mean_squared_error',

    # Clustering
    'adjusted_rand_score',
    'normalized_mutual_info_score',
    'silhouette_score'
]
