"""
mlx-ml: A high-performance machine learning library for Apple Silicon, built on MLX.
Compatible with scikit-learn API, optimized for GPU acceleration and unified memory.
"""

__version__ = "0.1.0"

# Import core modules
from . import base
from . import neighbors
from . import spatial
from . import preprocessing
from . import linear_model
from . import cluster
from . import decomposition
from . import naive_bayes
from . import linalg
from . import stats
from . import metrics
from . import ensemble
from . import svm
from . import manifold

# Import commonly used classes for easy access
from .base import BaseEstimator
from .neighbors import (
    KNeighborsClassifier,
    KNeighborsRegressor,
    NearestNeighbors
)
from .spatial import DistanceMetric, get_metric
from .preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    LabelEncoder,
    OneHotEncoder,
    OrdinalEncoder,
    SimpleImputer,
    train_test_split,
    KFold,
    StratifiedKFold
)
from .linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
    LogisticRegression
)
from .cluster import KMeans, DBSCAN, GaussianMixture
from .decomposition import PCA
from .tree import DecisionTreeClassifier, DecisionTreeRegressor
from .ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor

__all__ = [
    # Base
    'BaseEstimator',

    # Neighbors
    'KNeighborsClassifier',
    'KNeighborsRegressor',
    'NearestNeighbors',

    # Spatial
    'DistanceMetric',
    'get_metric',

    # Tree
    'DecisionTreeClassifier',
    'DecisionTreeRegressor',

    # Ensemble
    'RandomForestClassifier',
    'RandomForestRegressor',
    'GradientBoostingClassifier',
    'GradientBoostingRegressor',

    # Clustering
    'KMeans',
    'DBSCAN',
    'GaussianMixture',

    # SVM
    'SVC',
    'SVR',

    # Naive Bayes
    'GaussianNB',
    'MultinomialNB',
    'BernoulliNB',

    # Metrics
    'accuracy_score',
    'confusion_matrix',
    'precision_score',
    'recall_score',
    'f1_score',
    'roc_auc_score',
    'mean_squared_error',
    'mean_absolute_error',
    'r2_score',
    'adjusted_rand_score',
    'silhouette_score',

    # Manifold
    'TSNE',
    'UMAP',

    # Modules
    'base',
    'neighbors',
    'spatial',
    'preprocessing',
    'linear_model',
    'cluster',
    'decomposition',
    'naive_bayes',
    'linalg',
    'stats',
    'metrics',
    'ensemble',
    'svm',
    'manifold'
]
