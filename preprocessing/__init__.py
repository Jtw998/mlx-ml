"""
Preprocessing module for mlx-ml, compatible with scikit-learn API.
"""

# Scalers
from .scalers import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler
)

# Encoders
from .encoders import (
    LabelEncoder,
    OneHotEncoder,
    OrdinalEncoder
)

# Imputers
from .impute import (
    SimpleImputer
)

# Model selection utilities
from .model_selection import (
    train_test_split,
    KFold,
    StratifiedKFold
)

__all__ = [
    # Scalers
    'StandardScaler',
    'MinMaxScaler',
    'RobustScaler',

    # Encoders
    'LabelEncoder',
    'OneHotEncoder',
    'OrdinalEncoder',

    # Imputers
    'SimpleImputer',

    # Model selection
    'train_test_split',
    'KFold',
    'StratifiedKFold'
]
