import mlx.core as mx
import numpy as np
from typing import Optional, Union, Tuple, List, Iterator


def train_test_split(*arrays: Union[mx.array, np.array], test_size: Optional[Union[float, int]] = None,
                     train_size: Optional[Union[float, int]] = None, random_state: Optional[int] = None,
                     shuffle: bool = True, stratify: Optional[Union[mx.array, np.array]] = None) -> Tuple:
    """
    Split arrays or matrices into random train and test subsets.

    Parameters:
        *arrays: Sequence of indexables with same length / shape[0]
        test_size: If float, should be between 0.0 and 1.0 and represent the proportion
            of the dataset to include in the test split. If int, represents the absolute
            number of test samples. If None, the value is set to the complement of the train size.
            If train_size is also None, it will be set to 0.25.
        train_size: If float, should be between 0.0 and 1.0 and represent the proportion
            of the dataset to include in the train split. If int, represents the absolute
            number of train samples. If None, the value is automatically set to the complement of the test size.
        random_state: Controls the shuffling applied to the data before applying the split
        shuffle: Whether or not to shuffle the data before splitting. If shuffle=False then stratify must be None.
        stratify: If not None, data is split in a stratified fashion, using this as the class labels.

    Returns:
        List containing train-test split of inputs
    """
    if len(arrays) == 0:
        raise ValueError("At least one array required as input")

    # Convert all arrays to numpy for processing
    arrays_np = [np.array(arr) if isinstance(arr, mx.array) else arr for arr in arrays]
    n_samples = arrays_np[0].shape[0]

    for arr in arrays_np[1:]:
        if arr.shape[0] != n_samples:
            raise ValueError("All arrays must have the same first dimension")

    if stratify is not None and not shuffle:
        raise ValueError("If stratify is not None, shuffle must be True")

    # Determine train and test sizes
    if test_size is None and train_size is None:
        test_size = 0.25

    if isinstance(test_size, float):
        if test_size <= 0 or test_size >= 1:
            raise ValueError("test_size must be between 0 and 1")
        n_test = int(n_samples * test_size)
    elif isinstance(test_size, int):
        if test_size < 0 or test_size >= n_samples:
            raise ValueError(f"test_size must be between 0 and {n_samples - 1}")
        n_test = test_size
    else:
        n_test = 0

    if isinstance(train_size, float):
        if train_size <= 0 or train_size >= 1:
            raise ValueError("train_size must be between 0 and 1")
        n_train = int(n_samples * train_size)
    elif isinstance(train_size, int):
        if train_size < 0 or train_size > n_samples:
            raise ValueError(f"train_size must be between 0 and {n_samples}")
        n_train = train_size
    else:
        n_train = n_samples - n_test

    if n_train + n_test > n_samples:
        raise ValueError(f"sum of train_size and test_size exceeds sample count {n_samples}")

    # Generate indices
    if shuffle:
        rng = np.random.RandomState(random_state)
        indices = rng.permutation(n_samples)
    else:
        indices = np.arange(n_samples)

    # Handle stratification
    if stratify is not None:
        stratify_np = np.array(stratify) if isinstance(stratify, mx.array) else stratify
        if stratify_np.shape[0] != n_samples:
            raise ValueError("stratify array must have same length as input arrays")

        from sklearn.model_selection import train_test_split as sk_train_test_split
        _, _, train_idx, test_idx = sk_train_test_split(
            indices, indices, test_size=test_size, train_size=train_size,
            random_state=random_state, shuffle=shuffle, stratify=stratify_np
        )
    else:
        test_idx = indices[:n_test]
        train_idx = indices[n_test:n_test + n_train]

    # Split all arrays
    result = []
    for i, arr in enumerate(arrays_np):
        train_arr = arr[train_idx]
        test_arr = arr[test_idx]
        # Convert back to mlx array if original was mlx array
        if isinstance(arrays[i], mx.array):
            result.append(mx.array(train_arr))
            result.append(mx.array(test_arr))
        else:
            result.append(train_arr)
            result.append(test_arr)

    return tuple(result)


class KFold:
    """
    K-Folds cross-validator.

    Provides train/test indices to split data in train/test sets. Split dataset into k consecutive folds
    (without shuffling by default).

    Parameters:
        n_splits: Number of folds, must be at least 2
        shuffle: Whether to shuffle the data before splitting into batches
        random_state: When shuffle is True, random_state affects the ordering of the indices
    """

    def __init__(self, n_splits: int = 5, shuffle: bool = False, random_state: Optional[int] = None):
        if n_splits < 2:
            raise ValueError(f"n_splits must be at least 2, got {n_splits}")
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def get_n_splits(self) -> int:
        """Returns the number of splitting iterations in the cross-validator"""
        return self.n_splits

    def split(self, X: Union[mx.array, np.array], y: Optional[Union[mx.array, np.array]] = None,
              groups: Optional[Union[mx.array, np.array]] = None) -> Iterator[Tuple[np.array, np.array]]:
        """
        Generate indices to split data into training and test set.

        Parameters:
            X: Training data of shape (n_samples, n_features)
            y: Target variable, ignored for this CV strategy
            groups: Group labels, ignored for this CV strategy

        Yields:
            train: The training set indices for that split
            test: The testing set indices for that split
        """
        X_np = np.array(X) if isinstance(X, mx.array) else X
        n_samples = X_np.shape[0]

        if self.n_splits > n_samples:
            raise ValueError(f"n_splits={self.n_splits} cannot be greater than the number of samples={n_samples}")

        indices = np.arange(n_samples)
        if self.shuffle:
            rng = np.random.RandomState(self.random_state)
            rng.shuffle(indices)

        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits, dtype=int)
        fold_sizes[:n_samples % self.n_splits] += 1
        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            test_idx = indices[start:stop]
            train_idx = np.concatenate([indices[:start], indices[stop:]])
            yield train_idx, test_idx
            current = stop


class StratifiedKFold(KFold):
    """
    Stratified K-Folds cross-validator.

    Provides train/test indices to split data in train/test sets.
    This cross-validation object is a variation of KFold that returns stratified folds.
    The folds are made by preserving the percentage of samples for each class.

    Parameters:
        n_splits: Number of folds, must be at least 2
        shuffle: Whether to shuffle each class's samples before splitting into batches
        random_state: When shuffle is True, random_state affects the ordering of the indices
    """

    def split(self, X: Union[mx.array, np.array], y: Union[mx.array, np.array],
              groups: Optional[Union[mx.array, np.array]] = None) -> Iterator[Tuple[np.array, np.array]]:
        """
        Generate indices to split data into training and test set.

        Parameters:
            X: Training data of shape (n_samples, n_features)
            y: Target variable of shape (n_samples,)
            groups: Group labels, ignored for this CV strategy

        Yields:
            train: The training set indices for that split
            test: The testing set indices for that split
        """
        X_np = np.array(X) if isinstance(X, mx.array) else X
        y_np = np.array(y) if isinstance(y, mx.array) else y
        n_samples = X_np.shape[0]

        if y_np.shape[0] != n_samples:
            raise ValueError("y must have the same length as X")

        if self.n_splits > n_samples:
            raise ValueError(f"n_splits={self.n_splits} cannot be greater than the number of samples={n_samples}")

        # Use scikit-learn's StratifiedKFold for proper stratification
        from sklearn.model_selection import StratifiedKFold as SKStratifiedKFold
        skf = SKStratifiedKFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state)
        for train_idx, test_idx in skf.split(X_np, y_np):
            yield train_idx, test_idx
