import mlx.core as mx
import numpy as np
from typing import Union


def mean_squared_error(
    y_true: Union[mx.array, np.array],
    y_pred: Union[mx.array, np.array],
    squared: bool = True
) -> float:
    """
    Mean squared error regression loss.

    Parameters:
        y_true: Ground truth (correct) target values.
        y_pred: Estimated target values.
        squared: If True returns MSE value, if False returns RMSE value.

    Returns:
        Mean squared error or root mean squared error.
    """
    y_true_np = np.array(y_true).flatten()
    y_pred_np = np.array(y_pred).flatten()

    if len(y_true_np) != len(y_pred_np):
        raise ValueError("y_true and y_pred have different lengths")

    mse = np.mean((y_true_np - y_pred_np) ** 2)

    if squared:
        return float(mse)
    else:
        return float(np.sqrt(mse))


def mean_absolute_error(
    y_true: Union[mx.array, np.array],
    y_pred: Union[mx.array, np.array]
) -> float:
    """
    Mean absolute error regression loss.

    Parameters:
        y_true: Ground truth (correct) target values.
        y_pred: Estimated target values.

    Returns:
        Mean absolute error.
    """
    y_true_np = np.array(y_true).flatten()
    y_pred_np = np.array(y_pred).flatten()

    if len(y_true_np) != len(y_pred_np):
        raise ValueError("y_true and y_pred have different lengths")

    mae = np.mean(np.abs(y_true_np - y_pred_np))
    return float(mae)


def mean_absolute_percentage_error(
    y_true: Union[mx.array, np.array],
    y_pred: Union[mx.array, np.array]
) -> float:
    """
    Mean absolute percentage error regression loss.

    Note: This function is not symmetric and can result in infinite values
    when y_true contains zero values.

    Parameters:
        y_true: Ground truth (correct) target values.
        y_pred: Estimated target values.

    Returns:
        Mean absolute percentage error as a fraction (not percentage).
    """
    y_true_np = np.array(y_true).flatten()
    y_pred_np = np.array(y_pred).flatten()

    if len(y_true_np) != len(y_pred_np):
        raise ValueError("y_true and y_pred have different lengths")

    if np.any(y_true_np == 0):
        raise ValueError("Mean absolute percentage error is not defined when y_true contains zero values")

    mape = np.mean(np.abs((y_true_np - y_pred_np) / y_true_np))
    return float(mape)


def r2_score(
    y_true: Union[mx.array, np.array],
    y_pred: Union[mx.array, np.array]
) -> float:
    """
    R^2 (coefficient of determination) regression score function.

    Best possible score is 1.0 and it can be negative (because the model can be arbitrarily worse).
    A constant model that always predicts the expected value of y, disregarding the input features,
    would get an R^2 score of 0.0.

    Parameters:
        y_true: Ground truth (correct) target values.
        y_pred: Estimated target values.

    Returns:
        R^2 score.
    """
    y_true_np = np.array(y_true).flatten()
    y_pred_np = np.array(y_pred).flatten()

    if len(y_true_np) != len(y_pred_np):
        raise ValueError("y_true and y_pred have different lengths")

    ss_res = np.sum((y_true_np - y_pred_np) ** 2)
    ss_tot = np.sum((y_true_np - np.mean(y_true_np)) ** 2)

    if ss_tot == 0:
        return 0.0  # y_true is constant

    r2 = 1 - (ss_res / ss_tot)
    return float(r2)


def root_mean_squared_error(
    y_true: Union[mx.array, np.array],
    y_pred: Union[mx.array, np.array]
) -> float:
    """
    Root mean squared error regression loss.

    Parameters:
        y_true: Ground truth (correct) target values.
        y_pred: Estimated target values.

    Returns:
        Root mean squared error.
    """
    return mean_squared_error(y_true, y_pred, squared=False)
