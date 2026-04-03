import mlx.core as mx
import numpy as np
from typing import Optional, Union


def accuracy_score(
    y_true: Union[mx.array, np.array],
    y_pred: Union[mx.array, np.array],
    normalize: bool = True
) -> Union[float, int]:
    """
    Accuracy classification score.

    Parameters:
        y_true: Ground truth (correct) labels.
        y_pred: Predicted labels, as returned by a classifier.
        normalize: If False, return the number of correctly classified samples.
            Otherwise, return the fraction of correctly classified samples.

    Returns:
        If normalize is True, return the fraction of correctly classified samples (float),
        else return the number of correctly classified samples (int).
    """
    y_true_np = np.array(y_true).flatten()
    y_pred_np = np.array(y_pred).flatten()

    if len(y_true_np) != len(y_pred_np):
        raise ValueError("y_true and y_pred have different lengths")

    correct = np.sum(y_true_np == y_pred_np)

    if normalize:
        return correct / len(y_true_np)
    else:
        return correct


def confusion_matrix(
    y_true: Union[mx.array, np.array],
    y_pred: Union[mx.array, np.array],
    labels: Optional[Union[list, np.array, mx.array]] = None
) -> np.array:
    """
    Compute confusion matrix to evaluate the accuracy of a classification.

    Parameters:
        y_true: Ground truth (correct) target values.
        y_pred: Estimated targets as returned by a classifier.
        labels: List of labels to index the matrix. This may be used to reorder
            or select a subset of labels. If None is given, those that appear
            at least once in y_true or y_pred are used in sorted order.

    Returns:
        Confusion matrix of shape (n_classes, n_classes)
    """
    y_true_np = np.array(y_true).flatten()
    y_pred_np = np.array(y_pred).flatten()

    if len(y_true_np) != len(y_pred_np):
        raise ValueError("y_true and y_pred have different lengths")

    if labels is None:
        labels = np.unique(np.concatenate([y_true_np, y_pred_np]))
    else:
        labels = np.array(labels).flatten()

    n_classes = len(labels)
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)

    label_to_idx = {label: i for i, label in enumerate(labels)}

    for t, p in zip(y_true_np, y_pred_np):
        if t in label_to_idx and p in label_to_idx:
            cm[label_to_idx[t], label_to_idx[p]] += 1

    return cm


def precision_recall_fscore_support(
    y_true: Union[mx.array, np.array],
    y_pred: Union[mx.array, np.array],
    average: Optional[str] = None,
    labels: Optional[Union[list, np.array, mx.array]] = None,
    zero_division: Union[str, int, float] = "warn"
) -> tuple:
    """
    Compute precision, recall, F-measure and support for each class.

    Parameters:
        y_true: Ground truth (correct) target values.
        y_pred: Estimated targets as returned by a classifier.
        average: This parameter is required for multiclass/multilabel targets.
            If None, the scores for each class are returned. Otherwise, this
            determines the type of averaging performed on the data:
            'micro': Calculate metrics globally by counting the total true positives,
                false negatives and false positives.
            'macro': Calculate metrics for each label, and find their unweighted mean.
                This does not take label imbalance into account.
            'weighted': Calculate metrics for each label, and find their average weighted
                by support (the number of true instances for each label). This alters
                'macro' to account for label imbalance.
        labels: Optional list of label indices to include when average != None.
        zero_division: Sets the value to return when there is a zero division. If set to
            "warn", this acts as 0 but warnings are also raised.

    Returns:
        precision: Precision score(s)
        recall: Recall score(s)
        fscore: F1 score(s)
        support: Number of occurrences of each label in y_true
    """
    y_true_np = np.array(y_true).flatten()
    y_pred_np = np.array(y_pred).flatten()

    if labels is None:
        labels = np.unique(np.concatenate([y_true_np, y_pred_np]))
    else:
        labels = np.array(labels).flatten()

    cm = confusion_matrix(y_true_np, y_pred_np, labels=labels)
    n_classes = len(labels)

    tp = np.diag(cm)
    fp = np.sum(cm, axis=0) - tp
    fn = np.sum(cm, axis=1) - tp
    support = np.sum(cm, axis=1)

    # Handle zero division
    if zero_division == "warn":
        zero_val = 0.0
    elif isinstance(zero_division, (int, float)):
        zero_val = float(zero_division)
    else:
        raise ValueError("zero_division must be 'warn', int or float")

    # Compute precision
    denom = tp + fp
    precision = np.where(denom > 0, tp / denom, zero_val)

    # Compute recall
    denom = tp + fn
    recall = np.where(denom > 0, tp / denom, zero_val)

    # Compute f1 score
    denom = precision + recall
    fscore = np.where(denom > 0, 2 * (precision * recall) / denom, zero_val)

    if average is None:
        return precision, recall, fscore, support
    elif average == 'micro':
        tp_total = np.sum(tp)
        fp_total = np.sum(fp)
        fn_total = np.sum(fn)
        prec_micro = tp_total / (tp_total + fp_total) if tp_total + fp_total > 0 else zero_val
        rec_micro = tp_total / (tp_total + fn_total) if tp_total + fn_total > 0 else zero_val
        f1_micro = 2 * prec_micro * rec_micro / (prec_micro + rec_micro) if prec_micro + rec_micro > 0 else zero_val
        return prec_micro, rec_micro, f1_micro, None
    elif average == 'macro':
        return np.mean(precision), np.mean(recall), np.mean(fscore), None
    elif average == 'weighted':
        weights = support / np.sum(support) if np.sum(support) > 0 else np.ones_like(support) / n_classes
        prec_weighted = np.sum(precision * weights)
        rec_weighted = np.sum(recall * weights)
        f1_weighted = np.sum(fscore * weights)
        return prec_weighted, rec_weighted, f1_weighted, None
    else:
        raise ValueError(f"Unsupported average type: {average}")


def precision_score(
    y_true: Union[mx.array, np.array],
    y_pred: Union[mx.array, np.array],
    average: Optional[str] = 'macro',
    labels: Optional[Union[list, np.array, mx.array]] = None,
    zero_division: Union[str, int, float] = "warn"
) -> Union[float, np.array]:
    """
    Compute the precision.

    Parameters:
        See precision_recall_fscore_support for parameters.

    Returns:
        Precision score(s)
    """
    p, _, _, _ = precision_recall_fscore_support(
        y_true, y_pred, average=average, labels=labels, zero_division=zero_division
    )
    return p


def recall_score(
    y_true: Union[mx.array, np.array],
    y_pred: Union[mx.array, np.array],
    average: Optional[str] = 'macro',
    labels: Optional[Union[list, np.array, mx.array]] = None,
    zero_division: Union[str, int, float] = "warn"
) -> Union[float, np.array]:
    """
    Compute the recall.

    Parameters:
        See precision_recall_fscore_support for parameters.

    Returns:
        Recall score(s)
    """
    _, r, _, _ = precision_recall_fscore_support(
        y_true, y_pred, average=average, labels=labels, zero_division=zero_division
    )
    return r


def f1_score(
    y_true: Union[mx.array, np.array],
    y_pred: Union[mx.array, np.array],
    average: Optional[str] = 'macro',
    labels: Optional[Union[list, np.array, mx.array]] = None,
    zero_division: Union[str, int, float] = "warn"
) -> Union[float, np.array]:
    """
    Compute the F1 score, also known as balanced F-score or F-measure.

    Parameters:
        See precision_recall_fscore_support for parameters.

    Returns:
        F1 score(s)
    """
    _, _, f, _ = precision_recall_fscore_support(
        y_true, y_pred, average=average, labels=labels, zero_division=zero_division
    )
    return f


def roc_auc_score(
    y_true: Union[mx.array, np.array],
    y_score: Union[mx.array, np.array],
    average: Optional[str] = 'macro'
) -> float:
    """
    Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores.

    Note: This implementation currently only supports binary classification.

    Parameters:
        y_true: True binary labels.
        y_score: Target scores, can either be probability estimates of the positive class,
            confidence values, or non-thresholded measure of decisions.
        average: Average type to use. Currently only 'macro' is supported for binary case.

    Returns:
        ROC AUC score.
    """
    y_true_np = np.array(y_true).flatten()
    y_score_np = np.array(y_score).flatten()

    if len(y_true_np) != len(y_score_np):
        raise ValueError("y_true and y_score have different lengths")

    classes = np.unique(y_true_np)
    if len(classes) != 2:
        raise NotImplementedError("roc_auc_score currently only supports binary classification")

    # Sort by score
    desc_score_indices = np.argsort(y_score_np, kind="mergesort")[::-1]
    y_true_sorted = y_true_np[desc_score_indices]
    y_score_sorted = y_score_np[desc_score_indices]

    # Find distinct threshold values
    distinct_value_indices = np.where(np.diff(y_score_sorted))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true_sorted.size - 1]

    tps = np.cumsum(y_true_sorted)[threshold_idxs]
    fps = 1 + threshold_idxs - tps

    # Add end points
    tps = np.r_[0, tps]
    fps = np.r_[0, fps]

    # Compute AUC
    fpr = fps / fps[-1] if fps[-1] > 0 else fps
    tpr = tps / tps[-1] if tps[-1] > 0 else tps

    # Compute trapezoidal AUC manually
    auc = 0.0
    for i in range(1, len(fpr)):
        auc += (fpr[i] - fpr[i-1]) * (tpr[i] + tpr[i-1]) / 2
    return float(auc)
