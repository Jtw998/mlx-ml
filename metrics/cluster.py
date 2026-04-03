import mlx.core as mx
import numpy as np
from typing import Union
from scipy.spatial.distance import cdist as scipy_cdist


def adjusted_rand_score(
    labels_true: Union[mx.array, np.array],
    labels_pred: Union[mx.array, np.array]
) -> float:
    """
    Rand index adjusted for chance.

    The Rand Index computes a similarity measure between two clusterings
    by considering all pairs of samples and counting pairs that are
    assigned in the same or different clusters in the predicted and
    true clusterings.

    The adjusted Rand index is corrected for chance: the expected value of
    the ARI for two random clusterings is 0, and 1 when the clusterings are
    identical.

    Parameters:
        labels_true: Ground truth class labels.
        labels_pred: Cluster labels to evaluate.

    Returns:
        Adjusted Rand index, between -1.0 and 1.0.
    """
    labels_true_np = np.array(labels_true).flatten()
    labels_pred_np = np.array(labels_pred).flatten()

    if len(labels_true_np) != len(labels_pred_np):
        raise ValueError("labels_true and labels_pred have different lengths")

    # Compute contingency table
    classes = np.unique(labels_true_np)
    clusters = np.unique(labels_pred_np)

    n_classes = len(classes)
    n_clusters = len(clusters)

    contingency = np.zeros((n_classes, n_clusters), dtype=np.int64)
    for i, c in enumerate(classes):
        for j, k in enumerate(clusters):
            contingency[i, j] = np.sum((labels_true_np == c) & (labels_pred_np == k))

    # Compute sum of combinations for each row/column
    sum_comb_c = np.sum([n * (n - 1) / 2 for n in np.sum(contingency, axis=1)])
    sum_comb_k = np.sum([n * (n - 1) / 2 for n in np.sum(contingency, axis=0)])
    sum_comb = np.sum([n * (n - 1) / 2 for n in contingency.flatten()])

    n_samples = len(labels_true_np)
    total_comb = n_samples * (n_samples - 1) / 2

    # Compute ARI
    expected_index = sum_comb_c * sum_comb_k / total_comb if total_comb > 0 else 0
    max_index = (sum_comb_c + sum_comb_k) / 2

    if max_index == expected_index:
        return 1.0
    else:
        ari = (sum_comb - expected_index) / (max_index - expected_index)
        return float(ari)


def normalized_mutual_info_score(
    labels_true: Union[mx.array, np.array],
    labels_pred: Union[mx.array, np.array]
) -> float:
    """
    Normalized Mutual Information between two clusterings.

    Normalized Mutual Information (NMI) is an normalization of the
    Mutual Information score to scale the results between 0 (no mutual
    information) and 1 (perfect correlation).

    Parameters:
        labels_true: Ground truth class labels.
        labels_pred: Cluster labels to evaluate.

    Returns:
        Normalized Mutual Information score, between 0.0 and 1.0.
    """
    labels_true_np = np.array(labels_true).flatten()
    labels_pred_np = np.array(labels_pred).flatten()

    if len(labels_true_np) != len(labels_pred_np):
        raise ValueError("labels_true and labels_pred have different lengths")

    # Compute contingency table
    classes = np.unique(labels_true_np)
    clusters = np.unique(labels_pred_np)

    n_classes = len(classes)
    n_clusters = len(clusters)

    if n_classes == 1 and n_clusters == 1:
        return 1.0
    if n_classes == 0 or n_clusters == 0:
        return 0.0

    contingency = np.zeros((n_classes, n_clusters), dtype=np.int64)
    for i, c in enumerate(classes):
        for j, k in enumerate(clusters):
            contingency[i, j] = np.sum((labels_true_np == c) & (labels_pred_np == k))

    contingency = contingency / np.sum(contingency) if np.sum(contingency) > 0 else contingency

    # Compute marginals
    p_i = np.sum(contingency, axis=1)
    p_j = np.sum(contingency, axis=0)

    # Compute entropies
    h_true = -np.sum(p_i * np.log2(p_i + 1e-10))
    h_pred = -np.sum(p_j * np.log2(p_j + 1e-10))

    # Compute mutual information
    mi = 0.0
    for i in range(n_classes):
        for j in range(n_clusters):
            if contingency[i, j] > 0:
                mi += contingency[i, j] * np.log2(contingency[i, j] / (p_i[i] * p_j[j] + 1e-10) + 1e-10)

    # Normalize
    if h_true + h_pred == 0:
        return 0.0
    nmi = 2 * mi / (h_true + h_pred)
    return float(nmi)


def silhouette_score(
    X: Union[mx.array, np.array],
    labels: Union[mx.array, np.array],
    metric: str = 'euclidean'
) -> float:
    """
    Compute the mean Silhouette Coefficient of all samples.

    The Silhouette Coefficient is calculated using the mean intra-cluster
    distance (a) and the mean nearest-cluster distance (b) for each sample.
    The Silhouette Coefficient for a sample is (b - a) / max(a, b).

    The best value is 1 and the worst value is -1. Values near 0 indicate
    overlapping clusters. Negative values generally indicate that a sample has
    been assigned to the wrong cluster, as a different cluster is more similar.

    Parameters:
        X: Feature array of shape (n_samples, n_features).
        labels: Predicted cluster labels for each sample.
        metric: The metric to use when calculating distance between instances.

    Returns:
        Mean Silhouette Coefficient for all samples.
    """
    X_np = np.array(X)
    labels_np = np.array(labels).flatten()

    if len(X_np) != len(labels_np):
        raise ValueError("X and labels have different lengths")

    unique_labels = np.unique(labels_np)
    n_labels = len(unique_labels)

    if n_labels <= 1 or n_labels >= len(X_np):
        raise ValueError("Number of labels must be between 2 and n_samples - 1")

    # Compute pairwise distances
    dist_matrix = scipy_cdist(X_np, X_np, metric=metric)
    n_samples = len(X_np)

    silhouette_vals = np.zeros(n_samples, dtype=np.float64)

    for i in range(n_samples):
        label_i = labels_np[i]

        # a: mean distance to all other points in the same cluster
        same_cluster_mask = labels_np == label_i
        same_cluster_mask[i] = False  # exclude self
        if np.sum(same_cluster_mask) == 0:
            a = 0.0
        else:
            a = np.mean(dist_matrix[i, same_cluster_mask])

        # b: mean distance to all points in the nearest other cluster
        min_other_dist = np.inf
        for label in unique_labels:
            if label == label_i:
                continue
            other_cluster_mask = labels_np == label
            if np.sum(other_cluster_mask) == 0:
                continue
            mean_dist = np.mean(dist_matrix[i, other_cluster_mask])
            if mean_dist < min_other_dist:
                min_other_dist = mean_dist
        b = min_other_dist

        # Compute silhouette for this sample
        if a < b:
            silhouette_vals[i] = (b - a) / b if b > 0 else 0.0
        elif a > b:
            silhouette_vals[i] = (b - a) / a if a > 0 else 0.0
        else:
            silhouette_vals[i] = 0.0

    return float(np.mean(silhouette_vals))
