import sys
sys.path.insert(0, '/Users/jw/mlx')

import numpy as np
import mlx.core as mx
from sklearn.datasets import load_iris, load_diabetes, make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score as sk_accuracy_score,
    confusion_matrix as sk_confusion_matrix,
    precision_recall_fscore_support as sk_precision_recall_fscore_support,
    precision_score as sk_precision_score,
    recall_score as sk_recall_score,
    f1_score as sk_f1_score,
    roc_auc_score as sk_roc_auc_score,
    mean_squared_error as sk_mean_squared_error,
    mean_absolute_error as sk_mean_absolute_error,
    mean_absolute_percentage_error as sk_mean_absolute_percentage_error,
    r2_score as sk_r2_score,
    adjusted_rand_score as sk_adjusted_rand_score,
    normalized_mutual_info_score as sk_normalized_mutual_info_score,
    silhouette_score as sk_silhouette_score
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans as SKKMeans

from mlx_ml.metrics import *
from mlx_ml.ensemble import RandomForestClassifier as MLXRandomForestClassifier

print("=== Testing Metrics Module ===\n")

# Test classification metrics
print("=== Classification Metrics ===")
# Generate classification test data
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
clf = MLXRandomForestClassifier(n_estimators=10, random_state=42)
clf.fit(X_train, y_train)
y_pred = np.array(clf.predict(X_test))
y_proba = np.array(clf.predict_proba(X_test))

# Test accuracy_score
acc = accuracy_score(y_test, y_pred)
sk_acc = sk_accuracy_score(y_test, y_pred)
print(f"accuracy_score: ours={acc:.4f}, sklearn={sk_acc:.4f}")
assert abs(acc - sk_acc) < 1e-6, "accuracy_score mismatch"
print("✓ accuracy_score passed")

# Test confusion_matrix
cm = confusion_matrix(y_test, y_pred)
sk_cm = sk_confusion_matrix(y_test, y_pred)
print(f"\nconfusion_matrix:\n{cm}")
print(f"Sklearn confusion_matrix:\n{sk_cm}")
assert np.array_equal(cm, sk_cm), "confusion_matrix mismatch"
print("✓ confusion_matrix passed")

# Test precision, recall, f1 with different averages
for average in ['macro', 'micro', 'weighted', None]:
    p, r, f, s = precision_recall_fscore_support(y_test, y_pred, average=average)
    sk_p, sk_r, sk_f, sk_s = sk_precision_recall_fscore_support(y_test, y_pred, average=average)

    if average is None:
        assert np.allclose(p, sk_p, atol=1e-4), f"precision mismatch for average={average}"
        assert np.allclose(r, sk_r, atol=1e-4), f"recall mismatch for average={average}"
        assert np.allclose(f, sk_f, atol=1e-4), f"f1 mismatch for average={average}"
    else:
        assert abs(p - sk_p) < 1e-4, f"precision mismatch for average={average}"
        assert abs(r - sk_r) < 1e-4, f"recall mismatch for average={average}"
        assert abs(f - sk_f) < 1e-4, f"f1 mismatch for average={average}"
    print(f"\n✓ precision_recall_fscore_support (average={average}) passed")

# Test roc_auc_score (binary classification only)
y_bin = (y == 2).astype(int)  # Binary classification task
X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(X, y_bin, test_size=0.3, random_state=42)
clf_bin = MLXRandomForestClassifier(n_estimators=10, random_state=42)
clf_bin.fit(X_train_bin, y_train_bin)
y_proba_bin = np.array(clf_bin.predict_proba(X_test_bin))[:, 1]

roc_auc = roc_auc_score(y_test_bin, y_proba_bin)
sk_roc_auc = sk_roc_auc_score(y_test_bin, y_proba_bin)
print(f"\nroc_auc_score: ours={roc_auc:.4f}, sklearn={sk_roc_auc:.4f}")
assert abs(roc_auc - sk_roc_auc) < 1e-4, "roc_auc_score mismatch"
print("✓ roc_auc_score passed")

# Test regression metrics
print("\n\n=== Regression Metrics ===")
diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
reg = RandomForestRegressor(n_estimators=10, random_state=42)
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)

# Test mean_squared_error
mse = mean_squared_error(y_test, y_pred)
sk_mse = sk_mean_squared_error(y_test, y_pred)
print(f"mean_squared_error: ours={mse:.4f}, sklearn={sk_mse:.4f}")
assert abs(mse - sk_mse) < 1e-6, "mean_squared_error mismatch"
print("✓ mean_squared_error passed")

# Test RMSE
rmse = mean_squared_error(y_test, y_pred, squared=False)
sk_rmse = np.sqrt(sk_mean_squared_error(y_test, y_pred))
print(f"root_mean_squared_error: ours={rmse:.4f}, sklearn={sk_rmse:.4f}")
assert abs(rmse - sk_rmse) < 1e-6, "root_mean_squared_error mismatch"
print("✓ root_mean_squared_error passed")

# Test mean_absolute_error
mae = mean_absolute_error(y_test, y_pred)
sk_mae = sk_mean_absolute_error(y_test, y_pred)
print(f"mean_absolute_error: ours={mae:.4f}, sklearn={sk_mae:.4f}")
assert abs(mae - sk_mae) < 1e-6, "mean_absolute_error mismatch"
print("✓ mean_absolute_error passed")

# Test r2_score
r2 = r2_score(y_test, y_pred)
sk_r2 = sk_r2_score(y_test, y_pred)
print(f"r2_score: ours={r2:.4f}, sklearn={sk_r2:.4f}")
assert abs(r2 - sk_r2) < 1e-6, "r2_score mismatch"
print("✓ r2_score passed")

# Test MAPE
# Remove zero values for MAPE test
mask = y_test != 0
y_test_nonzero = y_test[mask]
y_pred_nonzero = y_pred[mask]
mape = mean_absolute_percentage_error(y_test_nonzero, y_pred_nonzero)
sk_mape = sk_mean_absolute_percentage_error(y_test_nonzero, y_pred_nonzero)
print(f"mean_absolute_percentage_error: ours={mape:.4f}, sklearn={sk_mape:.4f}")
assert abs(mape - sk_mape) < 1e-6, "mean_absolute_percentage_error mismatch"
print("✓ mean_absolute_percentage_error passed")

# Test clustering metrics
print("\n\n=== Clustering Metrics ===")
X_blobs, y_blobs = make_blobs(n_samples=200, centers=4, random_state=42, cluster_std=1.0)
kmeans = SKKMeans(n_clusters=4, random_state=42)
y_pred_clust = kmeans.fit_predict(X_blobs)

# Test adjusted_rand_score
ari = adjusted_rand_score(y_blobs, y_pred_clust)
sk_ari = sk_adjusted_rand_score(y_blobs, y_pred_clust)
print(f"adjusted_rand_score: ours={ari:.4f}, sklearn={sk_ari:.4f}")
assert abs(ari - sk_ari) < 1e-4, "adjusted_rand_score mismatch"
print("✓ adjusted_rand_score passed")

# Test normalized_mutual_info_score
nmi = normalized_mutual_info_score(y_blobs, y_pred_clust)
sk_nmi = sk_normalized_mutual_info_score(y_blobs, y_pred_clust)
print(f"normalized_mutual_info_score: ours={nmi:.4f}, sklearn={sk_nmi:.4f}")
assert abs(nmi - sk_nmi) < 1e-4, "normalized_mutual_info_score mismatch"
print("✓ normalized_mutual_info_score passed")

# Test silhouette_score
sil = silhouette_score(X_blobs, y_pred_clust)
sk_sil = sk_silhouette_score(X_blobs, y_pred_clust)
print(f"silhouette_score: ours={sil:.4f}, sklearn={sk_sil:.4f}")
assert abs(sil - sk_sil) < 1e-4, "silhouette_score mismatch"
print("✓ silhouette_score passed")

print("\n✅ All metrics tests passed!")
