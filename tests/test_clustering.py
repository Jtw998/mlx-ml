import sys
sys.path.insert(0, '/Users/jw/mlx')

import numpy as np
import mlx.core as mx
from sklearn.datasets import make_moons, make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.cluster import DBSCAN as SKDBSCAN
from sklearn.mixture import GaussianMixture as SKGaussianMixture

from mlx_ml.cluster import DBSCAN, GaussianMixture, KMeans

print("=== Testing Advanced Clustering Algorithms ===\n")

scaler = StandardScaler()

# Test 1: DBSCAN on Moons dataset
print("Test 1: DBSCAN (Moons Dataset)")
X_moons, y_moons = make_moons(n_samples=200, noise=0.05, random_state=42)
X_moons_scaled = scaler.fit_transform(X_moons)

# Our DBSCAN
dbscan = DBSCAN(eps=0.3, min_samples=5)
labels = np.array(dbscan.fit_predict(X_moons_scaled))
ari = adjusted_rand_score(y_moons, labels)
n_clusters = len(np.unique(labels[labels != -1]))
print(f"Our DBSCAN: ARI={ari:.4f}, Found {n_clusters} clusters, {np.sum(labels == -1)} noise points")

# Sklearn DBSCAN
sk_dbscan = SKDBSCAN(eps=0.3, min_samples=5)
sk_labels = sk_dbscan.fit_predict(X_moons_scaled)
sk_ari = adjusted_rand_score(y_moons, sk_labels)
sk_n_clusters = len(np.unique(sk_labels[sk_labels != -1]))
print(f"Sklearn DBSCAN: ARI={sk_ari:.4f}, Found {sk_n_clusters} clusters, {np.sum(sk_labels == -1)} noise points")

assert ari >= 0.9, "ARI should be >= 0.9 for DBSCAN on moons dataset"
assert n_clusters == 2, "DBSCAN should find 2 clusters on moons dataset"
print("Test 1 passed\n")

# Test 2: DBSCAN with different distance metrics
print("Test 2: DBSCAN with Manhattan metric")
dbscan_manhattan = DBSCAN(eps=0.5, min_samples=5, metric='manhattan')
labels_manhattan = np.array(dbscan_manhattan.fit_predict(X_moons_scaled))
ari_manhattan = adjusted_rand_score(y_moons, labels_manhattan)
print(f"DBSCAN (Manhattan): ARI={ari_manhattan:.4f}")
assert ari_manhattan >= 0.8, "ARI should be >= 0.8 with Manhattan metric"
print("Test 2 passed\n")

# Test 3: Gaussian Mixture on Blobs dataset
print("Test 3: GaussianMixture (Blobs Dataset)")
X_blobs, y_blobs = make_blobs(n_samples=300, centers=4, random_state=42, cluster_std=1.0)
X_blobs_scaled = scaler.fit_transform(X_blobs)

# Our GMM
gmm = GaussianMixture(n_components=4, covariance_type='full', random_state=42)
gmm.fit(X_blobs_scaled)
labels_gmm = np.array(gmm.predict(X_blobs_scaled))
ari_gmm = adjusted_rand_score(y_blobs, labels_gmm)
sil_gmm = silhouette_score(X_blobs_scaled, labels_gmm)
print(f"Our GMM: ARI={ari_gmm:.4f}, Silhouette={sil_gmm:.4f}")

# Sklearn GMM
sk_gmm = SKGaussianMixture(n_components=4, covariance_type='full', random_state=42)
sk_gmm.fit(X_blobs_scaled)
sk_labels_gmm = sk_gmm.predict(X_blobs_scaled)
sk_ari_gmm = adjusted_rand_score(y_blobs, sk_labels_gmm)
sk_sil_gmm = silhouette_score(X_blobs_scaled, sk_labels_gmm)
print(f"Sklearn GMM: ARI={sk_ari_gmm:.4f}, Silhouette={sk_sil_gmm:.4f}")

assert ari_gmm >= 0.7, "ARI should be >= 0.7 for GMM on blobs dataset"
print("Test 3 passed\n")

# Test 4: GMM with diagonal covariance
print("Test 4: GaussianMixture with diagonal covariance")
gmm_diag = GaussianMixture(n_components=4, covariance_type='diag', random_state=42)
gmm_diag.fit(X_blobs_scaled)
labels_diag = np.array(gmm_diag.predict(X_blobs_scaled))
ari_diag = adjusted_rand_score(y_blobs, labels_diag)
print(f"GMM (diag covariance): ARI={ari_diag:.4f}")
assert ari_diag >= 0.6, "ARI should be >= 0.6 with diagonal covariance"
print("Test 4 passed\n")

# Test 5: GMM predict probabilities
print("Test 5: GMM predict probabilities")
proba = gmm.predict_proba(X_blobs_scaled[:5])
print(f"Probabilities shape: {proba.shape}")
print(f"Sample probabilities:\n{np.array2string(np.array(proba), precision=4)}")
assert proba.shape == (5, 4), "Probabilities shape should be (n_samples, n_components)"
assert np.allclose(np.sum(np.array(proba), axis=1), 1.0), "Probabilities should sum to 1"
print("Test 5 passed\n")

# Test 6: Compare with KMeans
print("Test 6: Comparison with KMeans")
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(X_blobs_scaled)
labels_kmeans = np.array(kmeans.predict(X_blobs_scaled))
ari_kmeans = adjusted_rand_score(y_blobs, labels_kmeans)
print(f"KMeans ARI: {ari_kmeans:.4f}")
print(f"GMM ARI: {ari_gmm:.4f}")
print("Test 6 passed\n")

print("✅ All advanced clustering tests passed!")
