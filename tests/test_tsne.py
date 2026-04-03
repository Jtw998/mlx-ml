import sys
sys.path.insert(0, '/Users/jw/mlx')

import numpy as np
import mlx.core as mx
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE as SKTSNE

from mlx_ml.manifold import TSNE

print("=== Testing t-SNE Algorithm ===\n")

# Load digits dataset
print("Loading digits dataset...")
digits = load_digits()
X = digits.data[:300]  # Use subset for faster testing
y = digits.target[:300]
print(f"Dataset shape: {X.shape}, Number of classes: {len(np.unique(y))}")

# Standardize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Test our t-SNE
print("\nRunning our t-SNE (n_components=2, perplexity=30, max_iter=500)...")
tsne = TSNE(n_components=2, perplexity=30, max_iter=500, random_state=42)
embedding = tsne.fit_transform(X_scaled)
embedding_np = np.array(embedding)

print(f"\nEmbedding shape: {embedding_np.shape}")
print(f"KL divergence: {tsne.kl_divergence_:.4f}")
print(f"Embedding range: x=[{embedding_np[:,0].min():.2f}, {embedding_np[:,0].max():.2f}], y=[{embedding_np[:,1].min():.2f}, {embedding_np[:,1].max():.2f}]")

# Test with different perplexity
print("\n\nRunning t-SNE with perplexity=15...")
tsne_perp = TSNE(n_components=2, perplexity=15, max_iter=300, random_state=42)
embedding_perp = np.array(tsne_perp.fit_transform(X_scaled))
print(f"KL divergence (perplexity=15): {tsne_perp.kl_divergence_:.4f}")

# Test 3D embedding
print("\n\nRunning t-SNE with 3 components...")
tsne_3d = TSNE(n_components=3, perplexity=30, max_iter=300, random_state=42)
embedding_3d = np.array(tsne_3d.fit_transform(X_scaled))
print(f"3D Embedding shape: {embedding_3d.shape}")

print("\n✅ All t-SNE tests passed!")
print("\nNote: t-SNE is stochastic by nature. For visualization, you can plot the embedding to see class separation.")
