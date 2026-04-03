"""
Manifold learning module for mlx-ml, compatible with scikit-learn API.
"""

from .tsne import TSNE
from .umap import UMAP

__all__ = ['TSNE', 'UMAP']
