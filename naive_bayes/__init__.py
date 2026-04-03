"""
Naive Bayes module for mlx-ml, compatible with scikit-learn API.
"""

from .naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

__all__ = ['GaussianNB', 'MultinomialNB', 'BernoulliNB']
