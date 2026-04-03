import sys
sys.path.insert(0, '/Users/jw/mlx')

import numpy as np
import mlx.core as mx
from sklearn.datasets import load_iris, load_digits, fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB as SKGaussianNB
from sklearn.naive_bayes import MultinomialNB as SKMultinomialNB
from sklearn.naive_bayes import BernoulliNB as SKBernoulliNB

from mlx_ml.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

print("=== Testing Naive Bayes Models ===\n")

# Test 1: GaussianNB on Iris dataset
print("Test 1: GaussianNB (Iris Dataset)")
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Our model
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)
acc = accuracy_score(y_test, np.array(y_pred))
print(f"Our GaussianNB accuracy: {acc:.4f}")

# Sklearn model
sk_gnb = SKGaussianNB()
sk_gnb.fit(X_train, y_train)
sk_y_pred = sk_gnb.predict(X_test)
sk_acc = accuracy_score(y_test, sk_y_pred)
print(f"Sklearn GaussianNB accuracy: {sk_acc:.4f}")

assert abs(acc - sk_acc) < 1e-3, "Accuracy difference too large"
print("Test 1 passed\n")

# Test 2: GaussianNB with custom priors
print("Test 2: GaussianNB with custom priors")
priors = np.array([0.2, 0.3, 0.5])
gnb_prior = GaussianNB(priors=priors)
gnb_prior.fit(X_train, y_train)
y_pred_prior = gnb_prior.predict(X_test)
acc_prior = accuracy_score(y_test, np.array(y_pred_prior))
print(f"GaussianNB with custom priors accuracy: {acc_prior:.4f}")
print("Test 2 passed\n")

# Test 3: MultinomialNB on Digits dataset
print("Test 3: MultinomialNB (Digits Dataset)")
digits = load_digits()
X, y = digits.data, digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Our model
mnb = MultinomialNB(alpha=1.0)
mnb.fit(X_train, y_train)
y_pred = mnb.predict(X_test)
acc = accuracy_score(y_test, np.array(y_pred))
print(f"Our MultinomialNB accuracy: {acc:.4f}")

# Sklearn model
sk_mnb = SKMultinomialNB(alpha=1.0)
sk_mnb.fit(X_train, y_train)
sk_y_pred = sk_mnb.predict(X_test)
sk_acc = accuracy_score(y_test, sk_y_pred)
print(f"Sklearn MultinomialNB accuracy: {sk_acc:.4f}")

assert abs(acc - sk_acc) < 0.01, "Accuracy difference too large"
print("Test 3 passed\n")

# Test 4: BernoulliNB on binarized Digits
print("Test 4: BernoulliNB (Binarized Digits Dataset)")
# Binarize the data
X_bin = (X > 7).astype(np.float64)  # Threshold at midpoint of 0-16 range
X_train_bin, X_test_bin, y_train, y_test = train_test_split(X_bin, y, test_size=0.3, random_state=42)

# Our model
bnb = BernoulliNB(alpha=1.0, binarize=None)  # Already binarized
bnb.fit(X_train_bin, y_train)
y_pred = bnb.predict(X_test_bin)
acc = accuracy_score(y_test, np.array(y_pred))
print(f"Our BernoulliNB accuracy: {acc:.4f}")

# Sklearn model
sk_bnb = SKBernoulliNB(alpha=1.0, binarize=None)
sk_bnb.fit(X_train_bin, y_train)
sk_y_pred = sk_bnb.predict(X_test_bin)
sk_acc = accuracy_score(y_test, sk_y_pred)
print(f"Sklearn BernoulliNB accuracy: {sk_acc:.4f}")

assert abs(acc - sk_acc) < 0.01, "Accuracy difference too large"
print("Test 4 passed\n")

# Test 5: BernoulliNB with auto-binarization
print("Test 5: BernoulliNB with auto-binarization")
bnb_auto = BernoulliNB(alpha=1.0, binarize=7.0)  # Auto binarize with threshold 7
bnb_auto.fit(X_train, y_train)  # Use non-binarized input
y_pred_auto = bnb_auto.predict(X_test)
acc_auto = accuracy_score(y_test, np.array(y_pred_auto))
print(f"BernoulliNB with auto-binarization accuracy: {acc_auto:.4f}")
assert abs(acc_auto - acc) < 1e-3, "Auto-binarization should give same result as manual binarization"
print("Test 5 passed\n")

# Test 6: Predict probabilities
print("Test 6: Predict probabilities")
# Use Iris test data for GaussianNB probabilities
_, X_test_iris, _, _ = train_test_split(iris.data, iris.target, test_size=0.3, random_state=42)
y_proba = gnb.predict_proba(X_test_iris[:5])
print(f"Probabilities shape: {y_proba.shape}")
print(f"Sample probabilities:\n{np.array2string(np.array(y_proba), precision=4)}")
assert y_proba.shape == (5, 3), "Probabilities shape should be (n_samples, n_classes)"
assert np.allclose(np.sum(np.array(y_proba), axis=1), 1.0), "Probabilities should sum to 1"
print("Test 6 passed\n")

print("✅ All Naive Bayes tests passed!")
