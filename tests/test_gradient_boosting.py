import sys
sys.path.insert(0, '/Users/jw/mlx')

import numpy as np
import mlx.core as mx
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingClassifier as SKGradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor as SKGradientBoostingRegressor

from mlx_ml.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

print("=== Testing Gradient Boosting Models ===\n")

# Test 1: Gradient Boosting Classifier on Breast Cancer dataset
print("Test 1: Gradient Boosting Classifier (Breast Cancer Dataset)")
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Our model
gb_clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gb_clf.fit(X_train, y_train)
y_pred = gb_clf.predict(X_test)
acc = accuracy_score(y_test, np.array(y_pred))
print(f"Our model accuracy: {acc:.4f}")

# Sklearn model
sk_gb_clf = SKGradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
sk_gb_clf.fit(X_train, y_train)
sk_y_pred = sk_gb_clf.predict(X_test)
sk_acc = accuracy_score(y_test, sk_y_pred)
print(f"Sklearn model accuracy: {sk_acc:.4f}")

assert abs(acc - sk_acc) < 0.03, "Accuracy difference too large"
assert acc >= 0.95, "Accuracy should be >= 0.95 on breast cancer dataset"
print("Test 1 passed\n")

# Test 2: Gradient Boosting Classifier with subsampling
print("Test 2: Gradient Boosting Classifier with subsampling")
gb_clf_sub = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, subsample=0.8, random_state=42)
gb_clf_sub.fit(X_train, y_train)
y_pred_sub = gb_clf_sub.predict(X_test)
acc_sub = accuracy_score(y_test, np.array(y_pred_sub))
print(f"Our model with subsampling accuracy: {acc_sub:.4f}")

sk_gb_clf_sub = SKGradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, subsample=0.8, random_state=42)
sk_gb_clf_sub.fit(X_train, y_train)
sk_y_pred_sub = sk_gb_clf_sub.predict(X_test)
sk_acc_sub = accuracy_score(y_test, sk_y_pred_sub)
print(f"Sklearn model with subsampling accuracy: {sk_acc_sub:.4f}")

assert abs(acc_sub - sk_acc_sub) < 0.04, "Accuracy difference too large"
print("Test 2 passed\n")

# Test 3: Gradient Boosting Regressor on Diabetes dataset
print("Test 3: Gradient Boosting Regressor (Diabetes Dataset)")
diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

gb_reg = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gb_reg.fit(X_train, y_train)
y_pred = gb_reg.predict(X_test)
mse = mean_squared_error(y_test, np.array(y_pred))
r2 = r2_score(y_test, np.array(y_pred))
print(f"Our model: MSE={mse:.4f}, R2={r2:.4f}")

sk_gb_reg = SKGradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
sk_gb_reg.fit(X_train, y_train)
sk_y_pred = sk_gb_reg.predict(X_test)
sk_mse = mean_squared_error(y_test, sk_y_pred)
sk_r2 = r2_score(y_test, sk_y_pred)
print(f"Sklearn model: MSE={sk_mse:.4f}, R2={sk_r2:.4f}")

assert abs(r2 - sk_r2) < 0.1, "R2 score difference too large"
print("Test 3 passed\n")

# Test 4: Gradient Boosting Regressor with absolute error loss
print("Test 4: Gradient Boosting Regressor (absolute_error loss)")
gb_reg_mae = GradientBoostingRegressor(n_estimators=150, learning_rate=0.08, max_depth=3, loss='absolute_error', random_state=42)
gb_reg_mae.fit(X_train, y_train)
y_pred_mae = gb_reg_mae.predict(X_test)
mae = np.mean(np.abs(y_test - np.array(y_pred_mae)))
r2_mae = r2_score(y_test, np.array(y_pred_mae))
print(f"Absolute error loss - MAE: {mae:.4f}, R2: {r2_mae:.4f}")

assert r2_mae > 0.1, "R2 should be > 0.1 for absolute error loss"
assert mae < 60, "MAE should be < 60 for diabetes dataset"
print("Test 4 passed\n")

# Test 5: Predict probabilities for classification
print("Test 5: Predict probabilities")
# Use breast cancer test data for classification probabilities
_, X_test_clf, _, _ = train_test_split(cancer.data, cancer.target, test_size=0.2, random_state=42)
y_proba = gb_clf.predict_proba(X_test_clf[:5])
print(f"Sample probabilities shape: {y_proba.shape}")
print(f"Sample probabilities:\n{np.array2string(np.array(y_proba), precision=4)}")
assert y_proba.shape == (5, 2), "Probability shape should be (n_samples, n_classes)"
assert np.allclose(np.sum(np.array(y_proba), axis=1), 1.0), "Probabilities should sum to 1"
print("Test 5 passed\n")

# Test 6: Huber loss for regression
print("Test 6: Gradient Boosting Regressor with Huber loss")
gb_reg_huber = GradientBoostingRegressor(n_estimators=80, learning_rate=0.1, max_depth=3, loss='huber', random_state=42)
gb_reg_huber.fit(X_train, y_train)
y_pred_huber = gb_reg_huber.predict(X_test)
mse_huber = mean_squared_error(y_test, np.array(y_pred_huber))
r2_huber = r2_score(y_test, np.array(y_pred_huber))
print(f"Huber loss - MSE: {mse_huber:.4f}, R2: {r2_huber:.4f}")
assert r2_huber > 0.05, "R2 should be > 0.05 for Huber loss"
print("Test 6 passed\n")

print("✅ All Gradient Boosting tests passed!")
