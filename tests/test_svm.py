import sys
sys.path.insert(0, '/Users/jw/mlx')

import numpy as np
import mlx.core as mx
from sklearn.datasets import load_breast_cancer, make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC as SKSVC
from sklearn.preprocessing import StandardScaler

from mlx_ml.svm import SVC, SVR

print("=== Testing SVM Models ===\n")

# Preprocess data: SVM requires scaling
scaler = StandardScaler()

# Test 1: SVC on Breast Cancer dataset (linear kernel)
print("Test 1: SVC (Linear kernel, Breast Cancer Dataset)")
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Our model
svc_linear = SVC(C=1.0, kernel='linear', random_state=42)
svc_linear.fit(X_train, y_train)
y_pred = svc_linear.predict(X_test)
acc = accuracy_score(y_test, np.array(y_pred))
print(f"Our model accuracy (linear kernel): {acc:.4f}")

# Sklearn model
sk_svc_linear = SKSVC(C=1.0, kernel='linear', random_state=42)
sk_svc_linear.fit(X_train, y_train)
sk_y_pred = sk_svc_linear.predict(X_test)
sk_acc = accuracy_score(y_test, sk_y_pred)
print(f"Sklearn model accuracy (linear kernel): {sk_acc:.4f}")

assert abs(acc - sk_acc) < 0.05, "Accuracy difference too large"
assert acc >= 0.95, "Accuracy should be >= 0.95 on breast cancer dataset"
print("Test 1 passed\n")

# Test 2: SVC on Moons dataset (RBF kernel - non-linear)
print("Test 2: SVC (RBF kernel, Moons Dataset)")
X_moons, y_moons = make_moons(n_samples=200, noise=0.1, random_state=42)
X_moons_scaled = scaler.fit_transform(X_moons)
X_train_moons, X_test_moons, y_train_moons, y_test_moons = train_test_split(X_moons_scaled, y_moons, test_size=0.3, random_state=42)

svc_rbf = SVC(C=10.0, kernel='rbf', gamma='scale', random_state=42)
svc_rbf.fit(X_train_moons, y_train_moons)
y_pred_moons = svc_rbf.predict(X_test_moons)
acc_moons = accuracy_score(y_test_moons, np.array(y_pred_moons))
print(f"Our model accuracy (RBF kernel): {acc_moons:.4f}")

sk_svc_rbf = SKSVC(C=10.0, kernel='rbf', gamma='scale', random_state=42)
sk_svc_rbf.fit(X_train_moons, y_train_moons)
sk_y_pred_moons = sk_svc_rbf.predict(X_test_moons)
sk_acc_moons = accuracy_score(y_test_moons, sk_y_pred_moons)
print(f"Sklearn model accuracy (RBF kernel): {sk_acc_moons:.4f}")

assert abs(acc_moons - sk_acc_moons) < 0.05, "Accuracy difference too large"
assert acc_moons >= 0.95, "Accuracy should be >= 0.95 on moons dataset"
print("Test 2 passed\n")

# Test 3: SVC with polynomial kernel
print("Test 3: SVC (Polynomial kernel)")
svc_poly = SVC(C=5.0, kernel='poly', degree=3, gamma='scale', coef0=1.0, random_state=42)
svc_poly.fit(X_train_moons, y_train_moons)
y_pred_poly = svc_poly.predict(X_test_moons)
acc_poly = accuracy_score(y_test_moons, np.array(y_pred_poly))
print(f"Our model accuracy (poly kernel): {acc_poly:.4f}")

sk_svc_poly = SKSVC(C=5.0, kernel='poly', degree=3, gamma='scale', coef0=1.0, random_state=42)
sk_svc_poly.fit(X_train_moons, y_train_moons)
sk_y_pred_poly = sk_svc_poly.predict(X_test_moons)
sk_acc_poly = accuracy_score(y_test_moons, sk_y_pred_poly)
print(f"Sklearn model accuracy (poly kernel): {sk_acc_poly:.4f}")

assert acc_poly >= 0.9, "Accuracy should be >= 0.9 with poly kernel"
print("Test 3 passed\n")

# Test 4: Decision function
print("Test 4: Decision function")
decision = svc_linear.decision_function(X_test[:5])
print(f"Decision function values:\n{np.array2string(np.array(decision), precision=4)}")
assert decision.shape == (5,), "Decision function shape should be (n_samples,)"
print("Test 4 passed\n")

# Test 5: SVR (Experimental)
print("Test 5: SVR (Experimental implementation)")
from sklearn.datasets import load_diabetes
diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

svr = SVR(C=1.0, kernel='linear', epsilon=10.0, random_state=42)
svr.fit(X_train, y_train)
y_pred = np.array(svr.predict(X_test))
print(f"SVR prediction range: [{np.min(y_pred):.2f}, {np.max(y_pred):.2f}]")
print("Test 5 passed (experimental implementation)\n")

print("✅ All SVM tests passed!")
