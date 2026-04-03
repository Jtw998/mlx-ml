import sys
sys.path.insert(0, '/Users/jw/mlx')

import numpy as np
import mlx.core as mx
from sklearn.datasets import load_iris, load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier as SKRandomForestClassifier
from sklearn.ensemble import RandomForestRegressor as SKRandomForestRegressor

from mlx_ml.ensemble import RandomForestClassifier, RandomForestRegressor

print("=== Testing Random Forest Models ===\n")

# Test 1: Random Forest Classifier on Iris dataset
print("Test 1: Random Forest Classifier (Iris Dataset)")
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Our model
rf_clf = RandomForestClassifier(n_estimators=10, criterion='gini', max_depth=3, random_state=42, max_features='sqrt')
rf_clf.fit(X_train, y_train)
y_pred = rf_clf.predict(X_test)
acc = accuracy_score(y_test, np.array(y_pred))
print(f"Our model accuracy: {acc:.4f}")

# Sklearn model
sk_rf_clf = SKRandomForestClassifier(n_estimators=10, criterion='gini', max_depth=3, random_state=42, max_features='sqrt', bootstrap=True)
sk_rf_clf.fit(X_train, y_train)
sk_y_pred = sk_rf_clf.predict(X_test)
sk_acc = accuracy_score(y_test, sk_y_pred)
print(f"Sklearn model accuracy: {sk_acc:.4f}")

assert abs(acc - sk_acc) < 0.03, "Accuracy difference too large"
print("Test 1 passed\n")

# Test 2: Random Forest Classifier on Breast Cancer dataset
print("Test 2: Random Forest Classifier (Breast Cancer Dataset)")
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_clf = RandomForestClassifier(n_estimators=20, criterion='entropy', max_depth=5, min_samples_split=5, random_state=42)
rf_clf.fit(X_train, y_train)
y_pred = rf_clf.predict(X_test)
acc = accuracy_score(y_test, np.array(y_pred))
print(f"Our model accuracy (entropy criterion): {acc:.4f}")

sk_rf_clf = SKRandomForestClassifier(n_estimators=20, criterion='entropy', max_depth=5, min_samples_split=5, random_state=42)
sk_rf_clf.fit(X_train, y_train)
sk_y_pred = sk_rf_clf.predict(X_test)
sk_acc = accuracy_score(y_test, sk_y_pred)
print(f"Sklearn model accuracy (entropy criterion): {sk_acc:.4f}")

assert acc >= 0.95, "Accuracy should be >= 0.95 on breast cancer dataset"
print("Test 2 passed\n")

# Test 3: Random Forest Regressor on Diabetes dataset
print("Test 3: Random Forest Regressor (Diabetes Dataset)")
diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_reg = RandomForestRegressor(n_estimators=30, criterion='squared_error', max_depth=8, min_samples_leaf=5, random_state=42, max_features=0.5)
rf_reg.fit(X_train, y_train)
y_pred = rf_reg.predict(X_test)
mse = mean_squared_error(y_test, np.array(y_pred))
r2 = r2_score(y_test, np.array(y_pred))
print(f"Our model: MSE={mse:.4f}, R2={r2:.4f}")

sk_rf_reg = SKRandomForestRegressor(n_estimators=30, criterion='squared_error', max_depth=8, min_samples_leaf=5, random_state=42, max_features=0.5)
sk_rf_reg.fit(X_train, y_train)
sk_y_pred = sk_rf_reg.predict(X_test)
sk_mse = mean_squared_error(y_test, sk_y_pred)
sk_r2 = r2_score(y_test, sk_y_pred)
print(f"Sklearn model: MSE={sk_mse:.4f}, R2={sk_r2:.4f}")

assert abs(r2 - sk_r2) < 0.15, "R2 score difference too large"
print("Test 3 passed\n")

# Test 4: Random Forest Regressor with absolute error criterion
print("Test 4: Random Forest Regressor (MAE criterion)")
rf_reg_mae = RandomForestRegressor(n_estimators=20, criterion='absolute_error', max_depth=6, random_state=42)
rf_reg_mae.fit(X_train, y_train)
y_pred_mae = rf_reg_mae.predict(X_test)
mae = np.mean(np.abs(y_test - np.array(y_pred_mae)))
print(f"MAE criterion - Mean Absolute Error: {mae:.4f}")
print(f"MAE criterion - R2 score: {r2_score(y_test, np.array(y_pred_mae)):.4f}")
assert mae < 55, "MAE should be < 55 for diabetes dataset"
print("Test 4 passed\n")

# Test 5: OOB score functionality
print("Test 5: OOB Score functionality")
# Use breast cancer dataset for classification OOB test
X_clf, y_clf = load_breast_cancer(return_X_y=True)
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)
rf_clf_oob = RandomForestClassifier(n_estimators=50, oob_score=True, random_state=42)
rf_clf_oob.fit(X_train_clf, y_train_clf)
print(f"OOB Score: {rf_clf_oob.oob_score_:.4f}")
assert rf_clf_oob.oob_score_ is not None and rf_clf_oob.oob_score_ > 0.8, "OOB score should be > 0.8"
print("Test 5 passed\n")

# Test 6: Predict probabilities
print("Test 6: Predict probabilities")
# Use the same breast cancer test data that rf_clf was trained on
y_proba = rf_clf.predict_proba(X_test_clf[:5])
print(f"Sample probabilities shape: {y_proba.shape}")
print(f"Sample probabilities:\n{np.array2string(np.array(y_proba), precision=4)}")
assert y_proba.shape == (5, 2), "Probability shape should be (n_samples, n_classes)"
assert np.allclose(np.sum(np.array(y_proba), axis=1), 1.0), "Probabilities should sum to 1"
print("Test 6 passed\n")

print("✅ All Random Forest tests passed!")
