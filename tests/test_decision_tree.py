import sys
sys.path.insert(0, '/Users/jw/mlx')

import numpy as np
import mlx.core as mx
from sklearn.datasets import load_iris, load_breast_cancer, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeClassifier as SKDecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor as SKDecisionTreeRegressor

from mlx_ml.tree import DecisionTreeClassifier, DecisionTreeRegressor

print("=== Testing Decision Tree Models ===\n")

# Test 1: Decision Tree Classifier on Iris dataset
print("Test 1: Decision Tree Classifier (Iris Dataset)")
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Our model
dt_clf = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)
dt_clf.fit(X_train, y_train)
y_pred = dt_clf.predict(X_test)
acc = accuracy_score(y_test, np.array(y_pred))
print(f"Our model accuracy: {acc:.4f}")

# Sklearn model
sk_dt_clf = SKDecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)
sk_dt_clf.fit(X_train, y_train)
sk_y_pred = sk_dt_clf.predict(X_test)
sk_acc = accuracy_score(y_test, sk_y_pred)
print(f"Sklearn model accuracy: {sk_acc:.4f}")

assert abs(acc - sk_acc) < 0.02, "Accuracy difference too large"
print("Test 1 passed\n")

# Test 2: Decision Tree Classifier on Breast Cancer dataset
print("Test 2: Decision Tree Classifier (Breast Cancer Dataset)")
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dt_clf = DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_split=5, random_state=42)
dt_clf.fit(X_train, y_train)
y_pred = dt_clf.predict(X_test)
acc = accuracy_score(y_test, np.array(y_pred))
print(f"Our model accuracy (entropy criterion): {acc:.4f}")

sk_dt_clf = SKDecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_split=5, random_state=42)
sk_dt_clf.fit(X_train, y_train)
sk_y_pred = sk_dt_clf.predict(X_test)
sk_acc = accuracy_score(y_test, sk_y_pred)
print(f"Sklearn model accuracy (entropy criterion): {sk_acc:.4f}")

assert acc >= 0.92, "Accuracy should be >= 0.92 on breast cancer dataset"
print("Test 2 passed\n")

# Test 3: Decision Tree Regressor on Diabetes dataset
print("Test 3: Decision Tree Regressor (Diabetes Dataset)")
from sklearn.datasets import load_diabetes
diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dt_reg = DecisionTreeRegressor(criterion='squared_error', max_depth=8, min_samples_leaf=5, random_state=42)
dt_reg.fit(X_train, y_train)
y_pred = dt_reg.predict(X_test)
mse = mean_squared_error(y_test, np.array(y_pred))
r2 = r2_score(y_test, np.array(y_pred))
print(f"Our model: MSE={mse:.4f}, R2={r2:.4f}")

sk_dt_reg = SKDecisionTreeRegressor(criterion='squared_error', max_depth=8, min_samples_leaf=5, random_state=42)
sk_dt_reg.fit(X_train, y_train)
sk_y_pred = sk_dt_reg.predict(X_test)
sk_mse = mean_squared_error(y_test, sk_y_pred)
sk_r2 = r2_score(y_test, sk_y_pred)
print(f"Sklearn model: MSE={sk_mse:.4f}, R2={sk_r2:.4f}")

assert abs(r2 - sk_r2) < 0.1, "R2 score difference too large"
print("Test 3 passed\n")

# Test 4: Decision Tree Regressor with absolute error criterion
print("Test 4: Decision Tree Regressor (MAE criterion)")
dt_reg_mae = DecisionTreeRegressor(criterion='absolute_error', max_depth=6, random_state=42)
dt_reg_mae.fit(X_train, y_train)
y_pred_mae = dt_reg_mae.predict(X_test)
mae = np.mean(np.abs(y_test - np.array(y_pred_mae)))
print(f"MAE criterion - Mean Absolute Error: {mae:.4f}")
print(f"MAE criterion - R2 score: {r2_score(y_test, np.array(y_pred_mae)):.4f}")
assert mae < 60, "MAE should be < 60 for diabetes dataset"
print("Test 4 passed\n")

# Test 5: Overfitting control with max_depth (using classification data)
print("Test 5: Overfitting control with max_depth")
X_clf, y_clf = load_breast_cancer(return_X_y=True)
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)
for depth in [1, 3, 5, 10, None]:
    dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
    dt.fit(X_train_clf, y_train_clf)
    train_acc = accuracy_score(y_train_clf, np.array(dt.predict(X_train_clf)))
    test_acc = accuracy_score(y_test_clf, np.array(dt.predict(X_test_clf)))
    print(f"max_depth={depth}: Train acc={train_acc:.4f}, Test acc={test_acc:.4f}")
# Deep trees should have higher train accuracy but may overfit
assert train_acc > 0.99, "Unlimited depth tree should have near perfect train accuracy"
print("Test 5 passed\n")

print("✅ All Decision Tree tests passed!")
