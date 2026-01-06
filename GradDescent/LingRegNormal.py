import numpy as np
from sklearn.preprocessing import add_dummy_feature

rng = np.random.default_rng(seed=41)
m = 200  # number of instances
X = 2 * rng.random((m, 1))  # column vector
y = 4 + 3 * X + rng.standard_normal((m, 1))  # column vector
print(X.shape, y.shape)
X_b = add_dummy_feature(X)
print(X_b)
fit = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y # Normal Equation
print(fit)

# Predict using new model
X_test = np.array([[0], [1], [2], [3]])
X_test_b = add_dummy_feature(X_test)
print(X_test_b)
print(X_test_b.shape)
print(fit.shape)
print(X_test_b @ fit)
