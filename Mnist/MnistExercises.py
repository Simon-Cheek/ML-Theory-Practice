import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# GOAL: 97% Accuracy on Multiclass Test Set (Identifying all 9 digits hand drawn)
# Hints: KNeighborsClassifier, Tune Weights and N_Neighbors (using grid search)

data = np.load('mnist.npz', allow_pickle=True)
X = data['X']
y = data['y']
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]

# Default KNeighborsClassifier Analysis
knn_classifier = KNeighborsClassifier()
knn_classifier.fit(X_train, y_train)
y_pred = knn_classifier.predict(X_test)
accuracy = knn_classifier.score(X_test, y_test)
print("Initial Model Accuracy:", accuracy)
# print(accuracy) # 96.8% Accuracy, CLOSE

# Run GridSearchCV to tune weight and n_neighbors
paramsMap = {
    "n_neighbors": [2, 4, 6],
    "weights": ["uniform", "distance"],
}
grid = GridSearchCV(
    knn_classifier,
    paramsMap,
    scoring='accuracy',
    cv=3
)
# grid.fit(X_train, y_train)
# print("Best Params: ")
# print(grid.best_params_)
# print("Best Score: ")
# print(grid.best_score_)
# print("Best Estimator: ")
# print(grid.best_estimator_)
# print("CV Results: ")
# print(grid.cv_results_)

# Best Params: Weight = Distance, N_Neighbors = 4
# Manual Retrain
# best_knn_classifier = KNeighborsClassifier(n_neighbors=4, weights="distance")
# best_knn_classifier.fit(X_train, y_train)
# accuracy = best_knn_classifier.score(X_test, y_test)
# print("Best Model Accuracy:", accuracy)

# Best Model Trained Automatically with Scaled Data
pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", KNeighborsClassifier()),
])
newParamsMap = {
    "classifier__n_neighbors": [2, 4, 6, 8],
    "classifier__weights": ["uniform", "distance"],
}
newGrid = GridSearchCV(pipeline, newParamsMap, scoring='accuracy', cv=3, n_jobs=-1)
newGrid.fit(X_train, y_train)
accuracy = newGrid.score(X_test, y_test)
# Accuracy ends up being LOWER here, turns out scaling HURTS
print("Best Model Accuracy:", accuracy)
print("Best Params:", newGrid.best_params_)

