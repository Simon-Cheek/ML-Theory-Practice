from sklearn.datasets import fetch_openml
import numpy as np
# import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, precision_recall_curve

# 70000 images of 28x28 pixels representing drawings of numbers
# mnist = fetch_openml('mnist_784', as_frame=False)
# X,y = mnist.data, mnist.target
# np.savez('mnist.npz', X=X, y=y)

data = np.load('mnist.npz', allow_pickle=True)
X = data['X']
y = data['y']

# Already split into test and training (first 60000 and last 10000)
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

# Binary Classifier (5 or Not 5) - Stochastic Gradient Descent
y_train_5 = (y_train == '5')
y_test_5 = (y_test == '5')

# Train using Stochastic Gradient Descent
sgd_classifier = SGDClassifier(random_state=27)
sgd_classifier.fit(X_train, y_train_5)

# QUALITY METRICS for Default Training
y_train_pred = cross_val_predict(sgd_classifier, X_train, y_train_5, cv=3)
cm = confusion_matrix(y_train_5, y_train_pred)
print(cm)

true_positive = cm[1, 1]
false_positive = cm[0, 1]
false_negative = cm[1, 0]
true_negative = cm[0, 0]

precision = true_positive / (true_positive + false_positive)
recall = true_positive / (true_positive + false_negative)
print(f"Precision: {precision}, Recall: {recall}")

# Use SKlearn for F1Score
precision_sklearn = precision_score(y_train_5, y_train_pred)
recall_sklearn = recall_score(y_train_5, y_train_pred)
f1_score = f1_score(y_train_5, y_train_pred)
print(f"SKLEARN: Precision: {precision_sklearn}, Recall: {recall_sklearn}, F1 Score: {f1_score}")

# Examine Precision / Recall Curve
y_scores = cross_val_predict(sgd_classifier, X_train, y_train_5, cv=3, method='decision_function')
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)
# plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
# plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
# plt.vlines(thresholds, 0, 1.0, "k", "dotted", label="threshold")
# plt.show()

# Find Correct Threshold for 90% Precision
precision_90_idx = (precisions > 0.9).argmax()
threshold_for_90_precision = precisions[precision_90_idx]

# Now Try to Get a Model with 90% Precision
y_scores_for_90_precision = (y_scores >= threshold_for_90_precision)
print(y_scores_for_90_precision)
