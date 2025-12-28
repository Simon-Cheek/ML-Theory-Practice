from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, precision_recall_curve

# 70000 images of 28x28 pixels representing drawings of numbers
mnist = fetch_openml('mnist_784', as_frame=False)

X,y = mnist.data, mnist.target

# Already split into test and training (first 60000 and last 10000)
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

# Binary Classifier (5 or Not 5) - Stochastic Gradient Descent
y_train_5 = (y_train == '5')
y_test_5 = (y_test == '5')
sgd_classifier = SGDClassifier(random_state=27)
sgd_classifier.fit(X_train, y_train_5)

# ~95% accuracy, but consider that the positive case is rare (~10%)
# K - fold cross validation on training set
print(cross_val_score(sgd_classifier, X_train, y_train_5, cv=3, scoring="accuracy"))

# Confusion Matrix of actual vs predicted values
y_train_pred = cross_val_predict(sgd_classifier, X_train, y_train_5, cv=3)
cm = confusion_matrix(y_train_5, y_train_pred)
print(cm)

# Calculate Precision: TP / TP + FP
# Confidence that a positive diagnosis is authoritative
print(precision_score(y_train_5, y_train_pred))

# Calculate Recall: TP / TP + FN
# Confidence that the positive instances are being correctly identified as positive
print(recall_score(y_train_5, y_train_pred))

# Calculate F1 Score: Mixes Precision and Recall Scores
# Favors similar precision and recall scores
# Consider which metric is more important to your model
print(f1_score(y_train_5, y_train_pred))

# Manually Determine Threshold for Positive vs Negative
y_scores = cross_val_predict(sgd_classifier, X_train, y_train_5, cv=3, method="decision_function")

# Gets all Precisions and Recalls for Each Possible Threshold
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)


