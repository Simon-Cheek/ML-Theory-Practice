import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.svm import SVC
from matplotlib import pyplot as plt

# 70000 images of 28x28 pixels representing drawings of numbers
# mnist = fetch_openml('mnist_784', as_frame=False)
# X,y = mnist.data, mnist.target
# np.savez('mnist.npz', X=X, y=y)

data = np.load('mnist.npz', allow_pickle=True)
X = data['X']
y = data['y']

svm_classifier = SVC(random_state=38)
svm_classifier.fit(X[:2000], y[:2000])

# print(svm_classifier.predict([X[2001], X[2002], X[2003]]))

print(cross_val_score(svm_classifier, X[:2000], y[:2000], cv=3))

y_train_pred = cross_val_predict(svm_classifier, X[:2000], y[:2000], cv=3)
# cm = confusion_matrix(y[:2000], y_train_pred)
# print(cm)
ConfusionMatrixDisplay.from_predictions(y[:2000], y_train_pred)
plt.show()

