import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.utils import shuffle

# GOAL: 97%+ Accuracy with an Augmented Training Set

data = np.load('mnist.npz', allow_pickle=True)
X = data['X']
y = data['y']
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]

total_imgs = []
total_results = []
for i in range(60000):
    img = X_train[i].reshape(28, 28)
    total_imgs.append(img.reshape(-1))
    total_results.append(y_train[i])
    total_imgs.append(np.roll(img, 1, axis=0).reshape(-1))
    total_results.append(y_train[i])
    total_imgs.append(np.roll(img, -1, axis=0).reshape(-1))
    total_results.append(y_train[i])
    total_imgs.append(np.roll(img, 1, axis=1).reshape(-1))
    total_results.append(y_train[i])
    total_imgs.append(np.roll(img, -1, axis=1).reshape(-1))
    total_results.append(y_train[i])

X_train_augmented = np.array(total_imgs)
y_train_augmented = np.array(total_results)

random_forest = RandomForestClassifier()
# random_forest.fit(X_train_augmented, y_train_augmented)
# print(random_forest.score(X_test, y_test))

X_shuff, y_shuff = shuffle(X_train_augmented, y_train_augmented, random_state=12)
X_cv = X_shuff[:50000]
y_cv = y_shuff[:50000]

grid = GridSearchCV(random_forest, param_grid = {
    'n_estimators': [100, 150, 200],
    'max_depth': [None, 25, 30]
}, cv=3, n_jobs=-1)
grid.fit(X_cv, y_cv)

best_rf_classifier = RandomForestClassifier(
    n_estimators=grid.best_params_['n_estimators'],
    max_depth=grid.best_params_['max_depth'],
    n_jobs=-1,
    random_state=42
)
best_rf_classifier.fit(X_train_augmented, y_train_augmented)
print(best_rf_classifier.score(X_test, y_test))
