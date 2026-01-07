from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier

iris = load_iris(as_frame=True)
X_iris = iris.data[["petal length (cm)", "petal width (cm)"]].values
y_iris = iris.target
X_train, X_test, y_train, y_test = train_test_split(X_iris, y_iris, test_size=0.2, random_state=42)

# Traditional Decision Tree
dt_clf = DecisionTreeClassifier(max_depth=2)
dt_clf.fit(X_train, y_train)
print(dt_clf.score(X_test, y_test))

# Now with Bagging
bag_clf = BaggingClassifier(DecisionTreeClassifier(max_depth=2), n_estimators=100, n_jobs=-1)
bag_clf.fit(X_train, y_train)
print(bag_clf.score(X_test, y_test))

