from model import NormalNaiveBayes
# --------------------------------------------------
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

model = NormalNaiveBayes().fit(X_train, y_train)
predict = model.predict(X_test)

accuracy = sum(y_test == predict)/len(y_test)
print(accuracy)
