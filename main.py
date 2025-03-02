# from model import NormalNaiveBayes
# # --------------------------------------------------
# from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split

# X, y = load_iris(return_X_y=True)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

# model = NormalNaiveBayes().fit(X_train, y_train)
# predict = model.predict(X_test)

# accuracy = sum(y_test == predict)/len(y_test)
# print(accuracy)

import make_dataset
from model import SpamClassifier
# --------------------------------------------------
import pandas as pd
import numpy as np

PATH = ["data/spam.json", "data/emails.json"]
CSV_PATH = "data.csv"
YEAR = 5

# make_dataset.make_dataset(PATH, CSV_PATH, YEAR)

data = pd.read_csv("data.csv", delimiter=';', index_col=0)
data = data.dropna()

model = SpamClassifier().fit(data)

X, y = data["text"].values, data["sign"].values
y_predict = []
for x in X:
    y_predict.append(model.predict(x.split()))

accuracy = sum(np.array(y) == np.array(y_predict))/y.shape[0]
print(f"Accuracy = {accuracy}")
