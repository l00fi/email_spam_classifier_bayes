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


from bs4 import BeautifulSoup
import json
import nltk
from nltk.tokenize import word_tokenize


def clean_email_content(raw_html):
    soup = BeautifulSoup(raw_html, "html.parser")
    return soup.get_text(separator=" ")

with open("data/emails.json", "r", encoding="utf-8") as file:
    emails = json.load(file)

email_texts = [clean_email_content(email.get("Body", "")) for email in emails]

nltk.download("punkt")
email_tokens = [word_tokenize(text) for text in email_texts]