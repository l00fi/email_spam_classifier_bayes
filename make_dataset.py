import copy
import re
import json
import pandas as pd
from bs4 import BeautifulSoup
import email.header
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download("punkt")
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')


def clean_email_content(raw_html):
    soup = BeautifulSoup(raw_html, "html.parser")
    return soup.get_text(separator=" ")

def clean_tokens(tokens):
    stop_words = set(stopwords.words('russian'))
    lemmatizer = WordNetLemmatizer()

    cleaned_tokens = []
    for token in tokens:
        token = token.lower()

        token = re.sub(r'[^а-яА-Я0-9]', '', token)
        token = re.sub(r'\b\d+\b', '', token)
        token = re.sub(r'(http[s]?://\S+|www\.\S+)', '', token)
        token = re.sub(r'\S+\.(jpg|png|gif|pdf|docx|xls|zip|rar)', '', token)

        if token and token not in stop_words:
            cleaned_tokens.append(lemmatizer.lemmatize(token))

    return cleaned_tokens

def create_csv(emails, path):
    texts = [(
                email.header.decode_header(e['X-Gmail-Labels'])[0][0].decode('utf-8').split(',')[0], # получаю метки (Входящее, Спам)
                clean_tokens(word_tokenize(clean_email_content(e.get("Body", "")))) # получаю очищенный от мусора набор слов  
            ) for e in emails]

    all_cleaned_tokens = []  
    for text in texts:
        all_cleaned_tokens += text[1] # складываю эти слова в один список

    unique_tokens = list(set(all_cleaned_tokens)) # с помощью множества избавляюсь от повторений

    pre_dataframe = {
        "word": unique_tokens,
        "spam_count": [1]*len(unique_tokens),
        "ham_count": [1]*len(unique_tokens)
        } 
    
    for text in texts:
        for word in text[1]:
            if word in pre_dataframe["word"]:
                if "Спам" in text[0]:
                    pre_dataframe["spam_count"][pre_dataframe["word"].index(word)] += 1
                if "Спам" not in text[0]:
                    pre_dataframe["ham_count"][pre_dataframe["word"].index(word)] += 1
    
    data = pd.DataFrame(pre_dataframe)
    data.to_csv(f"{path}", index=False, sep=";")

def make_dataset(PATH, CSV_PATH, YEAR):
    all_emails = []
    for path in PATH:
        with open(path, "r", encoding="utf-8") as file:
            emails = json.load(file)

        correct_emails = []
        for e in emails:
            try:
                if re.search(r'\d{1,2}\s[A-Za-z]{3}\s\d{4}', e.get("Received", '')).group()[-1] == f'{YEAR}': # получаю дату и отбираю только ту почту, чей год равен заданному
                    correct_emails.append(e)
            except:
                continue
        
        all_emails += correct_emails

    create_csv(all_emails, CSV_PATH)