import os
from pandas import DataFrame
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from transformers import RobertaTokenizer

nature_dir = "dataset/nature_articles/"
chatgpt_dir = "dataset/chatgpt_articles/"

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")


def tokenize(text):
    return tokenizer(text["text"], truncation=True, padding="max_length", max_length=512)


def load_articles_from_folder(folder, label):
    articles = []
    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        with open(filepath, "r", encoding="utf-8") as file:
            text = file.read()
            articles.append((text, label))
    return articles


nature_articles = load_articles_from_folder(nature_dir, 0)
chatgpt_articles = load_articles_from_folder(chatgpt_dir, 1)


data = nature_articles + chatgpt_articles
df = DataFrame(data, columns=["text", "label"])


custom_stop_word_list = list(ENGLISH_STOP_WORDS.union({"figure", "fig"}))
vectorizer = TfidfVectorizer(stop_words=custom_stop_word_list)


X = vectorizer.fit_transform(df["text"])
y = df["label"]
