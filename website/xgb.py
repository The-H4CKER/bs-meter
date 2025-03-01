import os
from pandas import DataFrame
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier


def score(text):
    model = XGBClassifier()
    model.load_model("../models/model.xgb")
    custom_stop_word_list = list(ENGLISH_STOP_WORDS.union({"figure", "fig"}))
    vectorizer = TfidfVectorizer(stop_words=custom_stop_word_list)
    vectorizer.fit_transform(text)
    return model.predict(text)
