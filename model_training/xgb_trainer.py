import os
from pandas import DataFrame
from re import sub
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import joblib
from re import sub


def preprocessor(text):
    figure_pattern = r"\b[Ff]ig\. ?[\d]+[a-zA-Z0-9,–-]*"
    text = sub(figure_pattern, "", text)
    return text.strip()


nature_dir = "../dataset/nature_articles/"
chatgpt_dir = "../dataset/chatgpt_articles/"


def load_articles_from_folder(folder, label):
    articles = []
    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        with open(filepath, "r", encoding="utf-8") as file:
            text = file.read()
            clean = sub(r"[^a-zA-Z0-9 ]", "", text)
            articles.append((clean, label))
    return articles


nature_articles = load_articles_from_folder(nature_dir, 0)
chatgpt_articles = load_articles_from_folder(chatgpt_dir, 1)


# Combine into a single dataset
data = nature_articles + chatgpt_articles
df = DataFrame(data, columns=["text", "label"])


# Define stop words
# stop_words = list(ENGLISH_STOP_WORDS.union({"figure", "fig", "title", "paper", "section"}))
vectorizer = TfidfVectorizer(
    preprocessor=preprocessor, stop_words=list(ENGLISH_STOP_WORDS), max_df=0.9, min_df=5
)

# Convert text data to TF-IDF features
X = vectorizer.fit_transform(df["text"])
y = df["label"]

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train an XGBoost model
model = XGBClassifier(eval_metric="logloss")
model.fit(X_train, y_train)

model.save_model("../models/XGBoost/model.json")
joblib.dump(vectorizer, "../models/XGBoost/tf-idf_model.joblib")

# Make predictions
y_pred = model.predict(X_test)
