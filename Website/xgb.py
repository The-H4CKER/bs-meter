import os
from pandas import DataFrame
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier


def load_model():
    model = XGBClassifier()
    model.load_model("../models/model.xgb")
    return model


load_model()


def prepare_data():

    # extract TF-IDF features
    pass


def score():
    # return BS-score
    pass


model = load_model()

nature_dir = "../dataset/nature_articles/"
chatgpt_dir = "../dataset/chatgpt_articles/"


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

# Combine into a single dataset
data = nature_articles + chatgpt_articles
df = DataFrame(data, columns=["text", "label"])

# Define stop words
custom_stop_word_list = list(ENGLISH_STOP_WORDS.union({"figure", "fig"}))
vectorizer = TfidfVectorizer(stop_words=custom_stop_word_list)

# Convert text data to TF-IDF features
X = vectorizer.fit_transform(df["text"])

print(X)


y = df["label"]

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Make predictions
y_pred = model.predict(X_test)

# Print accuracy
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
