import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb

data = pd.read_csv("dataset_test.csv")

custom_stop_word_list = list(ENGLISH_STOP_WORDS.union({"figure", "fig"}))
vectorizer = TfidfVectorizer(stop_words=custom_stop_word_list)

X = vectorizer.fit_transform(data["text"])
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = xgb.XGBClassifier(eval_metric="logloss")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
