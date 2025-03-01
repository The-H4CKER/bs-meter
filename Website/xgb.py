from joblib import load
from xgboost import XGBClassifier

# Define stop words
vectorizer = load("../models/tfidf_model.joblib")
model = XGBClassifier(eval_metric="logloss")

# Load the trained model from the saved file
loaded_model = XGBClassifier()
loaded_model.load_model("../models/model.json")


# Function to preprocess a single string
def preprocess_text(text):
    # Use the same vectorizer that was fitted on the training data
    return vectorizer.transform([text])


# Test with a single text
single_text = "Example text to classify using the trained XGBoost model"
processed_text = preprocess_text(single_text)

# Make a prediction on the new text
prediction = loaded_model.predict(processed_text)
