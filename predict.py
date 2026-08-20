"""
Load the saved model + vectorizer and predict on new text.

Usage:
    python predict.py "Some news article text here..."
"""

import sys
import joblib
from train_model import clean_text


def load_artifacts():
    model = joblib.load("fake_news_model.joblib")
    vectorizer = joblib.load("tfidf_vectorizer.joblib")
    return model, vectorizer


def predict(text: str, model, vectorizer) -> str:
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0]
    label = "REAL" if pred == 1 else "FAKE"
    confidence = prob[pred]
    return f"{label} (confidence: {confidence:.2%})"


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python predict.py "article text here"')
        sys.exit(1)

    article_text = " ".join(sys.argv[1:])
    model, vectorizer = load_artifacts()
    result = predict(article_text, model, vectorizer)
    print(result)