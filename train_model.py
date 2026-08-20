"""
Fake News Detection using NLP & Machine Learning
Pipeline: Text Preprocessing -> TF-IDF Vectorization -> Logistic Regression

Dataset expected: Kaggle "Fake and Real News Dataset" (Clement Bisaillon)
    https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
    Files: Fake.csv, True.csv  (place them in the `data/` folder)
"""

import re
import string
import joblib
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# ---------------------------------------------------------------------------
# 0. One-time NLTK downloads
# ---------------------------------------------------------------------------
for pkg in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
    try:
        nltk.data.find(pkg)
    except LookupError:
        nltk.download(pkg, quiet=True)

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


# ---------------------------------------------------------------------------
# 1. Load & label data
# ---------------------------------------------------------------------------
def load_data(fake_path="data/Fake.csv", true_path="data/True.csv"):
    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)

    fake_df["label"] = 0  # 0 = fake
    true_df["label"] = 1  # 1 = real

    df = pd.concat([fake_df, true_df], axis=0, ignore_index=True)

    # Combine title + text for a richer feature set (common in this dataset)
    if "title" in df.columns and "text" in df.columns:
        df["content"] = df["title"].fillna("") + " " + df["text"].fillna("")
    else:
        df["content"] = df["text"].fillna("")

    df = df[["content", "label"]].dropna()
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle
    return df


# ---------------------------------------------------------------------------
# 2. Text preprocessing
# ---------------------------------------------------------------------------
URL_RE = re.compile(r"https?://\S+|www\.\S+")
HTML_RE = re.compile(r"<.*?>")
NON_ALPHA_RE = re.compile(r"[^a-zA-Z\s]")


def clean_text(text: str) -> str:
    text = text.lower()
    text = URL_RE.sub(" ", text)
    text = HTML_RE.sub(" ", text)
    text = NON_ALPHA_RE.sub(" ", text)          # strip numbers/punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]
    tokens = [LEMMATIZER.lemmatize(t) for t in tokens]

    return " ".join(tokens)


# ---------------------------------------------------------------------------
# 3. Train / evaluate
# ---------------------------------------------------------------------------
def main():
    print("Loading data...")
    df = load_data()
    print(f"Total articles: {len(df)}")

    print("Cleaning text (this can take a few minutes on 40k+ articles)...")
    df["clean_content"] = df["content"].apply(clean_text)

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_content"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    print("Vectorizing with TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print("Training Logistic Regression model...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_tfidf, y_train)

    print("Evaluating...")
    y_pred = model.predict(X_test_tfidf)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\nAccuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Fake", "Real"]))

    print("Saving model + vectorizer...")
    joblib.dump(model, "fake_news_model.joblib")
    joblib.dump(vectorizer, "tfidf_vectorizer.joblib")
    print("Done. Saved: fake_news_model.joblib, tfidf_vectorizer.joblib")


if __name__ == "__main__":
    main()