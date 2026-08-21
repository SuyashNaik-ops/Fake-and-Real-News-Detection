"""
Streamlit web demo for the Fake News Detection model.

Usage:
    streamlit run app.py

Requires fake_news_model.joblib and tfidf_vectorizer.joblib
(produced by train_model.py) to be in the same folder.
"""

import streamlit as st
import joblib
from train_model import clean_text

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")


@st.cache_resource
def load_artifacts():
    model = joblib.load("fake_news_model.joblib")
    vectorizer = joblib.load("tfidf_vectorizer.joblib")
    return model, vectorizer


st.title("📰 Fake News Detector")
st.write(
    "Paste a news headline or article below. The model uses TF-IDF + "
    "Logistic Regression trained on 40,000+ labeled news articles to "
    "predict whether it's **real** or **fake**."
)

try:
    model, vectorizer = load_artifacts()
except FileNotFoundError:
    st.error(
        "Model files not found. Run `python train_model.py` first to "
        "generate fake_news_model.joblib and tfidf_vectorizer.joblib, "
        "then place them in this folder."
    )
    st.stop()

text_input = st.text_area("Article text or headline", height=200, placeholder="Paste text here...")

if st.button("Analyze", type="primary"):
    if not text_input.strip():
        st.warning("Please enter some text first.")
    else:
        cleaned = clean_text(text_input)
        vec = vectorizer.transform([cleaned])
        pred = model.predict(vec)[0]
        prob = model.predict_proba(vec)[0]
        confidence = prob[pred]

        if pred == 1:
            st.success(f"✅ Predicted: **REAL** ({confidence:.1%} confidence)")
        else:
            st.error(f"🚩 Predicted: **FAKE** ({confidence:.1%} confidence)")

        with st.expander("See cleaned/preprocessed text"):
            st.write(cleaned if cleaned else "*(empty after preprocessing)*")

        st.caption(
            f"Fake probability: {prob[0]:.1%}  |  Real probability: {prob[1]:.1%}"
        )

st.divider()
st.caption(
    "Model: Logistic Regression + TF-IDF (unigrams & bigrams, top 5,000 features). "
    "Trained on the Kaggle 'Fake and Real News Dataset'."
)
