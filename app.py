"""
Streamlit web demo for the Fake News Detection model.

Usage:
    streamlit run app.py

Requires fake_news_model.joblib and tfidf_vectorizer.joblib
(produced by train_model.py) to be in the same folder.
"""

import streamlit as st
import joblib
import pandas as pd
from train_model import clean_text

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

# ---------------------------------------------------------------------------
# Light custom styling
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #9aa0a6;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    .result-real {
        background-color: rgba(16, 185, 129, 0.12);
        border: 1px solid rgba(16, 185, 129, 0.4);
        border-radius: 10px;
        padding: 1rem 1.2rem;
        font-size: 1.2rem;
        font-weight: 600;
    }
    .result-fake {
        background-color: rgba(239, 68, 68, 0.12);
        border: 1px solid rgba(239, 68, 68, 0.4);
        border-radius: 10px;
        padding: 1rem 1.2rem;
        font-size: 1.2rem;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_artifacts():
    model = joblib.load("fake_news_model.joblib")
    vectorizer = joblib.load("tfidf_vectorizer.joblib")
    return model, vectorizer


EXAMPLES = {
    "Sensational / likely fake": "Doctors hate this one weird trick that cures any disease overnight, no pills needed",
    "Neutral / likely real": "The Federal Reserve raised interest rates by a quarter point on Wednesday, citing inflation data",
    "Conspiracy-style": "Government secretly controls the weather using hidden machines, whistleblower claims",
}

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("About")
    st.write(
        "TF-IDF + Logistic Regression model trained on 40,000+ labeled "
        "news articles from the Kaggle *Fake and Real News Dataset*."
    )
    st.divider()
    st.subheader("Try an example")
    for label, example_text in EXAMPLES.items():
        if st.button(label, use_container_width=True):
            st.session_state["text_input"] = example_text
    st.divider()
    with st.expander("⚠️ Known limitations"):
        st.write(
            "- Trained on 2016-2017 news, so current/2026 topics may confuse it.\n"
            "- Learns word patterns and writing style, not facts — it's not a "
            "fact-checker.\n"
            "- Real articles in training data mostly came from one wire "
            "service, which may bias predictions toward that style."
        )

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.markdown('<div class="main-title">📰 Fake News Detector</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Paste a headline or article below to check whether '
    "it looks real or fake, based on patterns learned from labeled news data.</div>",
    unsafe_allow_html=True,
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

if "text_input" not in st.session_state:
    st.session_state["text_input"] = ""

text_input = st.text_area(
    "Article text or headline",
    height=180,
    placeholder="Paste text here, or pick an example from the sidebar...",
    key="text_input",
)

analyze = st.button("Analyze", type="primary", use_container_width=True)

if analyze:
    if not text_input.strip():
        st.warning("Please enter some text first.")
    else:
        with st.spinner("Analyzing..."):
            cleaned = clean_text(text_input)
            vec = vectorizer.transform([cleaned])
            pred = model.predict(vec)[0]
            prob = model.predict_proba(vec)[0]
            confidence = prob[pred]

        st.write("")
        if pred == 1:
            st.markdown(
                f'<div class="result-real">✅ Predicted: REAL &nbsp; '
                f"({confidence:.1%} confidence)</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="result-fake">🚩 Predicted: FAKE &nbsp; '
                f"({confidence:.1%} confidence)</div>",
                unsafe_allow_html=True,
            )

        st.write("")
        chart_df = pd.DataFrame(
            {"Probability": [prob[0], prob[1]]}, index=["Fake", "Real"]
        )
        st.bar_chart(chart_df, height=200)

        with st.expander("See cleaned/preprocessed text"):
            st.write(cleaned if cleaned else "*(empty after preprocessing)*")

st.divider()
st.caption(
    "Model: Logistic Regression + TF-IDF (unigrams & bigrams, top 5,000 features). "
    "Trained on the Kaggle 'Fake and Real News Dataset'. Not a fact-checker — "
    "see limitations in the sidebar."
)
