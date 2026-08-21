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
# Styling
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Animated gradient background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e, #1a1a2e);
        background-size: 300% 300%;
        animation: gradientShift 18s ease infinite;
    }

    @keyframes gradientShift {
        0%   { background-position: 0% 50%; }
        50%  { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* Soft glowing orbs behind content */
    .stApp::before {
        content: "";
        position: fixed;
        top: -10%;
        left: -10%;
        width: 40vw;
        height: 40vw;
        background: radial-gradient(circle, rgba(99,102,241,0.25) 0%, rgba(0,0,0,0) 70%);
        pointer-events: none;
        z-index: 0;
    }
    .stApp::after {
        content: "";
        position: fixed;
        bottom: -15%;
        right: -10%;
        width: 45vw;
        height: 45vw;
        background: radial-gradient(circle, rgba(236,72,153,0.18) 0%, rgba(0,0,0,0) 70%);
        pointer-events: none;
        z-index: 0;
    }

    .block-container {
        position: relative;
        z-index: 1;
        max-width: 760px;
        padding-top: 2.5rem;
    }

    /* Header */
    .hero {
        text-align: center;
        margin-bottom: 2rem;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(99, 102, 241, 0.15);
        border: 1px solid rgba(99, 102, 241, 0.4);
        color: #a5b4fc;
        padding: 0.3rem 0.9rem;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 0.9rem;
    }
    .main-title {
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(90deg, #ffffff, #c7d2fe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.4rem;
        line-height: 1.1;
    }
    .subtitle {
        color: #a1a1aa;
        font-size: 1.02rem;
        max-width: 480px;
        margin: 0 auto;
        line-height: 1.5;
    }

    /* Glassmorphism card wrapping the input */
    .glass-card {
        background: rgba(255, 255, 255, 0.04);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 18px;
        padding: 1.6rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        margin-bottom: 1.2rem;
    }

    .stTextArea textarea {
        background: rgba(0, 0, 0, 0.25) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        color: #f4f4f5 !important;
        font-size: 0.98rem !important;
    }
    .stTextArea textarea:focus {
        border-color: rgba(139, 92, 246, 0.6) !important;
        box-shadow: 0 0 0 3px rgba(139, 92, 246, 0.15) !important;
    }

    .stButton > button {
        background: linear-gradient(90deg, #6366f1, #8b5cf6);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.7rem 1.2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
        box-shadow: 0 4px 14px rgba(99, 102, 241, 0.35);
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.5);
    }

    /* Result cards */
    .result-real {
        background: linear-gradient(135deg, rgba(16,185,129,0.18), rgba(16,185,129,0.05));
        border: 1px solid rgba(16, 185, 129, 0.45);
        border-radius: 14px;
        padding: 1.1rem 1.3rem;
        font-size: 1.25rem;
        font-weight: 700;
        color: #6ee7b7;
        text-align: center;
        box-shadow: 0 4px 20px rgba(16, 185, 129, 0.15);
    }
    .result-fake {
        background: linear-gradient(135deg, rgba(239,68,68,0.18), rgba(239,68,68,0.05));
        border: 1px solid rgba(239, 68, 68, 0.45);
        border-radius: 14px;
        padding: 1.1rem 1.3rem;
        font-size: 1.25rem;
        font-weight: 700;
        color: #fca5a5;
        text-align: center;
        box-shadow: 0 4px 20px rgba(239, 68, 68, 0.15);
    }

    .example-chip {
        font-size: 0.85rem !important;
        padding: 0.4rem 0.7rem !important;
        background: rgba(255,255,255,0.06) !important;
        box-shadow: none !important;
    }

    [data-testid="stSidebar"] {
        background: rgba(15, 12, 41, 0.85);
        backdrop-filter: blur(10px);
    }

    footer, header {visibility: hidden;}
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
    "🚨 Sensational / likely fake": "Doctors hate this one weird trick that cures any disease overnight, no pills needed",
    "📊 Neutral / likely real": "The Federal Reserve raised interest rates by a quarter point on Wednesday, citing inflation data",
    "🛸 Conspiracy-style": "Government secretly controls the weather using hidden machines, whistleblower claims",
}

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📰 About")
    st.write(
        "TF-IDF + Logistic Regression model trained on 40,000+ labeled "
        "news articles from the Kaggle *Fake and Real News Dataset*."
    )
    st.divider()
    st.markdown("### ⚡ Try an example")
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
# Hero header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <div class="hero-badge">NLP · Machine Learning</div>
        <div class="main-title">📰 Fake News Detector</div>
        <div class="subtitle">
            Paste a headline or article below to check whether it looks real
            or fake, based on patterns learned from 40,000+ labeled articles.
        </div>
    </div>
    """,
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

st.markdown('<div class="glass-card">', unsafe_allow_html=True)
text_input = st.text_area(
    "Article text or headline",
    height=180,
    placeholder="Paste text here, or pick an example from the sidebar...",
    key="text_input",
    label_visibility="collapsed",
)
analyze = st.button("🔍 Analyze", type="primary", use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

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
                f'<div class="result-real">✅ Predicted: REAL &nbsp;·&nbsp; '
                f"{confidence:.1%} confidence</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="result-fake">🚩 Predicted: FAKE &nbsp;·&nbsp; '
                f"{confidence:.1%} confidence</div>",
                unsafe_allow_html=True,
            )

        st.write("")
        chart_df = pd.DataFrame(
            {"Probability": [prob[0], prob[1]]}, index=["Fake", "Real"]
        )
        st.bar_chart(chart_df, height=200, color="#8b5cf6")

        with st.expander("See cleaned/preprocessed text"):
            st.write(cleaned if cleaned else "*(empty after preprocessing)*")

st.markdown(
    """
    <div style="text-align:center; color:#71717a; font-size:0.85rem; margin-top:2rem;">
        Model: Logistic Regression + TF-IDF (unigrams &amp; bigrams, top 5,000 features)
        · Trained on the Kaggle "Fake and Real News Dataset" · Not a fact-checker,
        see limitations in the sidebar
    </div>
    """,
    unsafe_allow_html=True,
)
