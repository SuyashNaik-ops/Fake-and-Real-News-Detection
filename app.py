"""
TruthLens — Fake News Detection (Streamlit web app)

Usage:
    streamlit run app.py

Requires fake_news_model.joblib and tfidf_vectorizer.joblib
(produced by train_model.py) to be in the same folder.

This file is UI/presentation only. Detection logic is untouched:
model loading (joblib), text cleaning (clean_text from train_model),
TF-IDF vectorization, and Logistic Regression prediction all work
exactly as before.
"""

from datetime import datetime
from html import escape

import streamlit as st
import joblib

from train_model import clean_text

st.set_page_config(
    page_title="TruthLens | AI News Verification",
    page_icon="🛡️",
    layout="wide",
)

# =============================================================================
# Design system
# =============================================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@500;600&display=swap');

    :root {
        --bg: #060a14;
        --card: rgba(255,255,255,0.035);
        --card-strong: rgba(255,255,255,0.06);
        --border: rgba(148,163,184,0.14);
        --border-strong: rgba(148,163,184,0.28);
        --text-primary: #edf1f7;
        --text-secondary: #97a3b6;
        --text-muted: #5b6679;
        --accent: #38bdf8;
        --accent-2: #818cf8;
        --success: #22c55e;
        --success-soft: rgba(34,197,94,0.14);
        --success-border: rgba(34,197,94,0.45);
        --danger: #f87171;
        --danger-soft: rgba(248,113,113,0.14);
        --danger-border: rgba(248,113,113,0.45);
        --warning: #fbbf24;
        --warning-soft: rgba(251,191,36,0.08);
        --warning-border: rgba(251,191,36,0.3);
        --font-display: 'Space Grotesk', sans-serif;
        --font-body: 'Inter', sans-serif;
        --font-mono: 'JetBrains Mono', monospace;
    }

    html, body, [class*="css"] { font-family: var(--font-body); }

    .stApp {
        background-color: var(--bg);
        background-image:
            linear-gradient(rgba(148,163,184,0.035) 1px, transparent 1px),
            linear-gradient(90deg, rgba(148,163,184,0.035) 1px, transparent 1px);
        background-size: 46px 46px;
    }
    .stApp::before {
        content: "";
        position: fixed;
        top: -12%;
        left: -8%;
        width: 42vw;
        height: 42vw;
        background: radial-gradient(circle, rgba(56,189,248,0.10) 0%, rgba(0,0,0,0) 70%);
        pointer-events: none;
        z-index: 0;
    }
    .stApp::after {
        content: "";
        position: fixed;
        bottom: -15%;
        right: -10%;
        width: 46vw;
        height: 46vw;
        background: radial-gradient(circle, rgba(129,140,248,0.08) 0%, rgba(0,0,0,0) 70%);
        pointer-events: none;
        z-index: 0;
    }

    .block-container {
        position: relative;
        z-index: 1;
        max-width: 1120px;
        padding-top: 1rem;
        padding-bottom: 3rem;
        margin: 0 auto;
    }

    h1, h2, h3 { font-family: var(--font-display); }

    a { color: var(--accent); }
    a:focus-visible, button:focus-visible, textarea:focus-visible, input:focus-visible {
        outline: 2px solid var(--accent);
        outline-offset: 2px;
    }

    @media (prefers-reduced-motion: reduce) {
        * { animation-duration: 0.001ms !important; animation-iteration-count: 1 !important; transition-duration: 0.001ms !important; }
    }

    /* ---------- Navbar ---------- */
    .navbar {
        position: sticky;
        top: 0;
        z-index: 50;
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.9rem 0.25rem;
        margin: -1rem -0.25rem 2.5rem;
        border-bottom: 1px solid var(--border);
        background: rgba(6,10,20,0.75);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
    }
    .navbar-brand {
        font-family: var(--font-display);
        font-weight: 700;
        font-size: 1.15rem;
        color: var(--text-primary);
        letter-spacing: -0.01em;
    }
    .navbar-links { display: flex; gap: 1.6rem; align-items: center; }
    .navbar-links a {
        color: var(--text-secondary);
        text-decoration: none;
        font-size: 0.88rem;
        font-weight: 500;
        transition: color 0.15s ease;
    }
    .navbar-links a:hover { color: var(--text-primary); }

    /* ---------- Hero ---------- */
    .hero { text-align: center; margin-bottom: 2.2rem; }
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        background: var(--card);
        border: 1px solid var(--border-strong);
        color: var(--accent);
        padding: 0.35rem 0.9rem;
        border-radius: 999px;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-bottom: 1.1rem;
        font-family: var(--font-mono);
    }
    .hero-title {
        font-family: var(--font-display);
        font-size: 2.7rem;
        font-weight: 700;
        color: var(--text-primary);
        letter-spacing: -0.02em;
        margin-bottom: 0.7rem;
        line-height: 1.1;
    }
    .hero-subtitle {
        color: var(--text-secondary);
        font-size: 1.02rem;
        max-width: 540px;
        margin: 0 auto;
        line-height: 1.6;
    }

    /* ---------- Cards / inputs ---------- */
    .panel {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 1.5rem 1.6rem 1.2rem;
        margin-bottom: 1.3rem;
    }
    .field-label {
        font-family: var(--font-mono);
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--text-muted);
        margin-bottom: 0.35rem;
    }
    .stTextInput input, .stTextArea textarea {
        background: rgba(0,0,0,0.28) !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
        color: var(--text-primary) !important;
        font-family: var(--font-body) !important;
    }
    .stTextInput input:focus, .stTextArea textarea:focus {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 3px rgba(56,189,248,0.15) !important;
    }
    .stTextInput label, .stTextArea label {
        font-family: var(--font-mono) !important;
        font-size: 0.7rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.06em !important;
        text-transform: uppercase;
        color: var(--text-muted) !important;
    }
    .char-count {
        font-family: var(--font-mono);
        font-size: 0.75rem;
        color: var(--text-muted);
        text-align: right;
        margin-top: -0.6rem;
    }

    .stButton > button {
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-family: var(--font-body) !important;
        transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease !important;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(90deg, var(--accent), var(--accent-2)) !important;
        border: none !important;
        color: #051019 !important;
        box-shadow: 0 4px 18px rgba(56,189,248,0.25) !important;
    }
    .stButton > button[kind="primary"]:hover { transform: translateY(-1px); box-shadow: 0 6px 22px rgba(56,189,248,0.4) !important; }
    .stButton > button[kind="secondary"] {
        background: transparent !important;
        border: 1px solid var(--border-strong) !important;
        color: var(--text-secondary) !important;
    }
    .stButton > button[kind="secondary"]:hover { border-color: var(--accent) !important; color: var(--text-primary) !important; }

    /* ---------- Result stamp ---------- */
    @keyframes fadeSlideUp { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
    .result-wrap { text-align: center; padding: 1.6rem 1rem 0.4rem; animation: fadeSlideUp 0.4s ease; }
    .result-stamp {
        display: inline-flex;
        align-items: center;
        gap: 0.55rem;
        padding: 0.7rem 1.6rem;
        border-radius: 12px;
        font-family: var(--font-display);
        font-weight: 700;
        font-size: 1.25rem;
        letter-spacing: 0.02em;
    }
    .stamp-real { background: var(--success-soft); border: 2px solid var(--success-border); color: var(--success); box-shadow: inset 0 0 0 4px rgba(34,197,94,0.08); }
    .stamp-fake { background: var(--danger-soft); border: 2px solid var(--danger-border); color: var(--danger); box-shadow: inset 0 0 0 4px rgba(248,113,113,0.08); }
    .result-confidence {
        font-family: var(--font-mono);
        font-size: 0.85rem;
        color: var(--text-secondary);
        margin-top: 0.7rem;
    }
    .result-confidence strong { color: var(--text-primary); }
    .confidence-track { width: 100%; max-width: 380px; height: 8px; background: rgba(148,163,184,0.14); border-radius: 999px; overflow: hidden; margin: 0.6rem auto 0.9rem; }
    .confidence-fill { height: 100%; border-radius: 999px; transition: width 0.7s ease; }
    .result-explanation { color: var(--text-secondary); font-size: 0.9rem; max-width: 460px; margin: 0 auto; line-height: 1.55; }
    .result-time { font-family: var(--font-mono); font-size: 0.72rem; color: var(--text-muted); margin-top: 0.5rem; }

    /* ---------- Stats ---------- */
    .stat-card { background: var(--card); border: 1px solid var(--border); border-radius: 14px; padding: 1.1rem; text-align: center; }
    .stat-dot { width: 8px; height: 8px; border-radius: 999px; margin: 0 auto 0.6rem; }
    .stat-value { font-family: var(--font-mono); font-size: 1.7rem; font-weight: 600; color: var(--text-primary); }
    .stat-label { font-size: 0.78rem; color: var(--text-muted); margin-top: 0.2rem; }

    /* ---------- History ---------- */
    .history-row {
        display: flex;
        align-items: center;
        gap: 0.8rem;
        padding: 0.65rem 0.2rem;
        border-bottom: 1px solid var(--border);
        font-size: 0.85rem;
    }
    .history-row:last-child { border-bottom: none; }
    .history-snippet { flex: 1; color: var(--text-secondary); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .history-badge { font-family: var(--font-mono); font-size: 0.68rem; font-weight: 600; padding: 0.15rem 0.5rem; border-radius: 999px; letter-spacing: 0.03em; }
    .history-badge.real { background: var(--success-soft); color: var(--success); }
    .history-badge.fake { background: var(--danger-soft); color: var(--danger); }
    .history-confidence, .history-time { font-family: var(--font-mono); color: var(--text-muted); font-size: 0.75rem; white-space: nowrap; }

    /* ---------- Section headers ---------- */
    .section-eyebrow { font-family: var(--font-mono); font-size: 0.72rem; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; color: var(--accent); margin-bottom: 0.4rem; }
    .section-title { font-family: var(--font-display); font-size: 1.6rem; font-weight: 700; color: var(--text-primary); margin-bottom: 0.3rem; }
    .section-desc { color: var(--text-secondary); font-size: 0.92rem; margin-bottom: 1.3rem; }

    /* ---------- Step cards ---------- */
    .step-card { background: var(--card); border: 1px solid var(--border); border-radius: 14px; padding: 1.2rem 1rem; height: 100%; transition: transform 0.15s ease, border-color 0.15s ease; }
    .step-card:hover { transform: translateY(-3px); border-color: var(--border-strong); }
    .step-number { font-family: var(--font-mono); font-size: 0.72rem; color: var(--accent); font-weight: 600; letter-spacing: 0.06em; margin-bottom: 0.5rem; }
    .step-icon { font-size: 1.4rem; margin-bottom: 0.4rem; }
    .step-title { font-family: var(--font-display); font-weight: 600; color: var(--text-primary); font-size: 0.98rem; margin-bottom: 0.35rem; }
    .step-desc { color: var(--text-secondary); font-size: 0.83rem; line-height: 1.5; }

    /* ---------- Info / warning cards ---------- */
    .info-card { background: var(--card); border: 1px solid var(--border); border-radius: 14px; padding: 1.3rem 1.4rem; height: 100%; }
    .info-card p { color: var(--text-secondary); font-size: 0.9rem; line-height: 1.65; margin-bottom: 0.7rem; }
    .info-card p:last-child { margin-bottom: 0; }
    .warning-card { background: var(--warning-soft); border: 1px solid var(--warning-border); border-radius: 14px; padding: 1.3rem 1.4rem; height: 100%; }
    .warning-title { font-family: var(--font-display); color: var(--warning); font-weight: 600; font-size: 0.95rem; margin-bottom: 0.6rem; }
    .warning-card ul { margin: 0; padding-left: 1.1rem; color: var(--text-secondary); font-size: 0.85rem; line-height: 1.7; }

    /* ---------- Tips ---------- */
    .tip-card { background: var(--card); border: 1px solid var(--border); border-radius: 14px; padding: 1.1rem 0.9rem; text-align: center; height: 100%; transition: transform 0.15s ease, border-color 0.15s ease; }
    .tip-card:hover { transform: translateY(-3px); border-color: var(--border-strong); }
    .tip-icon { font-size: 1.3rem; margin-bottom: 0.4rem; }
    .tip-title { font-family: var(--font-display); font-weight: 600; font-size: 0.85rem; color: var(--text-primary); margin-bottom: 0.3rem; }
    .tip-desc { color: var(--text-secondary); font-size: 0.76rem; line-height: 1.45; }

    /* ---------- Footer ---------- */
    .footer { text-align: center; margin-top: 3rem; padding-top: 1.6rem; border-top: 1px solid var(--border); }
    .footer-brand { font-family: var(--font-display); font-weight: 600; color: var(--text-secondary); font-size: 0.92rem; margin-bottom: 0.35rem; }
    .footer-sub { color: var(--text-muted); font-size: 0.8rem; }
    .footer-sub a { color: var(--text-muted); text-decoration: underline; }
    .footer-disclaimer { color: var(--text-muted); font-size: 0.72rem; margin-top: 0.5rem; font-style: italic; }

    footer, #MainMenu, header[data-testid="stHeader"] { visibility: hidden; height: 0; }

    @media (max-width: 768px) {
        .hero-title { font-size: 2rem; }
        [data-testid="stHorizontalBlock"] { flex-wrap: wrap; }
        [data-testid="stHorizontalBlock"] > div { min-width: 46% !important; flex: 1 1 46% !important; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# Model loading + prediction (unchanged core logic)
# =============================================================================
@st.cache_resource
def load_artifacts():
    model = joblib.load("fake_news_model.joblib")
    vectorizer = joblib.load("tfidf_vectorizer.joblib")
    return model, vectorizer


def build_input_text(headline: str, body: str) -> str:
    headline, body = headline.strip(), body.strip()
    return f"{headline} {body}".strip() if headline else body


def classify(raw_text: str):
    cleaned = clean_text(raw_text)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0]
    return pred, prob, cleaned


# =============================================================================
# Navbar
# =============================================================================
st.markdown(
    """
    <div class="navbar">
        <div class="navbar-brand">🛡️ TruthLens</div>
        <div class="navbar-links">
            <a href="#how-it-works">How it works</a>
            <a href="#about">About</a>
            <a href="https://github.com/SuyashNaik-ops/Fake-and-Real-News-Detection" target="_blank">GitHub ↗</a>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# Hero
# =============================================================================
st.markdown(
    """
    <div class="hero">
        <div class="hero-badge">🧠 NLP · Machine Learning</div>
        <div class="hero-title">Verify Before You Believe.</div>
        <div class="hero-subtitle">
            TruthLens analyzes news text with natural language processing and a
            trained classifier to flag writing patterns commonly associated
            with misinformation — instantly, in your browser.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    model, vectorizer = load_artifacts()
except FileNotFoundError:
    st.error(
        "Model files not found. Run `python train_model.py` first to generate "
        "fake_news_model.joblib and tfidf_vectorizer.joblib, then place them "
        "in this folder."
    )
    st.stop()

# =============================================================================
# Session state
# =============================================================================
st.session_state.setdefault("headline_input", "")
st.session_state.setdefault("text_input", "")
st.session_state.setdefault("last_result", None)
st.session_state.setdefault("history", [])

EXAMPLES = [
    {
        "chip": "🚨 Sensational",
        "headline": "Doctors Hate This One Weird Trick",
        "body": "Doctors hate this one weird trick that cures any disease overnight, no pills needed. Insiders reveal the shocking secret pharmaceutical companies don't want you to know.",
    },
    {
        "chip": "📊 Neutral report",
        "headline": "Federal Reserve Raises Interest Rates",
        "body": "The Federal Reserve raised interest rates by a quarter point on Wednesday, citing persistent inflation data and a resilient labor market in its latest policy statement.",
    },
    {
        "chip": "🛸 Conspiracy-style",
        "headline": "Whistleblower Claims Government Controls Weather",
        "body": "A self-described whistleblower claims the government secretly controls the weather using hidden machines, though no evidence has been provided to support the allegation.",
    },
]

# ---- Example chips (must run before the widgets below, so pre-fill works) ----
st.markdown('<div class="field-label">Quick examples</div>', unsafe_allow_html=True)
chip_cols = st.columns(3)
for col, ex in zip(chip_cols, EXAMPLES):
    with col:
        if st.button(ex["chip"], use_container_width=True, key=f"chip_{ex['chip']}"):
            st.session_state["headline_input"] = ex["headline"]
            st.session_state["text_input"] = ex["body"]

# =============================================================================
# Analysis panel
# =============================================================================
st.markdown('<div class="panel">', unsafe_allow_html=True)

headline = st.text_input(
    "Headline (optional)",
    key="headline_input",
    placeholder="e.g. Local Council Approves New Infrastructure Budget",
)
body = st.text_area(
    "Article text",
    key="text_input",
    height=170,
    placeholder="Paste a headline or article text here...",
)
st.markdown(f'<div class="char-count">{len(body)} characters</div>', unsafe_allow_html=True)

st.write("")
action_cols = st.columns([3, 1])
with action_cols[0]:
    analyze_clicked = st.button("🔍 Analyze News", type="primary", use_container_width=True)
with action_cols[1]:
    clear_clicked = st.button("Clear", type="secondary", use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)

if clear_clicked:
    st.session_state["headline_input"] = ""
    st.session_state["text_input"] = ""
    st.session_state["last_result"] = None
    st.rerun()

if analyze_clicked:
    combined = build_input_text(headline, body)
    if not combined:
        st.warning("Please enter some article text first.")
    else:
        with st.spinner("Analyzing linguistic patterns..."):
            pred, prob, cleaned = classify(combined)
        confidence = float(prob[pred])
        now_str = datetime.now().strftime("%I:%M:%S %p")

        st.session_state["last_result"] = {
            "pred": int(pred),
            "confidence": confidence,
            "cleaned": cleaned,
            "time": now_str,
        }
        snippet = (combined[:70] + "…") if len(combined) > 70 else combined
        st.session_state["history"].append(
            {
                "time": now_str,
                "label": "REAL" if pred == 1 else "FAKE",
                "confidence": confidence,
                "snippet": snippet,
            }
        )

# =============================================================================
# Result
# =============================================================================
result = st.session_state.get("last_result")
if result:
    is_real = result["pred"] == 1
    label_word = "REAL NEWS" if is_real else "FAKE NEWS"
    stamp_class = "stamp-real" if is_real else "stamp-fake"
    icon = "✅" if is_real else "🚩"
    bar_color = "var(--success)" if is_real else "var(--danger)"
    confidence = result["confidence"]

    st.markdown(
        f"""
        <div class="result-wrap">
            <div class="result-stamp {stamp_class}">{icon} {label_word}</div>
            <div class="result-confidence"><strong>{confidence:.1%}</strong> confidence</div>
            <div class="confidence-track">
                <div class="confidence-fill" style="width:{confidence*100:.1f}%; background:{bar_color};"></div>
            </div>
            <p class="result-explanation">
                The model classified this content as <strong>{label_word.lower()}</strong>,
                based on textual patterns learned from a labeled dataset of 40,000+ news articles.
            </p>
            <div class="result-time">Analyzed at {result['time']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.expander("See cleaned/preprocessed text"):
        st.write(result["cleaned"] if result["cleaned"] else "*(empty after preprocessing)*")

# =============================================================================
# Stats
# =============================================================================
history = st.session_state.get("history", [])
total = len(history)
real_count = sum(1 for h in history if h["label"] == "REAL")
fake_count = total - real_count

st.write("")
stat_cols = st.columns(3)
for col, (label, value, color) in zip(
    stat_cols,
    [
        ("Total analyses", total, "var(--accent)"),
        ("Real news", real_count, "var(--success)"),
        ("Fake news", fake_count, "var(--danger)"),
    ],
):
    with col:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-dot" style="background:{color};"></div>
                <div class="stat-value">{value}</div>
                <div class="stat-label">{label}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
st.caption("Stats reflect this browser session only and reset on reload — nothing is stored on a server.")

# ---- Recent history ----
if history:
    st.write("")
    st.markdown('<div class="section-title" style="font-size:1.1rem;">🕒 Recent analyses</div>', unsafe_allow_html=True)
    rows_html = ""
    for h in reversed(history[-5:]):
        badge_cls = "real" if h["label"] == "REAL" else "fake"
        rows_html += (
            f'<div class="history-row">'
            f'<span class="history-snippet">{escape(h["snippet"])}</span>'
            f'<span class="history-badge {badge_cls}">{h["label"]}</span>'
            f'<span class="history-confidence">{h["confidence"]:.0%}</span>'
            f'<span class="history-time">{h["time"]}</span>'
            f"</div>"
        )
    st.markdown(f'<div class="panel">{rows_html}</div>', unsafe_allow_html=True)

# =============================================================================
# How it works
# =============================================================================
st.markdown('<div id="how-it-works"></div>', unsafe_allow_html=True)
st.write("")
st.markdown('<div class="section-eyebrow">Pipeline</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">How it works</div>', unsafe_allow_html=True)
st.markdown('<div class="section-desc">A transparent, classical NLP pipeline — no black boxes.</div>', unsafe_allow_html=True)

steps = [
    ("🧹", "Preprocessing", "Lowercasing, stripping URLs and punctuation, tokenizing, removing stopwords, and lemmatizing the raw text."),
    ("🔢", "TF-IDF vectorization", "Converting cleaned text into numeric features weighted by word and phrase importance (unigrams + bigrams)."),
    ("🧠", "Logistic Regression", "A model trained on 40,000+ labeled articles scores the vector and estimates a probability for each class."),
    ("✅", "Prediction", "The higher-probability class — real or fake — is returned along with its confidence score."),
]
step_cols = st.columns(4)
for i, (col, (icon, title, desc)) in enumerate(zip(step_cols, steps), start=1):
    with col:
        st.markdown(
            f"""
            <div class="step-card">
                <div class="step-number">STEP 0{i}</div>
                <div class="step-icon">{icon}</div>
                <div class="step-title">{title}</div>
                <div class="step-desc">{desc}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# =============================================================================
# About + limitations
# =============================================================================
st.markdown('<div id="about"></div>', unsafe_allow_html=True)
st.write("")
st.markdown('<div class="section-eyebrow">Details</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">About this project</div>', unsafe_allow_html=True)
st.write("")

about_col, limits_col = st.columns([3, 2])
with about_col:
    st.markdown(
        """
        <div class="info-card">
            <p>TruthLens is a text classification project built on the Kaggle
            <em>"Fake and Real News Dataset"</em> — 40,000+ labeled news articles.</p>
            <p>Text is cleaned with NLTK (tokenization, stopword removal, lemmatization),
            converted into TF-IDF vectors (unigrams &amp; bigrams, top 5,000 features), and
            classified with a scikit-learn Logistic Regression model.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with limits_col:
    st.markdown(
        """
        <div class="warning-card">
            <div class="warning-title">⚠️ Known limitations</div>
            <ul>
                <li>Trained on 2016–2017 news — current events may be misclassified.</li>
                <li>Learns writing style and word patterns, not facts — not a fact-checker.</li>
                <li>Real-news examples skew toward one wire service's style.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

# =============================================================================
# Tips
# =============================================================================
st.write("")
st.markdown('<div class="section-eyebrow">Media literacy</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Spotting misinformation yourself</div>', unsafe_allow_html=True)
st.markdown('<div class="section-desc">A model is a starting point, not a verdict. These habits help too.</div>', unsafe_allow_html=True)

tips = [
    ("🔍", "Check the source", "Is it a known, reputable outlet — or an unfamiliar site?"),
    ("📅", "Check the date", "Old stories are often recirculated out of context."),
    ("🧾", "Verify elsewhere", "See if other credible outlets report the same story."),
    ("😡", "Watch the tone", "Heavy emotional or sensational language is a red flag."),
    ("🖼️", "Check images", "Reverse-image-search photos to confirm they aren't reused."),
]
tip_cols = st.columns(5)
for col, (icon, title, desc) in zip(tip_cols, tips):
    with col:
        st.markdown(
            f"""
            <div class="tip-card">
                <div class="tip-icon">{icon}</div>
                <div class="tip-title">{title}</div>
                <div class="tip-desc">{desc}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# =============================================================================
# Footer
# =============================================================================
st.markdown(
    """
    <div class="footer">
        <div class="footer-brand">🛡️ TruthLens</div>
        <div class="footer-sub">
            Built with Python, scikit-learn, NLTK &amp; Streamlit · TF-IDF + Logistic Regression ·
            <a href="https://github.com/SuyashNaik-ops/Fake-and-Real-News-Detection" target="_blank">View on GitHub ↗</a>
        </div>
        <div class="footer-disclaimer">Educational project — not a substitute for professional fact-checking.</div>
    </div>
    """,
    unsafe_allow_html=True,
)
