import streamlit as st
import pandas as pd
import numpy as np
import pickle

from xgboost import XGBRegressor

st.set_page_config(
    page_title="Ford Car Price Predictor",
    layout="wide",
    page_icon="🚗"
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500;600&family=Space+Mono:wght@400;700&display=swap');

/* ── Root palette ── */
:root {
    --ford-blue:   #003478;
    --ford-mid:    #0057B8;
    --accent:      #E8A020;
    --accent-dim:  #c47e0a;
    --bg-dark:     #080c14;
    --bg-card:     #0d1520;
    --bg-elevated: #111d2e;
    --border:      rgba(0,87,184,0.25);
    --border-glow: rgba(232,160,32,0.4);
    --text-primary:#f0f4ff;
    --text-muted:  #7a8fa8;
    --success:     #2ecc71;
}

/* ── Global reset ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    color: var(--text-primary);
}

.stApp {
    background: var(--bg-dark);
    background-image:
        radial-gradient(ellipse 80% 50% at 50% -10%, rgba(0,87,184,0.18) 0%, transparent 70%),
        repeating-linear-gradient(0deg, transparent, transparent 39px, rgba(0,87,184,0.04) 40px),
        repeating-linear-gradient(90deg, transparent, transparent 39px, rgba(0,87,184,0.04) 40px);
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem 4rem; max-width: 1400px; }

/* ── Hero header ── */
.hero-wrap {
    position: relative;
    overflow: hidden;
    border-radius: 20px;
    background: linear-gradient(135deg, #001a40 0%, #003478 50%, #0057B8 100%);
    padding: 3rem 3.5rem 2.5rem;
    margin-bottom: 2.5rem;
    border: 1px solid rgba(232,160,32,0.3);
    box-shadow: 0 0 60px rgba(0,52,120,0.5), inset 0 1px 0 rgba(255,255,255,0.07);
}
.hero-wrap::before {
    content: "FORD";
    position: absolute;
    right: -20px; top: -30px;
    font-family: 'Bebas Neue', sans-serif;
    font-size: 16rem;
    line-height: 1;
    color: rgba(255,255,255,0.03);
    pointer-events: none;
    user-select: none;
}
.hero-badge {
    display: inline-block;
    font-family: 'Space Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--accent);
    border: 1px solid var(--accent);
    border-radius: 4px;
    padding: 3px 10px;
    margin-bottom: 1rem;
}
.hero-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 3.8rem;
    letter-spacing: 0.06em;
    line-height: 1;
    color: #fff;
    margin: 0 0 0.6rem;
}
.hero-sub {
    font-size: 1rem;
    color: rgba(240,244,255,0.65);
    font-weight: 300;
    max-width: 520px;
    margin: 0;
    line-height: 1.6;
}
.hero-stripe {
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 4px;
    background: linear-gradient(90deg, var(--accent), var(--ford-mid), var(--accent));
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-card);
    border-radius: 12px;
    padding: 5px;
    gap: 4px;
    border: 1px solid var(--border);
    margin-bottom: 2rem;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
    font-size: 0.88rem;
    letter-spacing: 0.04em;
    padding: 0.55rem 1.4rem !important;
    color: var(--text-muted) !important;
    transition: all 0.2s ease;
}
.stTabs [aria-selected="true"] {
    background: var(--ford-mid) !important;
    color: #fff !important;
    box-shadow: 0 2px 12px rgba(0,87,184,0.45);
}
.stTabs [data-baseweb="tab-panel"] { padding: 0 !important; }
.stTabs [data-baseweb="tab-border"] { display: none !important; }

/* ── Section headings ── */
.section-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 0.3rem;
}
.section-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2rem;
    letter-spacing: 0.06em;
    color: var(--text-primary);
    margin: 0 0 1.6rem;
}

/* ── Cards ── */
.card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.8rem 2rem;
    margin-bottom: 1.4rem;
    transition: border-color 0.2s;
}
.card:hover { border-color: rgba(0,87,184,0.5); }

/* ── Form inputs ── */
.stSelectbox > div > div,
.stNumberInput > div > div > input {
    background: var(--bg-elevated) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text-primary) !important;
    transition: border-color 0.2s, box-shadow 0.2s;
}
.stSelectbox > div > div:hover,
.stNumberInput > div > div > input:focus {
    border-color: var(--ford-mid) !important;
    box-shadow: 0 0 0 3px rgba(0,87,184,0.15) !important;
}

/* ── Slider ── */
.stSlider [data-baseweb="slider"] div[role="slider"] {
    background: var(--accent) !important;
    border-color: var(--accent) !important;
}
.stSlider [data-baseweb="slider"] div[data-testid="stThumbValue"] {
    background: var(--ford-mid) !important;
    border-radius: 6px;
}

/* ── Labels ── */
label, .stSelectbox label, .stNumberInput label, .stSlider label {
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: var(--text-muted) !important;
    margin-bottom: 0.3rem !important;
}

/* ── Predict button ── */
.stButton > button {
    background: linear-gradient(135deg, var(--ford-mid), var(--ford-blue)) !important;
    color: #fff !important;
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 1.15rem !important;
    letter-spacing: 0.15em !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.85rem 2.5rem !important;
    width: 100% !important;
    cursor: pointer !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 4px 20px rgba(0,87,184,0.4) !important;
    position: relative;
    overflow: hidden;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(0,87,184,0.6) !important;
    background: linear-gradient(135deg, #0068d6, #003f8a) !important;
}
.stButton > button:active { transform: translateY(0px) !important; }

/* ── Success result box ── */
.stAlert {
    border-radius: 14px !important;
    border: 1px solid rgba(46,204,113,0.35) !important;
    background: rgba(46,204,113,0.08) !important;
}

/* ── Price result card ── */
.price-result {
    background: linear-gradient(135deg, rgba(0,52,120,0.5), rgba(0,87,184,0.3));
    border: 1px solid var(--border-glow);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    text-align: center;
    margin-top: 1.2rem;
    box-shadow: 0 0 40px rgba(232,160,32,0.1);
}
.price-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 0.5rem;
}
.price-value {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 3.5rem;
    letter-spacing: 0.04em;
    color: #fff;
    line-height: 1;
}
.price-sub {
    font-size: 0.8rem;
    color: var(--text-muted);
    margin-top: 0.5rem;
}

/* ── Metric cards ── */
.metric-row {
    display: flex;
    gap: 1rem;
    margin-bottom: 1.6rem;
    flex-wrap: wrap;
}
.metric-box {
    flex: 1;
    min-width: 140px;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    text-align: center;
}
.metric-box .m-val {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2rem;
    color: var(--accent);
    line-height: 1;
}
.metric-box .m-label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--text-muted);
    margin-top: 0.3rem;
}

/* ── Dataframe ── */
.stDataFrame {
    border-radius: 12px !important;
    overflow: hidden;
    border: 1px solid var(--border) !important;
}
.stDataFrame thead tr th {
    background: var(--bg-elevated) !important;
    color: var(--accent) !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
}

/* ── Charts ── */
.stBarChart, .stLineChart {
    background: var(--bg-card);
    border-radius: 14px;
    padding: 1rem;
    border: 1px solid var(--border);
}

/* ── Divider ── */
.fancy-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--border), transparent);
    margin: 2rem 0;
}

/* ── Input group titles ── */
.input-group-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.15rem;
    letter-spacing: 0.1em;
    color: var(--ford-mid);
    border-left: 3px solid var(--accent);
    padding-left: 10px;
    margin-bottom: 1rem;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-dark); }
::-webkit-scrollbar-thumb { background: var(--ford-mid); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ─── Hero Header ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
    <div class="hero-badge">🚗 AI-Powered Valuation Engine</div>
    <div class="hero-title">Ford Car Price Predictor</div>
    <p class="hero-sub">
        Enter your vehicle's specifications and get an instant market price estimate
        powered by XGBoost machine learning.
    </p>
    <div class="hero-stripe"></div>
</div>
""", unsafe_allow_html=True)

# ─── Load models ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    model      = pickle.load(open("models/xgboost_model.pkl", "rb"))
    poly       = pickle.load(open("models/xgbregressor.pkl", "rb"))
    label_enc  = pickle.load(open("models/label_encoder.pkl", "rb"))
    trans_cols = pickle.load(open("models/transmission_cols.pkl", "rb"))
    fuel_cols  = pickle.load(open("models/fuel_cols.pkl", "rb"))
    return model, poly, label_enc, trans_cols, fuel_cols

@st.cache_data
def load_data():
    scores = pd.read_csv("reports/model_scores.csv")
    df     = pd.read_csv("data/ford.csv")
    return scores, df

model, poly, label_model, trans_cols, fuel_cols = load_models()
scores, df = load_data()

# ─── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "💰  Price Prediction",
    "📊  Data Insights",
    "📈  Model Performance"
])

# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — Prediction
# ════════════════════════════════════════════════════════════════════════════════
with tab1:

    st.markdown('<div class="section-label">Vehicle Configuration</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Predict Your Ford\'s Value</div>', unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="input-group-title">Basic Details</div>', unsafe_allow_html=True)

        model_name = st.selectbox("Car Model", label_model.classes_)
        year = st.slider("Manufacturing Year", 2000, 2020, 2018)
        mileage = st.number_input("Mileage (miles)", 0, 200000, 10000, step=500)

        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="input-group-title">Technical Specs</div>', unsafe_allow_html=True)

        tax    = st.number_input("Road Tax (£)", 0, 500, 150)
        mpg    = st.number_input("Fuel Efficiency (MPG)", 10.0, 100.0, 50.0, step=0.5)
        engine = st.number_input("Engine Size (L)", 1.0, 5.0, 1.5, step=0.1)

        st.markdown('</div>', unsafe_allow_html=True)

    col_t, col_f = st.columns(2, gap="large")

    with col_t:
        transmission = st.selectbox(
            "Transmission Type",
            [c.split("_")[1] for c in trans_cols]
        )

    with col_f:
        fuel = st.selectbox(
            "Fuel Type",
            [c.split("_")[1] for c in fuel_cols]
        )

    st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
    predict_btn = st.button("⚡ Predict Market Price", use_container_width=True)

    if predict_btn:
        model_encoded = label_model.transform([model_name])[0]

        input_data = {
            "model":      [model_encoded],
            "year":       [year],
            "mileage":    [mileage],
            "tax":        [tax],
            "mpg":        [mpg],
            "engineSize": [engine],
        }
        df_input = pd.DataFrame(input_data)

        for col in trans_cols:
            df_input[col] = 1 if col.split("_")[1] == transmission else 0
        for col in fuel_cols:
            df_input[col] = 1 if col.split("_")[1] == fuel else 0

        prediction = model.predict(df_input)[0]

        st.markdown(f"""
        <div class="price-result">
            <div class="price-label">Estimated Market Value</div>
            <div class="price-value">£{prediction:,.0f}</div>
            <div class="price-sub">
                {model_name} · {year} · {transmission} · {fuel} · {engine}L engine
            </div>
        </div>
        """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — Data Insights
# ════════════════════════════════════════════════════════════════════════════════
with tab2:

    st.markdown('<div class="section-label">Exploratory Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Dataset Insights</div>', unsafe_allow_html=True)

    # Quick metrics
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f'<div class="metric-box"><div class="m-val">{len(df):,}</div><div class="m-label">Total Records</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="metric-box"><div class="m-val">£{df["price"].mean():,.0f}</div><div class="m-label">Avg Price</div></div>', unsafe_allow_html=True)
    with m3:
        st.markdown(f'<div class="metric-box"><div class="m-val">{df["model"].nunique()}</div><div class="m-label">Unique Models</div></div>', unsafe_allow_html=True)
    with m4:
        st.markdown(f'<div class="metric-box"><div class="m-val">{df["year"].min()}–{df["year"].max()}</div><div class="m-label">Year Range</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-label">Raw Data</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Dataset Preview</div>', unsafe_allow_html=True)
    st.dataframe(df.head(10), use_container_width=True)

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

    ch1, ch2 = st.columns(2, gap="large")

    with ch1:
        st.markdown('<div class="section-label">Distribution</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Price Distribution</div>', unsafe_allow_html=True)
        st.bar_chart(df["price"], use_container_width=True)

    with ch2:
        st.markdown('<div class="section-label">Trend Over Time</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Avg Price by Year</div>', unsafe_allow_html=True)
        st.line_chart(df.groupby("year")["price"].mean(), use_container_width=True)

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Model Comparison</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Avg Price by Model</div>', unsafe_allow_html=True)
    st.bar_chart(df.groupby("model")["price"].mean(), use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — Model Performance
# ════════════════════════════════════════════════════════════════════════════════
with tab3:

    st.markdown('<div class="section-label">Evaluation Metrics</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Model Performance Comparison</div>', unsafe_allow_html=True)

    # Best model highlight
    best_idx   = scores["Test_R2"].idxmax()
    best_model = scores.loc[best_idx, "Model"]
    best_r2    = scores.loc[best_idx, "Test_R2"]

    st.markdown(f"""
    <div class="card" style="border-color:rgba(232,160,32,0.4); background:rgba(232,160,32,0.05);">
        <div class="price-label">🏆 Top Performing Model</div>
        <div style="font-family:'Bebas Neue',sans-serif;font-size:1.8rem;letter-spacing:0.08em;margin-top:0.2rem">
            {best_model}
        </div>
        <div style="font-size:0.85rem;color:var(--text-muted);margin-top:0.3rem">
            R² Score: <span style="color:var(--success);font-weight:600">{best_r2:.4f}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.dataframe(scores, use_container_width=True)

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="large")

    with c1:
        st.markdown('<div class="section-label">Accuracy</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">R² Score</div>', unsafe_allow_html=True)
        st.bar_chart(scores.set_index("Model")["Test_R2"], use_container_width=True)

    with c2:
        st.markdown('<div class="section-label">Error</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">RMSE Comparison</div>', unsafe_allow_html=True)
        st.bar_chart(scores.set_index("Model")["RMSE"], use_container_width=True)