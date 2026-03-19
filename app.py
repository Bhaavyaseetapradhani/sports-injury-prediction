import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from model import (
    train_model, predict_injury_risk,
    FEATURES, FEATURE_LABELS, RISK_COLORS, SAMPLE_PLAYERS
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Cricket Injury Risk Predictor",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Oswald:wght@400;600;700&family=Source+Sans+3:wght@400;600&display=swap');

html, body, [class*="css"] { font-family: 'Source Sans 3', sans-serif; }

.stApp { background-color: #0f1923; color: #e8e8e8; }

section[data-testid="stSidebar"] {
    background: #0a1018;
    border-right: 1px solid #1e2d3d;
}

.main-title {
    font-family: 'Oswald', sans-serif;
    font-size: 2.4rem;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: 1px;
    line-height: 1.1;
}

.title-accent { color: #f5a623; }

.subtitle {
    font-size: 0.9rem;
    color: #7a8a99;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 1.5rem;
}

.risk-card {
    border-radius: 8px;
    padding: 1.5rem;
    text-align: center;
    margin: 0.5rem 0;
}

.stat-card {
    background: #0a1018;
    border: 1px solid #1e2d3d;
    border-radius: 6px;
    padding: 1rem;
    text-align: center;
}

.stat-value {
    font-family: 'Oswald', sans-serif;
    font-size: 1.8rem;
    font-weight: 700;
    color: #f5a623;
}

.stat-label {
    font-size: 0.75rem;
    color: #7a8a99;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.insight-box {
    background: #0a1018;
    border: 1px solid #1e2d3d;
    border-left: 3px solid #f5a623;
    border-radius: 4px;
    padding: 0.8rem 1rem;
    margin: 0.4rem 0;
    font-size: 0.92rem;
    color: #c8d6e0;
    line-height: 1.5;
}

.section-title {
    font-family: 'Oswald', sans-serif;
    font-size: 1.1rem;
    font-weight: 600;
    color: #f5a623;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin: 1.2rem 0 0.6rem 0;
    padding-bottom: 0.3rem;
    border-bottom: 1px solid #1e2d3d;
}

.stButton > button {
    background: #f5a623 !important;
    color: #0f1923 !important;
    border: none !important;
    border-radius: 4px !important;
    font-family: 'Oswald', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    letter-spacing: 1px !important;
    padding: 0.6rem 2rem !important;
}

.stButton > button:hover {
    background: #ffc04a !important;
    box-shadow: 0 4px 15px rgba(245,166,35,0.3) !important;
}

.stSlider > div > div > div { background: #f5a623 !important; }
.stSelectbox > div > div { background: #0a1018 !important; border-color: #1e2d3d !important; }
hr { border-color: #1e2d3d !important; }

.model-badge {
    display: inline-block;
    background: #1e2d3d;
    color: #f5a623;
    padding: 3px 10px;
    border-radius: 3px;
    font-size: 0.75rem;
    font-family: monospace;
    letter-spacing: 1px;
    margin: 2px;
}
</style>
""", unsafe_allow_html=True)

# ── Train model (cached) ──────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    return train_model()

with st.spinner(" Loading prediction model..."):
    model, scaler, accuracy, report = load_model()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-title"> Cricket Player<br><span class="title-accent">Injury Risk Predictor</span></div>
<div class="subtitle">Machine Learning · Random Forest · Sports Science</div>
""", unsafe_allow_html=True)

col_acc1, col_acc2, col_acc3, col_acc4 = st.columns(4)
with col_acc1:
    st.markdown(f'<div class="stat-card"><div class="stat-value">{accuracy*100:.1f}%</div><div class="stat-label">Model Accuracy</div></div>', unsafe_allow_html=True)
with col_acc2:
    st.markdown(f'<div class="stat-card"><div class="stat-value">3,000</div><div class="stat-label">Training Samples</div></div>', unsafe_allow_html=True)
with col_acc3:
    st.markdown(f'<div class="stat-card"><div class="stat-value">14</div><div class="stat-label">Features</div></div>', unsafe_allow_html=True)
with col_acc4:
    st.markdown(f'<div class="stat-card"><div class="stat-value">200</div><div class="stat-label">Decision Trees</div></div>', unsafe_allow_html=True)

st.markdown("---")

# ── Sidebar — Player Input ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("###  Player Profile")
    st.markdown("---")

    # Quick load sample player
    st.markdown("**Quick Load Sample Player**")
    sample_choice = st.selectbox("", ["Custom Player"] + list(SAMPLE_PLAYERS.keys()), label_visibility="collapsed")

    if sample_choice != "Custom Player":
        defaults = SAMPLE_PLAYERS[sample_choice]
    else:
        defaults = {
           "age": 25, "matches_last_30_days": 6, "balls_bowled_last_30_days": 120,
           "batting_innings_last_30_days": 5, "days_since_last_rest": 10,
           "career_matches": 80, "previous_injuries": 1, "travel_hours_last_14_days": 20,
           "training_hours_per_week": 20, "fatigue_score": 5, "pitch_hardness": 6,
           "player_role": 0, "format_intensity": 1, "bmi": 24.0,
        }

    st.markdown("---")
    st.markdown("**Basic Info**")

    player_name = st.text_input("Player Name", value=sample_choice if sample_choice != "Custom Player" else "My Player")
    age = st.slider("Age", 16, 45, defaults["age"])
    bmi = st.slider("BMI", 18.0, 35.0, float(defaults["bmi"]), 0.1)

    role_map = {0: "Batsman", 1: "Bowler", 2: "All-Rounder", 3: "Wicket-Keeper"}
    role_inv = {v: k for k, v in role_map.items()}
    player_role_str = st.selectbox("Player Role", list(role_map.values()), index=defaults["player_role"])
    player_role = role_inv[player_role_str]

    fmt_map = {0: "T20", 1: "ODI", 2: "Test"}
    fmt_inv = {v: k for k, v in fmt_map.items()}
    format_str = st.selectbox("Match Format", list(fmt_map.values()), index=defaults["format_intensity"])
    format_intensity = fmt_inv[format_str]

    st.markdown("---")
    st.markdown("**Workload (Last 30 Days)**")
    matches_last_30 = st.slider("Matches played", 0, 20, defaults["matches_last_30_days"])
    balls_bowled = st.slider("Balls bowled", 0, 800, defaults["balls_bowled_last_30_days"], 10)
    batting_innings = st.slider("Batting innings", 0, 20, defaults["batting_innings_last_30_days"])
    days_rest = st.slider("Days since last rest day", 0, 60, defaults["days_since_last_rest"])

    st.markdown("---")
    st.markdown("**Career & History**")
    career_matches = st.slider("Career matches", 0, 500, defaults["career_matches"])
    prev_injuries = st.slider("Previous injuries", 0, 15, defaults["previous_injuries"])

    st.markdown("---")
    st.markdown("**Physical & Lifestyle**")
    travel_hours = st.slider("Travel hours (last 14 days)", 0, 100, defaults["travel_hours_last_14_days"])
    training_hours = st.slider("Training hours/week", 0, 40, defaults["training_hours_per_week"])
    fatigue_score = st.slider("Fatigue score (1=fresh, 10=exhausted)", 1, 10, defaults["fatigue_score"])
    pitch_hardness = st.slider("Pitch hardness (1=soft, 10=rock hard)", 1, 10, defaults["pitch_hardness"])

    st.markdown("---")
    predict_btn = st.button(" PREDICT INJURY RISK", use_container_width=True)

# ── Build player data dict ────────────────────────────────────────────────────
player_data = {
   "age": age,
   "matches_last_30_days": matches_last_30,
   "balls_bowled_last_30_days": balls_bowled,
   "batting_innings_last_30_days": batting_innings,
   "days_since_last_rest": days_rest,
   "career_matches": career_matches,
   "previous_injuries": prev_injuries,
   "travel_hours_last_14_days": travel_hours,
   "training_hours_per_week": training_hours,
   "fatigue_score": fatigue_score,
   "pitch_hardness": pitch_hardness,
   "player_role": player_role,
   "format_intensity": format_intensity,
   "bmi": bmi,
}

# ── Auto-predict on load / on button ─────────────────────────────────────────
if predict_btn or "result" not in st.session_state:
    result = predict_injury_risk(model, scaler, player_data)
    st.session_state["result"] = result
    st.session_state["player_name"] = player_name
    st.session_state["player_data"] = player_data.copy()

result = st.session_state["result"]
player_name_display = st.session_state.get("player_name", player_name)

# ── Results Layout ────────────────────────────────────────────────────────────
left, right = st.columns([1, 2])

with left:
    # Risk level card
    rc = result["risk_color"]
    rl = result["risk_label"]
    st.markdown(f"""
    <div style='background:{rc}11; border:2px solid {rc}; border-radius:10px; padding:2rem; text-align:center; margin-bottom:1rem'>
        <div style='font-family:Oswald,sans-serif; font-size:0.8rem; color:{rc}; letter-spacing:3px; text-transform:uppercase'>Injury Risk Level</div>
        <div style='font-family:Oswald,sans-serif; font-size:3.5rem; font-weight:700; color:{rc}; line-height:1.1'>{rl}</div>
        <div style='font-size:0.85rem; color:#7a8a99; margin-top:0.3rem'>{player_name_display}</div>
    </div>
   """, unsafe_allow_html=True)

    # Probability bars
    st.markdown('<div class="section-title">Risk Probabilities</div>', unsafe_allow_html=True)
    for label, prob in result["probabilities"].items():
        color = RISK_COLORS[label]
        st.markdown(f"""
        <div style='margin:0.4rem 0'>
            <div style='display:flex; justify-content:space-between; font-size:0.85rem; margin-bottom:2px'>
                <span style='color:#c8d6e0'>{label}</span>
                <span style='color:{color}; font-weight:600'>{prob}%</span>
            </div>
            <div style='background:#1e2d3d; border-radius:3px; height:8px'>
                <div style='background:{color}; width:{prob}%; height:8px; border-radius:3px; transition:width 0.5s'></div>
            </div>
        </div>
       """, unsafe_allow_html=True)

    # Model info
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <span class="model-badge">Random Forest</span>
    <span class="model-badge">200 Trees</span>
    <span class="model-badge">14 Features</span>
   """, unsafe_allow_html=True)

with right:
    # Insights
    st.markdown('<div class="section-title"> Key Risk Factors</div>', unsafe_allow_html=True)
    for insight in result["insights"]:
        st.markdown(f'<div class="insight-box">{insight}</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Feature importance chart
    st.markdown('<div class="section-title"> Feature Importance (Top 6)</div>', unsafe_allow_html=True)

    feat_names = [FEATURE_LABELS.get(f, f) for f, _ in result["top_risk_factors"]]
    feat_vals = [round(v * 100, 2) for _, v in result["top_risk_factors"]]

    fig = go.Figure(go.Bar(
        x=feat_vals[::-1],
        y=feat_names[::-1],
        orientation='h',
        marker=dict(
            color=feat_vals[::-1],
            colorscale=[[0, '#1e2d3d'], [0.5, '#f5a62366'], [1, '#f5a623']],
            line=dict(color='#f5a623', width=0.5)
        ),
        text=[f"{v}%" for v in feat_vals[::-1]],
        textposition='outside',
        textfont=dict(color='#c8d6e0', size=11),
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(10,16,24,1)',
        margin=dict(l=0, r=60, t=10, b=10),
        height=260,
        xaxis=dict(showgrid=False, showticklabels=False, color='#7a8a99'),
        yaxis=dict(color='#c8d6e0', tickfont=dict(size=11)),
        font=dict(family='Source Sans 3'),
    )
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ── Player Comparison Section ─────────────────────────────────────────────────
st.markdown('<div class="section-title">️ Compare All Sample Players</div>', unsafe_allow_html=True)

compare_data = []
for pname, pdata in SAMPLE_PLAYERS.items():
    res = predict_injury_risk(model, scaler, pdata)
    compare_data.append({
       "Player": pname,
       "Risk Level": res["risk_label"],
       "Low %": res["probabilities"]["Low"],
       "Medium %": res["probabilities"]["Medium"],
       "High %": res["probabilities"]["High"],
       "Age": pdata["age"],
       "Matches (30d)": pdata["matches_last_30_days"],
       "Balls Bowled": pdata["balls_bowled_last_30_days"],
       "Prev. Injuries": pdata["previous_injuries"],
       "Fatigue": pdata["fatigue_score"],
    })

compare_df = pd.DataFrame(compare_data)

# Color the risk level column
def color_risk(val):
    colors_map = {"Low": "color: #00ff88", "Medium": "color: #ffcc00", "High": "color: #ff4444"}
    return colors_map.get(val, "")

st.dataframe(
    compare_df.style.applymap(color_risk, subset=["Risk Level"]),
    use_container_width=True,
    hide_index=True,
)

# Comparison bar chart
fig2 = go.Figure()
players = [d["Player"].split("(")[0].strip() for d in compare_data]

for risk_label, color in RISK_COLORS.items():
    vals = [d[f"{risk_label} %"] for d in compare_data]
    fig2.add_trace(go.Bar(
        name=risk_label,
        x=players,
        y=vals,
        marker_color=color,
        opacity=0.85,
    ))

fig2.update_layout(
    barmode='stack',
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(10,16,24,1)',
    margin=dict(l=0, r=0, t=20, b=0),
    height=300,
    legend=dict(orientation='h', yanchor='bottom', y=1.02, font=dict(color='#c8d6e0')),
    xaxis=dict(color='#c8d6e0', gridcolor='#1e2d3d'),
    yaxis=dict(color='#7a8a99', gridcolor='#1e2d3d', title="Risk %"),
    font=dict(family='Source Sans 3', color='#c8d6e0'),
)
st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")
st.markdown("""
<div style='text-align:center; font-size:0.78rem; color:#4a5568; padding:0.5rem'>
️ This tool is for educational and research purposes. Not a substitute for professional medical assessment.<br>
Built with Python · scikit-learn · Streamlit · Plotly
</div>
""", unsafe_allow_html=True)
