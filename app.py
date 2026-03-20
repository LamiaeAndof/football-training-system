"""
app.py
Football Training Intelligence System — Streamlit Interface
Premium dark football performance department aesthetic.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Path setup ──────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from utils import (
    build_drill_profiles, recommend_drills,
    get_intensity_label, get_sprint_label, get_accel_label,
    summarize_session_data, md_day_color,
)
from generate_data import generate_sessions, generate_player_readiness

# ── Page config ─────────────────────────────────────────────────
st.set_page_config(
    page_title="FTIS | Football Training Intelligence",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS — Dark Elite Performance Style ──────────────────────────
st.markdown("""
<style>
  .stApp { background-color: #0A0F0D; color: #E8F5E9; }

  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0D1A12 0%, #0A1410 100%);
    border-right: 1px solid #1E3A2A;
  }
  [data-testid="stSidebar"] .stSelectbox label,
  [data-testid="stSidebar"] .stRadio label,
  [data-testid="stSidebar"] p { color: #A7F3B8 !important; }

  .hero-container {
    background: linear-gradient(135deg, #0A2018 0%, #0F2D1E 40%, #122518 100%);
    border: 1px solid #1E4A2E;
    border-radius: 16px;
    padding: 40px 48px 32px 48px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
  }
  .hero-container::before {
    content: '';
    position: absolute;
    top: -50px; right: -50px;
    width: 250px; height: 250px;
    background: radial-gradient(circle, rgba(34,197,94,0.09) 0%, transparent 70%);
    border-radius: 50%;
  }
  .hero-title {
    font-size: 2.6rem;
    font-weight: 800;
    color: #FFFFFF;
    letter-spacing: -0.5px;
    margin: 0;
    line-height: 1.1;
  }
  .hero-accent { color: #22C55E; }
  .hero-subtitle {
    color: #86EFAC;
    font-size: 1.05rem;
    margin-top: 10px;
    font-weight: 400;
    letter-spacing: 0.3px;
  }
  .hero-badge {
    display: inline-block;
    background: #1A3D26;
    border: 1px solid rgba(34,197,94,0.27);
    color: #22C55E;
    font-size: 0.72rem;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 20px;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    margin-bottom: 14px;
  }

  .kpi-card {
    background: #0F1F16;
    border: 1px solid #1E3A2A;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
    transition: border-color 0.2s;
  }
  .kpi-card:hover { border-color: rgba(34,197,94,0.33); }
  .kpi-value {
    font-size: 2.2rem;
    font-weight: 800;
    color: #22C55E;
    line-height: 1;
  }
  .kpi-label {
    color: #6B9E78;
    font-size: 0.78rem;
    font-weight: 500;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    margin-top: 6px;
  }
  .kpi-sub {
    color: #4B7A5A;
    font-size: 0.70rem;
    margin-top: 3px;
  }

  .section-header {
    font-size: 1.15rem;
    font-weight: 700;
    color: #FFFFFF;
    letter-spacing: 0.5px;
    border-left: 3px solid #22C55E;
    padding-left: 12px;
    margin: 28px 0 16px 0;
  }

  .drill-card {
    background: linear-gradient(135deg, #0F1F16 0%, #0D1A12 100%);
    border: 1px solid #1E3A2A;
    border-radius: 14px;
    padding: 22px 26px;
    margin-bottom: 16px;
    transition: all 0.25s;
    position: relative;
  }
  .drill-card:hover {
    border-color: rgba(34,197,94,0.33);
    box-shadow: 0 4px 24px rgba(34,197,94,0.07);
    transform: translateY(-1px);
  }
  .drill-card-rank {
    position: absolute;
    top: 18px; right: 18px;
    font-size: 2.8rem;
    font-weight: 900;
    color: #1E3A2A;
    line-height: 1;
  }
  .drill-name {
    font-size: 1.05rem;
    font-weight: 700;
    color: #FFFFFF;
    font-family: 'Courier New', monospace;
    letter-spacing: 0.3px;
  }
  .drill-type-badge {
    display: inline-block;
    background: #162B1E;
    border: 1px solid rgba(34,197,94,0.20);
    color: #86EFAC;
    font-size: 0.68rem;
    font-weight: 600;
    padding: 2px 9px;
    border-radius: 20px;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    margin: 6px 0 10px 0;
  }
  .drill-score {
    font-size: 1.8rem;
    font-weight: 900;
    color: #22C55E;
    line-height: 1;
  }
  .drill-score-label {
    font-size: 0.65rem;
    color: #6B9E78;
    letter-spacing: 1px;
    text-transform: uppercase;
  }
  .metric-pill {
    display: inline-block;
    background: #162B1E;
    border-radius: 6px;
    padding: 4px 10px;
    font-size: 0.75rem;
    color: #A7F3B8;
    margin: 3px;
    border: 1px solid #1E3A2A;
  }
  .rationale-text {
    color: #6B9E78;
    font-size: 0.83rem;
    line-height: 1.6;
    margin-top: 12px;
    border-top: 1px solid #1E3A2A;
    padding-top: 12px;
  }
  .score-bar-row {
    display: flex;
    align-items: center;
    gap: 8px;
    margin: 4px 0;
    font-size: 0.75rem;
  }
  .score-bar-label { color: #4B7A5A; width: 90px; }
  .score-bar-fill {
    height: 6px;
    border-radius: 3px;
    background: #22C55E;
  }
  .score-bar-bg {
    flex: 1;
    height: 6px;
    background: #1E3A2A;
    border-radius: 3px;
    overflow: hidden;
  }
  .score-bar-num { color: #86EFAC; width: 32px; text-align: right; }

  .no-results {
    background: #0F1F16;
    border: 1px dashed #1E3A2A;
    border-radius: 12px;
    padding: 40px;
    text-align: center;
    color: #4B7A5A;
  }

  .stTabs [data-baseweb="tab"] {
    background: #0F1F16;
    color: #6B9E78;
    border-radius: 8px 8px 0 0;
  }
  .stTabs [aria-selected="true"] {
    background: #162B1E !important;
    color: #22C55E !important;
    border-bottom: 2px solid #22C55E !important;
  }

  .stSelectbox > div > div { background: #0F1F16 !important; border-color: #1E3A2A !important; }
  .stTextInput > div > div { background: #0F1F16 !important; border-color: #1E3A2A !important; }

  .stButton > button {
    background: linear-gradient(135deg, #166534 0%, #15803D 100%);
    color: #FFFFFF;
    border: none;
    border-radius: 8px;
    font-weight: 700;
    letter-spacing: 0.5px;
    padding: 12px 28px;
    width: 100%;
    transition: all 0.2s;
  }
  .stButton > button:hover {
    background: linear-gradient(135deg, #15803D 0%, #16A34A 100%);
    transform: translateY(-1px);
    box-shadow: 0 4px 20px rgba(34,197,94,0.20);
  }

  .metric-row {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    margin: 8px 0;
  }
</style>
""", unsafe_allow_html=True)


# ── Data Loading ─────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_data():
    def find_file(filename):
        path_data = os.path.join(ROOT, "data", filename)
        path_root = os.path.join(ROOT, filename)
        if os.path.exists(path_data):
            return path_data
        return path_root

    templates_path  = find_file("exercise_templates.csv")
    sessions_path   = find_file("exercise_sessions.csv")
    microcycle_path = find_file("microcycle_reference.csv")
    readiness_path  = find_file("optional_player_readiness.csv")

    templates = pd.read_csv(templates_path)

    if not os.path.exists(sessions_path):
        sessions = generate_sessions(n_sessions=155, target_rows=1000)
        sessions.to_csv(sessions_path, index=False)
    else:
        sessions = pd.read_csv(sessions_path)

    if not os.path.exists(readiness_path):
        readiness = generate_player_readiness(sessions, n_players=22)
        readiness.to_csv(readiness_path, index=False)
    else:
        readiness = pd.read_csv(readiness_path)

    microcycle = pd.read_csv(microcycle_path)
    profiles   = build_drill_profiles(sessions, templates)

    return templates, sessions, readiness, microcycle, profiles


# ── Sidebar ───────────────────────────────────────────────────────

def render_sidebar():
    st.sidebar.markdown("""
    <div style='text-align:center; padding:16px 0 8px 0;'>
      <span style='font-size:2rem'>⚽</span><br>
      <span style='color:#22C55E; font-weight:700; font-size:0.9rem;
                   letter-spacing:2px; text-transform:uppercase;'>FTIS</span><br>
      <span style='color:#4B7A5A; font-size:0.68rem; letter-spacing:1px'>
        Training Intelligence
      </span>
    </div>
    <div style='border-top:1px solid #1E3A2A; margin:12px 0 20px 0'></div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown("**🎯 Coaching Query**")

    objective = st.sidebar.selectbox(
        "Primary Objective",
        ["— Select —", "intensity", "speed", "acceleration", "aerobic",
         "recovery", "activation", "pressing", "possession", "transition"],
        key="objective",
    )

    intensity = st.sidebar.selectbox(
        "Target Intensity",
        ["— Any —", "very_low", "low", "moderate", "moderate_high", "high", "very_high"],
        key="intensity",
    )

    md_day = st.sidebar.selectbox(
        "Microcycle Day",
        ["— Any —", "MD_plus1", "MD_plus2", "MD4", "MD3", "MD2", "MD1"],
        key="md_day",
    )

    st.sidebar.markdown(
        "<div style='border-top:1px solid #1E3A2A; margin:16px 0'></div>",
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("**📡 Physical Targets**")

    sprint_need = st.sidebar.select_slider(
        "Sprint Exposure Need",
        options=["none", "very_low", "low", "moderate", "high", "very_high"],
        value="moderate",
        key="sprint_need",
    )

    accel_need = st.sidebar.select_slider(
        "Acceleration Need",
        options=["none", "very_low", "low", "moderate", "high", "very_high"],
        value="moderate",
        key="accel_need",
    )

    st.sidebar.markdown(
        "<div style='border-top:1px solid #1E3A2A; margin:16px 0'></div>",
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("**🔧 Drill Type Filter**")

    exercise_type_filter = st.sidebar.selectbox(
        "Exercise Type",
        ["all", "small_sided_game", "large_sided_game", "sprint_drill",
         "rondo", "possession_game", "transition_game", "pressing_drill",
         "finishing_drill", "activation_drill", "recovery_drill"],
        key="ex_type",
    )

    recovery_mode  = st.sidebar.checkbox("🏥 Recovery / Regen Mode", key="recovery_mode")
    recommend_btn  = st.sidebar.button("⚡ GET RECOMMENDATIONS", key="recommend")

    return {
        "objective":            None if objective == "— Select —" else objective,
        "intensity":            None if intensity  == "— Any —"   else intensity,
        "md_day":               None if md_day     == "— Any —"   else md_day,
        "sprint_need":          sprint_need,
        "accel_need":           accel_need,
        "exercise_type_filter": exercise_type_filter,
        "recovery_mode":        recovery_mode,
        "recommend":            recommend_btn,
    }


# ── Hero Section ──────────────────────────────────────────────────

def render_hero(kpis: dict):
    st.markdown("""
    <div class='hero-container'>
      <div class='hero-badge'>Elite Performance Department</div>
      <div class='hero-title'>Football Training<br>
        <span class='hero-accent'>Intelligence System</span>
      </div>
      <div class='hero-subtitle'>
        Data-driven drill classification · GPS load analytics · Coach query engine · Microcycle support
      </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    kpi_items = [
        (c1, kpis["total_sessions"],          "TRAINING SESSIONS",  "synthetic dataset"),
        (c2, kpis["total_drill_occurrences"],  "DRILL OCCURRENCES",  "recorded bouts"),
        (c3, kpis["unique_exercises"],         "DRILL TEMPLATES",    "classified"),
        (c4, f"{kpis['avg_sprint_m']}m",       "AVG SPRINT DIST",    "per player/drill"),
        (c5, f"{kpis['avg_accelerations']}",   "AVG ACCELS",         "per player/drill"),
        (c6, f"{kpis['avg_rpe']}/10",          "AVG SESSION RPE",    "CR10 scale"),
    ]
    for col, val, label, sub in kpi_items:
        with col:
            st.markdown(f"""
            <div class='kpi-card'>
              <div class='kpi-value'>{val}</div>
              <div class='kpi-label'>{label}</div>
              <div class='kpi-sub'>{sub}</div>
            </div>""", unsafe_allow_html=True)


# ── Drill Card Renderer ───────────────────────────────────────────

def render_drill_card(rank: int, row: pd.Series):
    score     = float(row.get("composite_score", 0) or 0)
    name      = str(row.get("exercise_name", "Unknown"))
    ex_type   = str(row.get("exercise_type", "")).replace("_", " ").title()
    sprint_m  = float(row.get("avg_sprint_distance_m", 0) or 0)
    accels    = float(row.get("avg_accelerations", 0) or 0)
    hr        = float(row.get("avg_hr_mean", 0) or 0)
    rpe       = float(row.get("rpe_mean", 0) or 0)
    pl        = float(row.get("avg_player_load", 0) or 0)
    hsr       = float(row.get("avg_hsr_m", 0) or 0)
    dur       = row.get("duration_min", "?")
    reps      = row.get("repetitions", "?")
    players   = int(row.get("players_total", 0) or 0)
    pitch_l   = row.get("pitch_length_m", "?")
    pitch_w   = row.get("pitch_width_m", "?")
    rec_md    = str(row.get("recommended_md_day", "")).replace("_", "-")
    rationale = str(row.get("rationale", ""))

    intensity_lbl = get_intensity_label(rpe, hr)
    sprint_lbl    = get_sprint_label(sprint_m)
    accel_lbl     = get_accel_label(accels)

    sub_scores = {
        "Intensity": float(row.get("score_intensity", 50) or 50),
        "Sprint":    float(row.get("score_sprint",    50) or 50),
        "Accel":     float(row.get("score_accel",     50) or 50),
        "Objective": float(row.get("score_objective", 50) or 50),
        "MD Day":    float(row.get("score_md",        50) or 50),
    }

    rank_display = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"][rank - 1] if rank <= 5 else str(rank)

    bars_html = ""
    for dim, val in sub_scores.items():
        pct = int(val)
        bars_html += f"""
        <div class='score-bar-row'>
          <span class='score-bar-label'>{dim}</span>
          <div class='score-bar-bg'>
            <div class='score-bar-fill' style='width:{pct}%'></div>
          </div>
          <span class='score-bar-num'>{pct}</span>
        </div>"""

    st.markdown(f"""
    <div class='drill-card'>
      <div class='drill-card-rank'>{rank_display}</div>
      <div class='drill-name'>{name}</div>
      <div class='drill-type-badge'>{ex_type}</div>

      <div style='display:flex; gap:24px; align-items:flex-start; flex-wrap:wrap;'>
        <div>
          <div class='drill-score'>{score:.0f}
            <span style='font-size:1rem; color:#4B7A5A'>/100</span>
          </div>
          <div class='drill-score-label'>MATCH SCORE</div>
          <div style='margin-top:12px'>{bars_html}</div>
        </div>
        <div style='flex:1; min-width:200px;'>
          <div class='metric-row'>
            <span class='metric-pill'>📍 {pitch_l}×{pitch_w} m</span>
            <span class='metric-pill'>👥 {players} players</span>
            <span class='metric-pill'>⏱ {dur}' × {reps} reps</span>
            <span class='metric-pill'>📅 {rec_md}</span>
          </div>
          <div class='metric-row'>
            <span class='metric-pill'>💥 Intensity: {intensity_lbl}</span>
            <span class='metric-pill'>🏃 Sprint: {sprint_lbl}</span>
            <span class='metric-pill'>⚡ Accel: {accel_lbl}</span>
          </div>
          <div class='metric-row'>
            <span class='metric-pill'>Sprint {sprint_m:.0f}m</span>
            <span class='metric-pill'>HSR {hsr:.0f}m</span>
            <span class='metric-pill'>Accels {accels:.0f}</span>
            <span class='metric-pill'>PL {pl:.0f} AU</span>
            <span class='metric-pill'>HR {hr:.0f}%</span>
            <span class='metric-pill'>RPE {rpe:.1f}</span>
          </div>
        </div>
      </div>
      <div class='rationale-text'>{rationale}</div>
    </div>
    """, unsafe_allow_html=True)


# ── Analytics Tab ─────────────────────────────────────────────────

def render_analytics(sessions: pd.DataFrame, templates: pd.DataFrame, profiles: pd.DataFrame):
    st.markdown(
        "<div class='section-header'>📊 Load Analytics by MD Day</div>",
        unsafe_allow_html=True,
    )

    md_order  = ["MD_plus1", "MD_plus2", "MD4", "MD3", "MD2", "MD1"]
    md_colors = {
        "MD_plus1": "#6B7280", "MD_plus2": "#9CA3AF",
        "MD4":      "#EF4444", "MD3":      "#F97316",
        "MD2":      "#EAB308", "MD1":      "#22C55E",
    }

    sessions["md_day"] = pd.Categorical(
        sessions["md_day"], categories=md_order, ordered=True
    )
    agg = (
        sessions
        .groupby("md_day", observed=True)
        .agg(
            avg_sprint=("avg_sprint_distance_m", "mean"),
            avg_accels=("avg_accelerations",      "mean"),
            avg_pl    =("avg_player_load",         "mean"),
            avg_hr    =("avg_hr_mean",             "mean"),
            avg_rpe   =("rpe_mean",                "mean"),
            avg_hsr   =("avg_hsr_m",               "mean"),
        )
        .reset_index()
    )

    col1, col2 = st.columns(2)

    with col1:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=agg["md_day"],
            y=agg["avg_sprint"],
            marker_color=[md_colors.get(d, "#22C55E") for d in agg["md_day"]],
            name="Sprint Distance",
        ))
        fig.update_layout(
            title="Avg Sprint Distance by MD Day",
            plot_bgcolor="#0F1F16", paper_bgcolor="#0F1F16",
            font=dict(color="#A7F3B8", size=12),
            xaxis=dict(gridcolor="#1E3A2A", title=""),
            yaxis=dict(gridcolor="#1E3A2A", title="meters"),
            showlegend=False,
            margin=dict(t=40, b=20, l=10, r=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=agg["md_day"],
            y=agg["avg_accels"],
            marker_color=[md_colors.get(d, "#22C55E") for d in agg["md_day"]],
            name="Accelerations",
        ))
        fig2.update_layout(
            title="Avg Accelerations by MD Day",
            plot_bgcolor="#0F1F16", paper_bgcolor="#0F1F16",
            font=dict(color="#A7F3B8", size=12),
            xaxis=dict(gridcolor="#1E3A2A", title=""),
            yaxis=dict(gridcolor="#1E3A2A", title="count"),
            showlegend=False,
            margin=dict(t=40, b=20, l=10, r=10),
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── Radar chart ──────────────────────────────────────────────
    st.markdown(
        "<div class='section-header'>🕸️ Physical Profile by Exercise Type</div>",
        unsafe_allow_html=True,
    )

    radar_cols   = ["avg_sprint_distance_m", "avg_accelerations", "avg_player_load",
                    "avg_hsr_m", "avg_hr_mean", "rpe_mean"]
    radar_labels = ["Sprint", "Accels", "Player Load", "HSR", "HR Mean", "RPE"]

    type_agg = sessions.groupby("exercise_type")[radar_cols].mean().reset_index()

    for col in radar_cols:
        mn, mx = type_agg[col].min(), type_agg[col].max()
        type_agg[col] = ((type_agg[col] - mn) / (mx - mn + 1e-6) * 100).round(1)

    # ✅ rgba() fills — Plotly does NOT support 8-digit hex (#RRGGBBAA)
    palette = [
        "#22C55E", "#3B82F6", "#F97316", "#A855F7", "#EF4444",
        "#EAB308", "#06B6D4", "#EC4899", "#6366F1", "#14B8A6",
    ]
    rgba_fills = [
        "rgba(34,197,94,0.08)",   "rgba(59,130,246,0.08)",
        "rgba(249,115,22,0.08)",  "rgba(168,85,247,0.08)",
        "rgba(239,68,68,0.08)",   "rgba(234,179,8,0.08)",
        "rgba(6,182,212,0.08)",   "rgba(236,72,153,0.08)",
        "rgba(99,102,241,0.08)",  "rgba(20,184,166,0.08)",
    ]

    fig3 = go.Figure()
    for i, (_, trow) in enumerate(type_agg.iterrows()):
        vals          = [trow[c] for c in radar_cols]
        vals_closed   = vals + [vals[0]]
        labels_closed = radar_labels + [radar_labels[0]]
        fig3.add_trace(go.Scatterpolar(
            r=vals_closed,
            theta=labels_closed,
            name=str(trow["exercise_type"]).replace("_", " ").title(),
            line=dict(color=palette[i % len(palette)], width=2),
            fill="toself",
            fillcolor=rgba_fills[i % len(rgba_fills)],   # ✅ fixed
        ))

    fig3.update_layout(
        polar=dict(
            bgcolor="#0F1F16",
            radialaxis=dict(
                visible=True, range=[0, 100],
                gridcolor="#1E3A2A",
                tickfont=dict(color="#4B7A5A", size=9),
            ),
            angularaxis=dict(
                gridcolor="#1E3A2A",
                tickfont=dict(color="#86EFAC", size=11),
            ),
        ),
        paper_bgcolor="#0F1F16",
        font=dict(color="#A7F3B8"),
        legend=dict(
            bgcolor="#0F1F16", bordercolor="#1E3A2A",
            font=dict(color="#A7F3B8", size=10),
        ),
        margin=dict(t=30, b=20, l=20, r=20),
        height=450,
    )
    st.plotly_chart(fig3, use_container_width=True)

    # ── Scatter ──────────────────────────────────────────────────
    st.markdown(
        "<div class='section-header'>📈 Sprint vs Acceleration Profile</div>",
        unsafe_allow_html=True,
    )

    prof_plot = profiles.dropna(subset=["avg_sprint_distance_m", "avg_accelerations"])
    fig4 = px.scatter(
        prof_plot,
        x="avg_sprint_distance_m",
        y="avg_accelerations",
        color="exercise_type",
        size="avg_player_load",
        hover_name="exercise_name",
        hover_data={"avg_hr_mean": True, "rpe_mean": True, "recommended_md_day": True},
        labels={
            "avg_sprint_distance_m": "Avg Sprint Distance (m)",
            "avg_accelerations":     "Avg Accelerations (count)",
            "exercise_type":         "Drill Type",
        },
        color_discrete_sequence=palette,
    )
    fig4.update_layout(
        plot_bgcolor="#0F1F16", paper_bgcolor="#0F1F16",
        font=dict(color="#A7F3B8", size=11),
        xaxis=dict(gridcolor="#1E3A2A"),
        yaxis=dict(gridcolor="#1E3A2A"),
        legend=dict(bgcolor="#0F1F16", bordercolor="#1E3A2A"),
        margin=dict(t=20, b=20, l=10, r=10),
    )
    st.plotly_chart(fig4, use_container_width=True)


# ── Drill Library Tab ─────────────────────────────────────────────

def render_drill_library(profiles: pd.DataFrame):
    st.markdown(
        "<div class='section-header'>📚 Full Drill Library</div>",
        unsafe_allow_html=True,
    )

    type_filter = st.selectbox(
        "Filter by type",
        ["All"] + sorted(profiles["exercise_type"].dropna().unique().tolist()),
        key="lib_type",
    )

    df = profiles.copy()
    if type_filter != "All":
        df = df[df["exercise_type"] == type_filter]

    display_cols = [
        "exercise_name", "exercise_type", "players_total", "pitch_length_m",
        "pitch_width_m", "duration_min", "recommended_md_day",
        "avg_sprint_distance_m", "avg_accelerations", "avg_player_load",
        "avg_hr_mean", "rpe_mean", "sprint_profile", "intensity_profile",
    ]
    df_disp = df[display_cols].copy()
    df_disp.columns = [
        "Drill Name", "Type", "Players", "Length(m)", "Width(m)",
        "Duration(min)", "MD Day", "Sprint(m)", "Accels", "PL(AU)",
        "HR(%)", "RPE", "Sprint Profile", "Intensity Profile",
    ]
    for col in ["Sprint(m)", "Accels", "PL(AU)", "HR(%)", "RPE"]:
        df_disp[col] = df_disp[col].apply(
            lambda x: f"{x:.1f}" if pd.notna(x) else "-"
        )

    st.dataframe(df_disp.reset_index(drop=True), use_container_width=True, height=500)


# ── Microcycle Tab ────────────────────────────────────────────────

def render_microcycle(microcycle: pd.DataFrame, profiles: pd.DataFrame):
    st.markdown(
        "<div class='section-header'>📅 Weekly Microcycle Reference</div>",
        unsafe_allow_html=True,
    )

    md_order = ["MD_plus1", "MD_plus2", "MD4", "MD3", "MD2", "MD1"]
    md_labels = {
        "MD_plus1": "MD+1 — Recovery",
        "MD_plus2": "MD+2 — Light",
        "MD4":      "MD-4 — Fitness",
        "MD3":      "MD-3 — Tactical",
        "MD2":      "MD-2 — Speed",
        "MD1":      "MD-1 — Activation",
    }
    md_colors_bg = {
        "MD_plus1": "#1A1F1E", "MD_plus2": "#1A1F1E",
        "MD4":      "#2A1010", "MD3":      "#2A1A0A",
        "MD2":      "#1F1C08", "MD1":      "#0F2018",
    }
    md_border = {
        "MD_plus1": "#4B5563", "MD_plus2": "#6B7280",
        "MD4":      "#EF4444", "MD3":      "#F97316",
        "MD2":      "#EAB308", "MD1":      "#22C55E",
    }

    cols = st.columns(6)
    for i, md in enumerate(md_order):
        row_mc = microcycle[microcycle["md_day"] == md]
        if row_mc.empty:
            continue
        row_mc     = row_mc.iloc[0]
        top_drills = (
            profiles[profiles["recommended_md_day"] == md]["exercise_name"]
            .tolist()[:3]
        )
        top_drill_str = (
            "<br>".join([f"• {d}" for d in top_drills]) if top_drills else "—"
        )

        with cols[i]:
            st.markdown(f"""
            <div style='background:{md_colors_bg[md]};
                        border:1px solid {md_border[md]};
                        border-radius:10px;
                        padding:14px 12px;
                        min-height:260px;'>
              <div style='color:{md_border[md]}; font-weight:800;
                          font-size:0.95rem; margin-bottom:6px;'>
                {md.replace("_", "-")}
              </div>
              <div style='color:#CCCCCC; font-size:0.72rem; margin-bottom:10px;'>
                {md_labels.get(md, "")}
              </div>
              <div style='color:#8CA; font-size:0.68rem; line-height:1.6;'>
                <b>Goal:</b> {str(row_mc.get("primary_goal","")).replace("_"," ")}<br>
                <b>Intensity:</b> {row_mc.get("target_intensity","")}<br>
                <b>Sprint:</b> {row_mc.get("target_sprint_exposure","")}<br>
                <b>Accel:</b> {row_mc.get("target_acceleration_exposure","")}<br>
                <b>Volume:</b> {row_mc.get("target_duration_min","")} min<br>
                <b>Typical:</b> {row_mc.get("typical_volume_min","")}<br>
              </div>
              <div style='margin-top:10px; border-top:1px solid #1E3A2A;
                          padding-top:8px; color:#6B9E78; font-size:0.68rem;'>
                <b>Example drills:</b><br>{top_drill_str}
              </div>
            </div>
            """, unsafe_allow_html=True)


# ── Main App ──────────────────────────────────────────────────────

def main():
    templates, sessions, readiness, microcycle, profiles = load_data()
    kpis         = summarize_session_data(sessions)
    query_params = render_sidebar()

    render_hero(kpis)

    tab1, tab2, tab3, tab4 = st.tabs([
        "  ⚡ Drill Recommendations  ",
        "  📊 Load Analytics  ",
        "  📚 Drill Library  ",
        "  📅 Microcycle  ",
    ])

    # ── TAB 1 : RECOMMENDATIONS ──────────────────────────────────
    with tab1:
        st.markdown(
            "<div class='section-header'>🎯 Coach Query Engine</div>",
            unsafe_allow_html=True,
        )

        if not query_params["recommend"] and not any(
            [query_params["objective"], query_params["md_day"]]
        ):
            st.markdown("""
            <div class='no-results'>
              <div style='font-size:2rem'>⚡</div>
              <div style='font-size:1.1rem; color:#A7F3B8; margin:8px 0'>
                Configure your query in the sidebar
              </div>
              <div style='font-size:0.85rem'>
                Set your objective, intensity target, microcycle day,<br>
                and physical exposure needs — then click GET RECOMMENDATIONS.
              </div>
              <div style='margin-top:20px; color:#4B7A5A; font-size:0.78rem'>
                Example queries:<br>
                High intensity drill for MD-4 · Speed exposure for MD-2 ·<br>
                High acceleration but low sprint · Recovery for MD+1
              </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            with st.spinner("🔍 Analysing drill profiles..."):
                results = recommend_drills(
                    profiles_df=profiles,
                    objective=query_params["objective"],
                    intensity=query_params["intensity"],
                    md_day=query_params["md_day"],
                    sprint_need=query_params["sprint_need"],
                    acceleration_need=query_params["accel_need"],
                    recovery_mode=query_params["recovery_mode"],
                    exercise_type_filter=query_params["exercise_type_filter"],
                    top_n=5,
                )

            if results.empty:
                st.warning("No drills matched your query. Try relaxing the filters.")
            else:
                active_filters = []
                if query_params["objective"]:
                    active_filters.append(f"Objective: **{query_params['objective']}**")
                if query_params["md_day"]:
                    active_filters.append(
                        f"MD Day: **{query_params['md_day'].replace('_','-')}**"
                    )
                if query_params["intensity"]:
                    active_filters.append(f"Intensity: **{query_params['intensity']}**")
                active_filters.append(f"Sprint: **{query_params['sprint_need']}**")
                active_filters.append(f"Accel: **{query_params['accel_need']}**")
                if query_params["recovery_mode"]:
                    active_filters.append("🏥 **Recovery Mode ON**")

                st.markdown(
                    "<div style='color:#6B9E78; font-size:0.82rem; margin-bottom:16px'>"
                    "Active filters: " + " · ".join(active_filters) + "</div>",
                    unsafe_allow_html=True,
                )

                for rank, (_, row) in enumerate(results.iterrows(), 1):
                    render_drill_card(rank, row)

    # ── TAB 2 : ANALYTICS ────────────────────────────────────────
    with tab2:
        render_analytics(sessions, templates, profiles)

    # ── TAB 3 : DRILL LIBRARY ────────────────────────────────────
    with tab3:
        render_drill_library(profiles)

    # ── TAB 4 : MICROCYCLE ───────────────────────────────────────
    with tab4:
        render_microcycle(microcycle, profiles)


if __name__ == "__main__":
    main()
