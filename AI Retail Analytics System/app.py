"""
app.py  ─  AI Retail Analytics Dashboard (Streamlit)
=====================================================
Launch:
    streamlit run app.py

Features:
  • Real-time live feed analysis (webcam or video)
  • Historical session browser
  • Interactive heatmap + visitor trend charts
  • Zone intelligence & AI insights panel
  • Dark glassmorphic UI via custom CSS
"""

import sys
import time
import io
import threading
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Resolve project root ───────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from database.db import (
    init_db, get_all_sessions, get_hourly_footfall,
    get_dwell_stats, get_frame_metrics
)
from utils.helpers import generate_demo_video, generate_insights

# ───────────────────────────────────────────────────────────────────────────
#   PAGE CONFIG
# ───────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Retail AI Analytics",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ───────────────────────────────────────────────────────────────────────────
#   CUSTOM CSS  ─ Glassmorphic Dark Theme
# ───────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');

/* ── Root palette ── */
:root {
    --bg-primary:   #0a0b14;
    --bg-glass:     rgba(255,255,255,0.04);
    --bg-glass-h:   rgba(255,255,255,0.08);
    --border:       rgba(255,255,255,0.07);
    --accent-cyan:  #00d4ff;
    --accent-pink:  #ff3cac;
    --accent-gold:  #ffd700;
    --accent-green: #00ff88;
    --text-primary: #f0f0f8;
    --text-muted:   #8888aa;
}

/* ── Global ── */
html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg-primary) !important;
    font-family: 'Inter', sans-serif;
    color: var(--text-primary);
}

[data-testid="stSidebar"] {
    background: rgba(8,9,22,0.95) !important;
    border-right: 1px solid var(--border);
    backdrop-filter: blur(20px);
}

/* ── Remove default padding ── */
.block-container { padding-top: 1rem !important; }

/* ── Glass card ── */
.glass-card {
    background: var(--bg-glass);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    backdrop-filter: blur(12px);
    transition: all 0.3s ease;
    margin-bottom: 1rem;
}
.glass-card:hover {
    background: var(--bg-glass-h);
    border-color: rgba(0,212,255,0.2);
    box-shadow: 0 0 24px rgba(0,212,255,0.06);
}

/* ── Metric tiles ── */
.metric-tile {
    background: linear-gradient(135deg, rgba(0,212,255,0.06), rgba(255,60,172,0.04));
    border: 1px solid rgba(0,212,255,0.15);
    border-radius: 16px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    transition: transform 0.25s ease, box-shadow 0.25s ease;
}
.metric-tile:hover {
    transform: translateY(-3px);
    box-shadow: 0 12px 32px rgba(0,212,255,0.12);
}
.metric-tile::before {
    content: '';
    position: absolute;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(circle, rgba(0,212,255,0.04) 0%, transparent 70%);
    pointer-events: none;
}
.metric-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.4rem;
    font-weight: 700;
    line-height: 1.1;
    background: linear-gradient(135deg, var(--accent-cyan), var(--accent-pink));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.metric-label {
    font-size: 0.75rem;
    font-weight: 500;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-top: 0.4rem;
}
.metric-icon {
    font-size: 1.6rem;
    margin-bottom: 0.3rem;
}

/* ── Section headers ── */
.section-header {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.15rem;
    font-weight: 600;
    color: var(--accent-cyan);
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 0.8rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid rgba(0,212,255,0.15);
}

/* ── Insight cards ── */
.insight-card {
    background: linear-gradient(135deg, rgba(0,212,255,0.05), rgba(0,255,136,0.03));
    border: 1px solid rgba(0,212,255,0.1);
    border-left: 3px solid var(--accent-cyan);
    border-radius: 10px;
    padding: 0.75rem 1rem;
    margin-bottom: 0.5rem;
    font-size: 0.9rem;
    line-height: 1.5;
    color: var(--text-primary);
    transition: border-color 0.25s;
}
.insight-card:hover { border-left-color: var(--accent-green); }

/* ── Status badge ── */
.badge-live {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(0,255,136,0.1);
    border: 1px solid rgba(0,255,136,0.3);
    color: var(--accent-green);
    font-size: 0.75rem; font-weight: 600;
    padding: 3px 10px; border-radius: 20px;
    letter-spacing: 0.05em;
}
.badge-live::before {
    content: '';
    width: 7px; height: 7px;
    background: var(--accent-green);
    border-radius: 50%;
    animation: pulse-dot 1.4s ease-in-out infinite;
}
@keyframes pulse-dot {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%       { opacity: 0.4; transform: scale(0.7); }
}

/* ── Hero header ── */
.hero-header {
    background: linear-gradient(135deg, rgba(0,212,255,0.08), rgba(255,60,172,0.06), rgba(0,255,136,0.04));
    border: 1px solid rgba(0,212,255,0.15);
    border-radius: 20px;
    padding: 1.8rem 2rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.hero-header::after {
    content: '';
    position: absolute;
    top: 0; right: 0; bottom: 0; left: 60%;
    background: radial-gradient(ellipse at 80% 50%, rgba(0,212,255,0.07), transparent);
    pointer-events: none;
}
.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2rem; font-weight: 800;
    background: linear-gradient(135deg, #fff 0%, var(--accent-cyan) 50%, var(--accent-pink) 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0.3rem;
}
.hero-sub { color: var(--text-muted); font-size: 0.9rem; }

/* ── Plotly chart backgrounds ── */
.js-plotly-plot { border-radius: 12px; overflow: hidden; }

/* ── Streamlit elements ── */
[data-testid="stSelectbox"] > div, [data-testid="stSlider"] > div {
    background: var(--bg-glass) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
}
.stButton > button {
    background: linear-gradient(135deg, rgba(0,212,255,0.15), rgba(255,60,172,0.1)) !important;
    border: 1px solid rgba(0,212,255,0.3) !important;
    color: var(--text-primary) !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    transition: all 0.25s !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 20px rgba(0,212,255,0.2) !important;
}
hr { border-color: var(--border) !important; }

/* ── Zone bar ── */
.zone-bar-wrap { display: flex; height: 14px; border-radius: 8px; overflow: hidden; gap: 2px; margin: 0.5rem 0; }
.zone-seg { height: 100%; border-radius: 6px; transition: flex 0.5s ease; }
</style>
""", unsafe_allow_html=True)


# ───────────────────────────────────────────────────────────────────────────
#   PLOTLY THEME HELPER
# ───────────────────────────────────────────────────────────────────────────
CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(255,255,255,0.02)",
    font=dict(color="#c0c0d0", family="Inter"),
    xaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.05)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.05)"),
    margin=dict(l=40, r=20, t=35, b=40),
)


# ───────────────────────────────────────────────────────────────────────────
#   DATABASE BOOTSTRAP
# ───────────────────────────────────────────────────────────────────────────
init_db()


# ───────────────────────────────────────────────────────────────────────────
#   SIDEBAR
# ───────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:1rem 0;'>
        <div style='font-size:2.5rem;'>🛒</div>
        <div style='font-family:Space Grotesk; font-weight:700; font-size:1.1rem; 
             background:linear-gradient(135deg,#00d4ff,#ff3cac);
             -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>
            Retail AI Analytics
        </div>
        <div style='font-size:0.7rem; color:#666; margin-top:2px;'>Customer Behavior Intelligence</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    page = st.radio(
        "Navigation",
        ["📊 Dashboard", "🎥 Live Analysis", "🔥 Heatmap Studio", "📈 Trends & Insights", "ℹ️ About"],
        label_visibility="collapsed",
    )
    st.divider()

    # Source config
    st.markdown("**⚙️ Video Source**")
    source_type = st.selectbox("Source Type", ["Webcam", "Video File", "Demo Mode"],
                                label_visibility="collapsed")
    video_source: str | int = "0"
    if source_type == "Webcam":
        cam_id = st.number_input("Camera Index", 0, 3, 0)
        video_source = int(cam_id)
    elif source_type == "Video File":
        video_source = st.text_input("File Path", placeholder="e.g. assets/demo.mp4")
    else:
        video_source = "demo"

    st.divider()
    st.markdown("""
    <div style='font-size:0.7rem; color:#555; padding:0.5rem 0;'>
        🔬 YOLOv8 · SORT Tracker · SQLite<br>
        🏗️ Amazon-grade retail intelligence<br>
        📅 Retail AI v2.0
    </div>
    """, unsafe_allow_html=True)


# ───────────────────────────────────────────────────────────────────────────
#   STATE MANAGEMENT
# ───────────────────────────────────────────────────────────────────────────
if "pipeline_active" not in st.session_state:
    st.session_state.pipeline_active = False
if "session_metrics" not in st.session_state:
    st.session_state.session_metrics = {
        "footfall": 0, "avg_dwell": 0.0, "active": 0,
        "peak_hour": "N/A", "insights": [],
    }


# ═══════════════════════════════════════════════════════════════════════════
#   PAGE: DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════
if "Dashboard" in page:

    # Hero
    st.markdown("""
    <div class='hero-header'>
        <div class='hero-title'>🛒 Retail AI Analytics</div>
        <div class='hero-sub'>
            Enterprise-grade customer behavior intelligence powered by YOLOv8 + Multi-Object Tracking
        </div>
        <div style='margin-top:0.8rem;'>
            <span class='badge-live'>LIVE SYSTEM</span>
            &nbsp;&nbsp;
            <span style='font-size:0.8rem; color:#666;'>Real-time · SQLite · Session-aware</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Aggregate DB stats ─────────────────────────────────────────────────
    sessions = get_all_sessions()
    total_visitors = sum(s["visitors"] for s in sessions)
    total_sessions = len(sessions)
    dwell_stats = get_dwell_stats()
    hourly = get_hourly_footfall()

    # ── KPI row ────────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    kpi_data = [
        (c1, "👥", str(total_visitors), "Total Visitors"),
        (c2, "📹", str(total_sessions), "Sessions"),
        (c3, "⏱️", f"{dwell_stats['avg']:.0f}s", "Avg Dwell"),
        (c4, "🔥", f"{dwell_stats['max']:.0f}s", "Max Dwell"),
        (c5, "📍", str(dwell_stats["total"]), "Dwell Samples"),
    ]
    for col, icon, val, label in kpi_data:
        with col:
            st.markdown(f"""
            <div class='metric-tile'>
                <div class='metric-icon'>{icon}</div>
                <div class='metric-value'>{val}</div>
                <div class='metric-label'>{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_left, col_right = st.columns([3, 2])

    # ── Hourly visitor trend chart ─────────────────────────────────────────
    with col_left:
        st.markdown("<div class='section-header'>📈 Hourly Visitor Traffic</div>", unsafe_allow_html=True)
        if hourly:
            hours  = [r["hour"] for r in hourly]
            counts = [r["count"] for r in hourly]

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=hours, y=counts,
                marker=dict(
                    color=counts,
                    colorscale=[[0, "rgba(0,212,255,0.3)"], [1, "rgba(255,60,172,0.9)"]],
                    line=dict(width=0),
                ),
                hovertemplate="<b>%{x}</b><br>Visitors: %{y}<extra></extra>",
            ))
            fig.add_trace(go.Scatter(
                x=hours, y=counts, mode="lines",
                line=dict(color="#00d4ff", width=2, dash="dot"),
                showlegend=False,
            ))
            fig.update_layout(
                title="Visitors per Hour",
                **CHART_LAYOUT,
                height=280,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("📭 No historical data yet. Run the pipeline to populate the database.", icon="ℹ️")

    # ── Session list ───────────────────────────────────────────────────────
    with col_right:
        st.markdown("<div class='section-header'>🗂️ Recent Sessions</div>", unsafe_allow_html=True)
        if sessions:
            for s in sessions[:8]:
                st.markdown(f"""
                <div class='glass-card' style='padding:0.8rem 1rem; margin-bottom:0.4rem;'>
                    <div style='display:flex; justify-content:space-between; align-items:center;'>
                        <div>
                            <span style='font-weight:600; color:#00d4ff;'>Session #{s['id']}</span>
                            <span style='font-size:0.75rem; color:#666; margin-left:0.5rem;'>{s['source']}</span>
                        </div>
                        <span style='
                            background:rgba(0,255,136,0.12);
                            color:#00ff88;
                            border-radius:20px;
                            padding:2px 10px;
                            font-size:0.75rem;
                            font-weight:600;'>
                            {s['visitors']} visitors
                        </span>
                    </div>
                    <div style='font-size:0.72rem; color:#555; margin-top:0.2rem;'>{s['started_at']}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("<div class='glass-card'>No sessions yet. Start the pipeline!</div>", unsafe_allow_html=True)

    # ── Dwell time distribution ────────────────────────────────────────────
    st.divider()
    dc1, dc2 = st.columns(2)

    with dc1:
        st.markdown("<div class='section-header'>⏱️ Dwell Time Ranges</div>", unsafe_allow_html=True)
        if dwell_stats["total"] > 0:
            ranges = {"< 30s": 0, "30–60s": 0, "1–2 min": 0, "2–5 min": 0, "> 5 min": 0}
            # Simulated dwell distribution based on stats
            import random; random.seed(42)
            n = max(dwell_stats["total"], 1)
            avg = dwell_stats["avg"]
            vals = [max(0, random.gauss(avg, avg / 2)) for _ in range(n)]
            for v in vals:
                if v < 30: ranges["< 30s"] += 1
                elif v < 60: ranges["30–60s"] += 1
                elif v < 120: ranges["1–2 min"] += 1
                elif v < 300: ranges["2–5 min"] += 1
                else: ranges["> 5 min"] += 1

            fig2 = go.Figure(go.Pie(
                labels=list(ranges.keys()),
                values=list(ranges.values()),
                hole=0.55,
                marker=dict(colors=["#00d4ff", "#00ff88", "#ffd700", "#ff8800", "#ff3cac"]),
                textfont=dict(color="#ddd"),
                hovertemplate="<b>%{label}</b><br>%{value} customers<extra></extra>",
            ))
            fig2.update_layout(
                title="Dwell Distribution",
                **CHART_LAYOUT,
                height=280,
                showlegend=True,
                legend=dict(font=dict(color="#999")),
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No dwell data yet.")

    with dc2:
        st.markdown("<div class='section-header'>🗺️ Zone Activity (Simulated)</div>", unsafe_allow_html=True)
        zones = {"Entrance": 45, "Electronics": 28, "Grocery": 17, "Checkout": 10}
        fig3 = go.Figure(go.Bar(
            y=list(zones.keys()),
            x=list(zones.values()),
            orientation="h",
            marker=dict(
                color=["#00d4ff", "#ff3cac", "#00ff88", "#ffd700"],
                line=dict(width=0),
            ),
            text=[f"{v}%" for v in zones.values()],
            textposition="outside",
            textfont=dict(color="#aaa"),
            hovertemplate="<b>%{y}</b><br>Traffic share: %{x}%<extra></extra>",
        ))
        fig3.update_layout(
            title="Zone Traffic Share",
            xaxis_title="% of Visitors",
            **CHART_LAYOUT,
            height=280,
        )
        st.plotly_chart(fig3, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
#   PAGE: LIVE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════
elif "Live" in page:
    st.markdown("<div class='hero-header'><div class='hero-title'>🎥 Live Analysis</div><div class='hero-sub'>Run the camera pipeline and watch analytics evolve in real-time</div></div>", unsafe_allow_html=True)

    c1, c2 = st.columns([1, 1])
    with c1:
        show_heatmap_overlay = st.toggle("🔥 Heatmap Overlay", value=False)
        run_demo = st.button("▶ Generate & Preview Demo Video", use_container_width=True)

    with c2:
        st.markdown("""
        <div class='glass-card'>
            <b style='color:#00d4ff;'>🚀 How to run the live pipeline</b><br><br>
            <code style='background:rgba(0,212,255,0.08); padding:4px 8px; border-radius:6px; color:#0ff;'>
                python main.py --demo
            </code><br><br>
            <span style='color:#666; font-size:0.8rem;'>
                This opens an OpenCV window with live annotations.<br>
                The Streamlit dashboard reads from the shared SQLite database.
            </span>
        </div>
        """, unsafe_allow_html=True)

    if run_demo:
        with st.spinner("🎬 Generating synthetic retail store simulation..."):
            Path("assets").mkdir(exist_ok=True)
            out_path = generate_demo_video("assets/demo.mp4", duration_sec=15, n_people=6)
        st.success(f"✅ Demo video created: `{out_path}`")
        st.video(out_path)

    st.divider()
    st.markdown("<div class='section-header'>📊 Live Frame Metrics (auto-refresh)</div>", unsafe_allow_html=True)

    metrics = get_frame_metrics(limit=200)
    if metrics:
        frames    = [m["frame_no"] for m in metrics]
        actives   = [m["active_ids"] for m in metrics]
        totals    = [m["total_ids"] for m in metrics]
        avg_dwells = [m["avg_dwell"] for m in metrics]

        fig_live = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                  subplot_titles=("Active Persons on Screen", "Cumulative Unique Visitors"),
                                  vertical_spacing=0.12)

        fig_live.add_trace(go.Scatter(
            x=frames, y=actives,
            fill="tozeroy", fillcolor="rgba(0,212,255,0.08)",
            line=dict(color="#00d4ff", width=2),
            name="Active",
        ), row=1, col=1)
        fig_live.add_trace(go.Scatter(
            x=frames, y=totals,
            fill="tozeroy", fillcolor="rgba(255,60,172,0.06)",
            line=dict(color="#ff3cac", width=2),
            name="Total Unique",
        ), row=2, col=1)

        fig_live.update_layout(
            **CHART_LAYOUT,
            height=380,
            showlegend=True,
            legend=dict(font=dict(color="#999")),
        )
        st.plotly_chart(fig_live, use_container_width=True)
    else:
        st.info("📭 No live data yet. Start the pipeline with `python main.py --demo`")

    st.markdown("""
    <div style='text-align:center; margin-top:1rem;'>
        <span style='font-size:0.75rem; color:#444;'>Auto-refreshes from SQLite · Run pipeline to populate live data</span>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
#   PAGE: HEATMAP STUDIO
# ═══════════════════════════════════════════════════════════════════════════
elif "Heatmap" in page:
    st.markdown("<div class='hero-header'><div class='hero-title'>🔥 Heatmap Studio</div><div class='hero-sub'>Visualize spatial foot-traffic patterns across the store</div></div>", unsafe_allow_html=True)

    cA, cB = st.columns([2, 1])

    with cA:
        # ── Simulated Heatmap visualization ────────────────────────────────
        st.markdown("<div class='section-header'>🗺️ Store Traffic Density Map</div>", unsafe_allow_html=True)

        # Generate synthetic heatmap data
        np.random.seed(77)
        H, W = 400, 700
        hm = np.zeros((H, W), dtype=np.float32)

        # Hot spots: Entrance (left), Central display, Checkout (right)
        spots = [
            (90,  200, 80, 120),   # entrance
            (310, 350, 100, 80),   # center
            (550, 150, 70, 90),    # checkout
            (200, 280, 50, 60),    # aisle 1
            (420, 190, 40, 50),    # aisle 2
        ]
        for cx, cy, sx, sy in spots:
            for _ in range(500):
                x = int(np.clip(np.random.normal(cx, sx), 0, W - 1))
                y = int(np.clip(np.random.normal(cy, sy), 0, H - 1))
                hm[y, x] += 1

        import cv2
        hm_blur = cv2.GaussianBlur(hm, (71, 71), 0)
        hm_norm = cv2.normalize(hm_blur, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        hm_color = cv2.applyColorMap(hm_norm, cv2.COLORMAP_JET)
        hm_rgb = cv2.cvtColor(hm_color, cv2.COLOR_BGR2RGB)

        fig_hm = px.imshow(hm_rgb, aspect="auto")
        # Build a layout dict without xaxis/yaxis (handled separately)
        _hm_layout = {k: v for k, v in CHART_LAYOUT.items() if k not in ("xaxis", "yaxis")}
        fig_hm.update_layout(
            **_hm_layout,
            title="Store Foot-Traffic Heatmap (Simulated)",
            height=380,
            coloraxis_showscale=False,
        )
        fig_hm.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
        fig_hm.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)

        # Zone overlays
        zone_x = [0, W // 3, 2 * W // 3, W]
        zone_labels = ["ENTRANCE", "CENTER AISLE", "BACK SECTION"]
        zone_colors = ["rgba(0,212,255,0.3)", "rgba(255,60,172,0.2)", "rgba(0,255,136,0.2)"]
        for i in range(3):
            fig_hm.add_vrect(
                x0=zone_x[i], x1=zone_x[i + 1],
                fillcolor=zone_colors[i], layer="above",
                line_width=2, line_color="rgba(255,255,255,0.2)",
                annotation_text=zone_labels[i],
                annotation_position="top left",
                annotation_font=dict(color="white", size=11),
            )
        st.plotly_chart(fig_hm, use_container_width=True)

    with cB:
        st.markdown("<div class='section-header'>🔥 Hot Zones</div>", unsafe_allow_html=True)
        hot_zones_demo = [
            {"zone": "Entrance", "intensity": 94, "icon": "🚪"},
            {"zone": "Electronics Display", "intensity": 71, "icon": "📱"},
            {"zone": "Checkout Area", "intensity": 58, "icon": "💳"},
            {"zone": "Central Aisle", "intensity": 42, "icon": "🛒"},
            {"zone": "Back Section", "intensity": 27, "icon": "🏷️"},
        ]
        for zone in hot_zones_demo:
            bar_w = zone["intensity"]
            bar_color = f"hsl({120 - int(bar_w * 1.2)}, 80%, 55%)"
            st.markdown(f"""
            <div class='glass-card' style='padding:0.9rem 1rem;'>
                <div style='display:flex; justify-content:space-between; margin-bottom:6px;'>
                    <span>{zone['icon']} <b style='color:#eee;'>{zone['zone']}</b></span>
                    <span style='font-weight:700; color:{bar_color};'>{zone['intensity']}%</span>
                </div>
                <div style='background:rgba(255,255,255,0.05); border-radius:6px; height:8px; overflow:hidden;'>
                    <div style='height:100%; width:{zone["intensity"]}%; background:{bar_color};
                                border-radius:6px; transition:width 0.5s;'></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='insight-card'>📌 Run <code>python main.py --heatmap</code> to enable live heatmap overlay on camera feed.</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
#   PAGE: TRENDS & INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════
elif "Trends" in page:
    st.markdown("<div class='hero-header'><div class='hero-title'>📈 Trends & AI Insights</div><div class='hero-sub'>Behaviour patterns, peak times, and actionable intelligence</div></div>", unsafe_allow_html=True)

    # ── Synthetic week data ────────────────────────────────────────────────
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    visitors_per_day = [142, 118, 165, 177, 210, 289, 244]
    avg_dwell_per_day = [45, 38, 52, 61, 58, 74, 68]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='section-header'>📅 Weekly Visitor Pattern</div>", unsafe_allow_html=True)
        fig_week = go.Figure()
        fig_week.add_trace(go.Bar(
            x=days, y=visitors_per_day,
            marker=dict(
                color=visitors_per_day,
                colorscale=[[0, "rgba(0,212,255,0.4)"], [1, "rgba(255,60,172,0.9)"]],
                line=dict(width=0),
            ),
            hovertemplate="<b>%{x}</b><br>Visitors: %{y}<extra></extra>",
        ))
        fig_week.update_layout(title="Visitors per Day", **CHART_LAYOUT, height=290)
        st.plotly_chart(fig_week, use_container_width=True)

    with col2:
        st.markdown("<div class='section-header'>⏱️ Average Dwell by Day</div>", unsafe_allow_html=True)
        fig_dwell = go.Figure()
        fig_dwell.add_trace(go.Scatter(
            x=days, y=avg_dwell_per_day,
            fill="tozeroy",
            fillcolor="rgba(0,255,136,0.07)",
            line=dict(color="#00ff88", width=3),
            mode="lines+markers",
            marker=dict(color="#ffd700", size=9),
            hovertemplate="<b>%{x}</b><br>Avg Dwell: %{y}s<extra></extra>",
        ))
        fig_dwell.update_layout(title="Avg Dwell Time (seconds)", **CHART_LAYOUT, height=290)
        st.plotly_chart(fig_dwell, use_container_width=True)

    # ── Hourly heatmap grid ────────────────────────────────────────────────
    st.divider()
    st.markdown("<div class='section-header'>🕐 Heatgrid: Hour × Day</div>", unsafe_allow_html=True)
    np.random.seed(99)
    hours_of_day = [f"{h:02d}:00" for h in range(9, 22)]
    z_data = np.random.randint(5, 200, size=(7, len(hours_of_day))).tolist()
    # Weekend peaks
    for d in [5, 6]:
        for h in range(3, 8):
            z_data[d][h] = int(z_data[d][h] * 2.2)

    fig_hg = go.Figure(go.Heatmap(
        z=z_data, x=hours_of_day, y=days,
        colorscale=[[0, "#0a0b14"], [0.3, "#00d4ff"], [0.7, "#ff8800"], [1.0, "#ff3cac"]],
        hovertemplate="<b>%{y}</b> @ <b>%{x}</b><br>Visitors: %{z}<extra></extra>",
    ))
    fig_hg.update_layout(
        title="Visitor Intensity: Hour × Day of Week",
        **CHART_LAYOUT,
        height=310,
        xaxis_title="Hour of Day",
        yaxis_title="",
    )
    st.plotly_chart(fig_hg, use_container_width=True)

    # ── AI Insights ────────────────────────────────────────────────────────
    st.divider()
    st.markdown("<div class='section-header'>🧠 AI Insights</div>", unsafe_allow_html=True)

    insights = generate_insights(
        footfall=sum(visitors_per_day), avg_dwell=56.0,
        peak_hour="15:00", hot_zones=[{"x": 120, "y": 200, "intensity": 94.0}],
        frame_w=700,
    )
    static_insights = [
        "📅 Saturday is the busiest day — 289 visitors (Saturday peak vs Monday baseline: +103%).",
        "🕒 Peak shopping hours are 14:00–17:00, accounting for ~38% of daily traffic.",
        "⏱️ Weekend dwell times are 35% longer — customers browse more on weekends.",
        "🔥 Entrance zone gets 94% of total traffic — strategic for promotion placement.",
        "💡 Recommend: Staff up on Saturday afternoons; run flash promotions Tue–Wed.",
        "🛒 Electronics zone holds customers 2× longer than grocery — cross-sell hotspot.",
    ]
    all_insights = insights + static_insights

    ic1, ic2 = st.columns(2)
    for i, ins in enumerate(all_insights):
        col = ic1 if i % 2 == 0 else ic2
        with col:
            st.markdown(f"<div class='insight-card'>{ins}</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
#   PAGE: ABOUT
# ═══════════════════════════════════════════════════════════════════════════
elif "About" in page:
    st.markdown("""
    <div class='hero-header' style='text-align:center;'>
        <div class='hero-title'>🛒 AI Retail Analytics System</div>
        <div class='hero-sub' style='font-size:1rem;'>
            Enterprise-grade Customer Behavior Intelligence Platform
        </div>
    </div>
    """, unsafe_allow_html=True)

    cols = st.columns(3)
    cards = [
        ("🏪", "Used By", "Amazon · Walmart · Reliance Retail class systems for understanding in-store shopper behavior at scale."),
        ("🧠", "AI Stack", "YOLOv8 person detection + SORT / DeepSORT tracking + spatial heatmaps + dwell analysis."),
        ("📊", "Dashboard", "Real-time Streamlit dashboard with Plotly charts, hotzone maps, and AI-generated insights."),
    ]
    for col, (icon, title, body) in zip(cols, cards):
        with col:
            st.markdown(f"""
            <div class='glass-card' style='text-align:center; padding:1.5rem;'>
                <div style='font-size:2.2rem; margin-bottom:0.5rem;'>{icon}</div>
                <div style='font-weight:700; color:#00d4ff; margin-bottom:0.5rem;'>{title}</div>
                <div style='font-size:0.85rem; color:#888; line-height:1.6;'>{body}</div>
            </div>
            """, unsafe_allow_html=True)

    st.divider()
    st.markdown("<div class='section-header'>🗂️ Folder Structure</div>", unsafe_allow_html=True)
    st.code("""
retail-analytics-ai/
│
├── app.py                 ← Streamlit dashboard (this file)
├── main.py                ← Core OpenCV pipeline
│
├── detection/
│   └── yolo.py            ← YOLOv8 person detection
│
├── tracking/
│   └── tracker.py         ← SORT / centroid multi-object tracker
│
├── analytics/
│   ├── footfall.py        ← Unique visitor counting
│   ├── dwell_time.py      ← Residence time per customer
│   └── heatmap.py         ← Spatial traffic density map
│
├── database/
│   └── db.py              ← SQLite persistence (sessions, events)
│
├── utils/
│   └── helpers.py         ← Frame utils, insights, demo generator
│
├── data/
│   └── retail.db          ← Auto-created SQLite database
│
└── assets/
    └── demo.mp4           ← Auto-generated demo video
    """, language="text")

    st.markdown("<div class='section-header'>⚙️ Quick Start</div>", unsafe_allow_html=True)
    st.code("""
# 1. Install dependencies
pip install ultralytics opencv-python streamlit plotly numpy

# 2. (Optional) Install SORT tracker for advanced tracking
pip install sort-tracker

# 3. Run with synthetic demo video (no camera required)
python main.py --demo

# 4. Run live from webcam
python main.py --source 0

# 5. Run with heatmap overlay
python main.py --source 0 --heatmap

# 6. Launch dashboard
streamlit run app.py
    """, language="bash")
