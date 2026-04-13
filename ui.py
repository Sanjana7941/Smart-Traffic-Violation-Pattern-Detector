# =============================================================================
# ui.py — UI Components, Styles, Charts & Homepage
# Smart Traffic Violation Pattern Detector
# =============================================================================

from __future__ import annotations

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import plotly.express as px
import plotly.graph_objects as go

from config import METRIC_STYLES, NAV_ITEMS


# =============================================================================
# 1. GLOBAL STYLES
# =============================================================================

def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp { color: var(--text-color); }
        .stApp * { box-sizing: border-box; }
        .block-container {
            padding-top: 1.45rem; padding-bottom: 2.1rem; margin-top: 3.5rem;
            max-width: 100% !important; padding-left: 2rem !important; padding-right: 2rem !important;
        }
        section[data-testid="stSidebar"] {
            background-color: var(--secondary-background-color) !important;
            border-right: 1px solid rgba(128, 128, 128, 0.2);
        }
        section[data-testid="stSidebar"] * { color: var(--text-color) !important; }
        [data-testid="stSidebarNav"] { display: none; }
        div[role="radiogroup"] > label { padding: 0.7rem 0.8rem; margin-bottom: 0.32rem; border-radius: 14px; }
        div[role="radiogroup"] > label:hover { background: rgba(128, 128, 128, 0.1); }
        div[role="radiogroup"] label[data-baseweb="radio"] > div:first-child { display: none; }
        div[role="radiogroup"] label[data-baseweb="radio"] span { font-size: 1.02rem; font-weight: 600; }
        .status-card, section[data-testid="stSidebar"] [data-testid="stFileUploader"] {
            background: var(--background-color); border: 1px solid rgba(128, 128, 128, 0.2);
            border-radius: 18px; padding: 1rem; margin-top: 2rem;
        }
        section[data-testid="stSidebar"] [data-testid="stFileUploader"] section { padding: 0.5rem; min-height: auto; }
        .filter-shell {
            background: var(--secondary-background-color); border: 1px solid rgba(128, 128, 128, 0.2);
            border-radius: 20px; padding: 1rem 1rem 0.25rem 1rem; margin-bottom: 1rem;
        }
        .filter-title { font-size: 1.05rem; font-weight: 700; color: var(--text-color); margin-bottom: 0.8rem; }
        .hero-banner {
            background: color-mix(in srgb, var(--secondary-background-color) 85%, var(--primary-color));
            border: 1px solid rgba(128, 128, 128, 0.2); border-radius: 24px;
            padding: 1.75rem 1.6rem 1.7rem 1.6rem; margin-bottom: 1.15rem;
        }
        .hero-title { font-size: 1.8rem; font-weight: 800; line-height: 1.22; margin: 0 0 0.65rem 0; color: var(--text-color); }
        .hero-copy { color: var(--text-color); opacity: 0.8; max-width: 44rem; font-size: 1rem; line-height: 1.65; margin: 0; }
        .metric-card { color: white; border-radius: 20px; padding: 1rem 1.1rem; min-height: 140px; box-shadow: 0 10px 20px rgba(0,0,0,0.1); }
        .metric-icon { width: 50px; height: 50px; border-radius: 14px; display: flex; align-items: center; justify-content: center; font-size: 1.5rem; background: rgba(255, 255, 255, 0.16); margin-bottom: 0.7rem; }
        .metric-label { font-size: 0.95rem; opacity: 0.9; color: white !important; }
        .metric-value { font-size: 2rem; font-weight: 800; margin: 0.2rem 0; color: white !important; }
        .metric-note  { font-size: 0.9rem; opacity: 0.88; color: white !important; }
        .panel-card, .analysis-text-panel, .home-card, .analysis-card {
            background: var(--secondary-background-color); border: 1px solid rgba(128, 128, 128, 0.2);
            border-radius: 20px; padding: 1.1rem 1.2rem; width: 100%; box-shadow: 0 10px 20px rgba(0,0,0,0.05); margin-bottom: 1rem;
        }
        .panel-title, .analysis-panel-title, .home-card-title, .analysis-value { font-size: 1.12rem; font-weight: 700; color: var(--text-color) !important; margin-bottom: 0.5rem; }
        .section-shell {
            background: var(--secondary-background-color); border: 1px solid rgba(128, 128, 128, 0.20);
            border-radius: 22px; padding: 1rem 1.15rem; margin: 1.25rem 0 0.95rem 0;
        }
        .section-kicker {
            display: inline-flex; align-items: center; gap: 0.45rem; padding: 0.28rem 0.7rem; border-radius: 999px;
            background: color-mix(in srgb, var(--primary-color) 15%, transparent); color: var(--primary-color);
            font-size: 0.78rem; font-weight: 700; text-transform: uppercase; margin-bottom: 0.65rem;
        }
        .section-heading-row, .analysis-card-head, .analysis-panel-head { display: flex; align-items: center; gap: 0.8rem; margin-bottom: 0.4rem; }
        .section-icon, .analysis-card-icon, .analysis-panel-icon, .home-card-icon {
            width: 44px; height: 44px; border-radius: 14px; display: inline-flex; align-items: center; justify-content: center;
            font-size: 1.2rem; background: color-mix(in srgb, var(--primary-color) 15%, transparent); color: var(--primary-color); flex-shrink: 0;
        }
        .section-title { font-size: 1.32rem; font-weight: 800; color: var(--text-color); margin: 0; }
        .section-copy, .analysis-note, .analysis-bullet, .home-feature-list, .analysis-label { color: var(--text-color); opacity: 0.8; margin: 0; }
        div[data-testid="column"], div[data-testid="column"] > div,
        div[data-testid="stPlotlyChart"], div[data-testid="stPlotlyChart"] > div,
        .stDataFrame, [data-testid="stDataFrame"] { width: 100% !important; }
        .progress-row { margin-bottom: 1rem; }
        .progress-line { height: 10px; border-radius: 999px; background: rgba(128,128,128,0.2); overflow: hidden; margin-top: 0.35rem; }
        .progress-fill { height: 100%; border-radius: 999px; }
        .insight-card { border-radius: 16px; padding: 1rem; margin-bottom: 0.8rem; border: 1px solid rgba(128, 128, 128, 0.2); }
        .insight-warn { background: color-mix(in srgb, #ef4444 15%, transparent); }
        .insight-info { background: color-mix(in srgb, #3b82f6 15%, transparent); }
        @media (max-width: 768px) { .block-container { padding-left: 1rem !important; padding-right: 1rem !important; } }
        [st-theme-mode="dark"] .stApp,
        [st-theme-mode="dark"] header[data-testid="stHeader"],
        [st-theme-mode="dark"] .block-container { background-color: #000000 !important; }
        [st-theme-mode="dark"] section[data-testid="stSidebar"] {
            background-color: #050505 !important; border-right: 1px solid rgba(255,255,255,0.12) !important;
        }
        [st-theme-mode="dark"] .filter-shell, [st-theme-mode="dark"] .hero-banner,
        [st-theme-mode="dark"] .panel-card, [st-theme-mode="dark"] .analysis-text-panel,
        [st-theme-mode="dark"] .analysis-card, [st-theme-mode="dark"] .section-shell,
        [st-theme-mode="dark"] .status-card, [st-theme-mode="dark"] .insight-card {
            background: #000000 !important; border: 1px solid rgba(255,255,255,0.15) !important;
            background-image: none !important; box-shadow: none !important;
        }
        [st-theme-mode="dark"] .stApp *, [st-theme-mode="dark"] .section-title,
        [st-theme-mode="dark"] .panel-title, [st-theme-mode="dark"] .filter-title,
        [st-theme-mode="dark"] .analysis-value, [st-theme-mode="dark"] .hero-title,
        [st-theme-mode="dark"] .metric-value, [st-theme-mode="dark"] .metric-label,
        [st-theme-mode="dark"] .metric-note { color: #ffffff !important; }
        [st-theme-mode="dark"] .section-copy, [st-theme-mode="dark"] .analysis-note,
        [st-theme-mode="dark"] .analysis-bullet, [st-theme-mode="dark"] .hero-copy {
            color: #dddddd !important; opacity: 1 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# 2. HOMEPAGE
# =============================================================================

def _inject_theme_tracker() -> None:
    components.html("""
    <script>
    function updateTheme() {
        try {
            var body = window.parent.document.body;
            var bgColor = window.parent.getComputedStyle(body).backgroundColor;
            var rgb = bgColor.match(/\\d+/g);
            if (rgb && rgb.length >= 3) {
                var brightness = (parseInt(rgb[0])*299 + parseInt(rgb[1])*587 + parseInt(rgb[2])*114) / 1000;
                body.setAttribute('st-theme-mode', brightness < 128 ? 'dark' : 'light');
            }
        } catch(e) {}
    }
    updateTheme();
    var obs = new MutationObserver(updateTheme);
    obs.observe(window.parent.document.body, { attributes: true, attributeFilter: ['style','class'] });
    setInterval(updateTheme, 500);
    </script>
    """, height=0, width=0)


def render_home_page() -> None:
    _inject_theme_tracker()
    st.markdown("""
        <style>
        .home-hero {
            text-align: center;
            background: linear-gradient(135deg, #e0f2fe 0%, #dbeafe 34%, #ede9fe 68%, #fef3c7 100%);
            border: 1px solid rgba(128,128,128,0.2); border-radius: 28px;
            padding: 2.45rem 1.8rem 2.3rem; box-shadow: 0 20px 44px rgba(0,0,0,0.10);
            margin-bottom: 1.35rem; position: relative; overflow: hidden;
        }
        .home-hero-head { display: inline-flex; align-items: center; gap: 0.9rem; margin-bottom: 1rem; }
        .home-hero-icon { width: 62px; height: 62px; border-radius: 18px; display: inline-flex; align-items: center; justify-content: center; font-size: 2rem; background: rgba(255,255,255,0.78); box-shadow: 0 12px 24px rgba(0,0,0,0.12); flex-shrink: 0; }
        .home-hero-title { font-size: 2.65rem; font-weight: 800; line-height: 1.18; color: var(--text-color); margin: 0; }
        .home-hero-copy { max-width: 52rem; margin: 0 auto; font-size: 1.08rem; color: var(--text-color); opacity: 0.85; line-height: 1.72; }
        .home-divider { border: none; border-top: 1px solid rgba(128,128,128,0.2); margin: 1.85rem 0 1.9rem; }
        .section-heading { display: flex; align-items: center; gap: 0.7rem; font-size: 1.95rem; font-weight: 800; color: var(--text-color); margin-bottom: 1rem; }
        .feature-visual-card { background: linear-gradient(145deg, #e0f2fe 0%, #dbeafe 35%, #ede9fe 100%); border: 1px solid rgba(128,128,128,0.2); border-radius: 22px; padding: 1.35rem; box-shadow: 0 16px 34px rgba(0,0,0,0.06); }
        .feature-visual-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 0.9rem; margin-top: 1rem; }
        .feature-mini-tile { background: rgba(255,255,255,0.8); border: 1px solid rgba(128,128,128,0.2); border-radius: 18px; padding: 1rem 0.9rem; }
        .feature-mini-tile strong { display: block; margin-bottom: 0.25rem; color: var(--text-color); }
        .feature-list-panel { background: rgba(255,255,255,0.96); border: 1px solid rgba(128,128,128,0.2); border-radius: 22px; padding: 1.3rem 1.45rem; box-shadow: 0 16px 34px rgba(0,0,0,0.06); }
        .feature-list-item { display: flex; gap: 0.8rem; align-items: flex-start; margin-bottom: 1.1rem; color: var(--text-color); }
        .feature-bullet { width: 40px; height: 40px; border-radius: 12px; display: inline-flex; align-items: center; justify-content: center; font-size: 1.1rem; background: color-mix(in srgb, var(--primary-color) 20%, transparent); flex-shrink: 0; }
        .feature-list-item strong { display: block; font-size: 1.05rem; margin-bottom: 0.2rem; }
        .feature-list-item span { opacity: 0.85; line-height: 1.55; }
        [st-theme-mode="dark"] .home-hero, [st-theme-mode="dark"] .feature-visual-card,
        [st-theme-mode="dark"] .feature-list-panel { background: #000000 !important; background-image: none !important; border-color: rgba(255,255,255,0.12) !important; }
        [st-theme-mode="dark"] .feature-mini-tile { background: #0a0a0a !important; border-color: rgba(255,255,255,0.15) !important; }
        [st-theme-mode="dark"] .home-hero *, [st-theme-mode="dark"] .feature-visual-card *,
        [st-theme-mode="dark"] .feature-list-panel * { color: #ffffff !important; opacity: 1 !important; }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class="home-hero">
            <div class="home-hero-head">
                <div class="home-hero-icon">&#128678;</div>
                <div class="home-hero-title">Smart Traffic Violation Pattern Detector</div>
            </div>
            <div class="home-hero-copy">
                An intelligent, data-driven dashboard designed to uncover trends, hotspots, and behavior patterns
                in traffic violations for smarter monitoring, faster reporting, and safer roads.
            </div>
        </div>
        <hr class="home-divider" />
        <div class="section-heading"><span>&#128269;</span><span>What This System Does</span></div>
    """, unsafe_allow_html=True)

    left, right = st.columns([1.05, 1.25])
    left.markdown("""
        <div class="feature-visual-card">
            <div style="font-size:1.1rem;font-weight:800;">Traffic Intelligence Overview</div>
            <div style="margin-top:0.45rem;line-height:1.65;opacity:0.85;">
                Explore how violations change across locations, time periods, and vehicle categories.
            </div>
            <div class="feature-visual-grid">
                <div class="feature-mini-tile"><strong>Pattern Detection</strong>Identify recurring violation behavior and peak activity windows.</div>
                <div class="feature-mini-tile"><strong>Hotspot Discovery</strong>Highlight regions with high risk and frequent incidents.</div>
                <div class="feature-mini-tile"><strong>Trend Monitoring</strong>Compare daily, monthly, and category-based traffic changes.</div>
                <div class="feature-mini-tile"><strong>Decision Support</strong>Turn raw records into insights for action and planning.</div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    right.markdown("""
        <div class="feature-list-panel">
            <div class="feature-list-item"><div class="feature-bullet">&#128202;</div><div><strong>Interactive Dashboard</strong><span>User-friendly interface for exploring traffic data, summaries, and actionable insights.</span></div></div>
            <div class="feature-list-item"><div class="feature-bullet">&#127912;</div><div><strong>Data Visualization</strong><span>Use charts and visual summaries to understand hotspots, comparisons, and risk signals.</span></div></div>
            <div class="feature-list-item"><div class="feature-bullet">&#128200;</div><div><strong>Trend Analysis</strong><span>Track peak traffic hours, long-term changes, and recurring violation behavior over time.</span></div></div>
            <div class="feature-list-item"><div class="feature-bullet">&#128506;</div><div><strong>Map Visualization</strong><span>Support geospatial hotspot detection and location-based enforcement planning.</span></div></div>
        </div>
    """, unsafe_allow_html=True)


# =============================================================================
# 3. SIDEBAR & HEADER COMPONENTS
# =============================================================================

def render_sidebar_nav() -> str:
    page = st.sidebar.radio("Navigation", NAV_ITEMS, label_visibility="collapsed")
    st.sidebar.markdown("""
        <div class="status-card">
            <div style="font-size:1rem;font-weight:700;color:white;margin-bottom:0.35rem;">System Status</div>
            <div style="color:#cbd5e1;font-size:0.9rem;">Insights, filters, reports, and prediction tools are available.</div>
        </div>
    """, unsafe_allow_html=True)
    return page


def render_dashboard_header() -> None:
    st.markdown("""
        <div class="hero-banner">
            <div class="hero-title">Smart Traffic Violation Summary Dashboard &#128202;</div>
            <div class="hero-copy">An icon-led summary dashboard for understanding traffic patterns, hotspots, speed behavior, payments, and category trends.</div>
        </div>
    """, unsafe_allow_html=True)


def render_analytics_header() -> None:
    st.markdown("""
        <div class="hero-banner">
            <div class="hero-title">&#128200; Advanced Analytics</div>
            <div class="hero-copy">A text-first interpretation layer that turns filtered traffic records into short, readable findings for operational review and stakeholder reporting.</div>
        </div>
    """, unsafe_allow_html=True)


def render_dashboard_filters(df: pd.DataFrame, title: str = "Dashboard Filters"):
    st.markdown(f'<div class="filter-shell"><div class="filter-title">{title}</div>', unsafe_allow_html=True)
    min_date = df["date"].min().date()
    max_date = df["date"].max().date()
    col1, col2, col3 = st.columns([1.2, 1, 1])

    date_range = col1.date_input("Date range", value=(min_date, max_date),
                                  min_value=min_date, max_value=max_date, key="dashboard_date_range")
    start_date, end_date = (date_range[0], date_range[1]) if isinstance(date_range, tuple) and len(date_range) == 2 else (min_date, min_date)
    locations      = col2.multiselect("Location",       options=sorted(df["location"].dropna().unique().tolist()),       key="dashboard_locations")
    violation_types = col3.multiselect("Violation type", options=sorted(df["violation_type"].dropna().unique().tolist()), key="dashboard_violation_types")
    st.markdown("</div>", unsafe_allow_html=True)
    return start_date, end_date, locations, violation_types


# =============================================================================
# 4. REUSABLE UI WIDGETS
# =============================================================================

def render_metric_card(style_key: str, label: str, value: str, note: str) -> None:
    style = METRIC_STYLES[style_key]
    st.markdown(f"""
        <div class="metric-card" style="background:{style['bg']};">
            <div class="metric-icon">{style['icon']}</div>
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
    """, unsafe_allow_html=True)


def render_chart_panel(fig) -> None:
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


def render_section_header(kicker: str, icon: str, title: str, copy: str) -> None:
    st.markdown(f"""
        <div class="section-shell">
            <div class="section-kicker">{kicker}</div>
            <div class="section-heading-row">
                <div class="section-icon">{icon}</div>
                <div class="section-title">{title}</div>
            </div>
            <p class="section-copy">{copy}</p>
        </div>
    """, unsafe_allow_html=True)


def render_analysis_cards(cards: list[dict[str, str]]) -> None:
    columns = st.columns(len(cards))
    for col, card in zip(columns, cards):
        with col:
            st.markdown(f"""
                <div class="analysis-card">
                    <div class="analysis-card-head">
                        <div class="analysis-card-icon">{card.get('icon','&#128202;')}</div>
                        <div class="analysis-label">{card['label']}</div>
                    </div>
                    <div class="analysis-value">{card['value']}</div>
                    <div class="analysis-note">{card['note']}</div>
                </div>
            """, unsafe_allow_html=True)


def render_analysis_text_panel(title: str, bullets: list[str], icon: str = "&#128221;") -> None:
    bullet_html = "".join(f'<div class="analysis-bullet">{item}</div>' for item in bullets)
    st.markdown(f"""
        <div class="analysis-text-panel">
            <div class="analysis-panel-head">
                <div class="analysis-panel-icon">{icon}</div>
                <div class="analysis-panel-title">{title}</div>
            </div>
            {bullet_html}
        </div>
    """, unsafe_allow_html=True)


def render_recent_violations(df: pd.DataFrame) -> None:
    st.markdown('<div class="panel-card"><div class="panel-title">Recent Violations</div>', unsafe_allow_html=True)
    recent = df.sort_values(["date", "time"], ascending=[False, False]).head(6).copy()
    if recent.empty:
        st.info("No recent records available.")
    else:
        if "Violation_ID" not in recent.columns:
            recent.insert(0, "Violation_ID", [f"TV{1000+i}" for i in range(len(recent))])
        recent_table = recent[["Violation_ID", "location", "violation_type", "time"]].rename(
            columns={"Violation_ID": "ID", "location": "Location", "violation_type": "Type", "time": "Time"}
        )
        if "Fine_Paid" in recent.columns:
            recent_table["Status"] = recent["Fine_Paid"].replace({"Yes": "Paid", "No": "Pending"})
        st.dataframe(recent_table, use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)


# =============================================================================
# 5. PLOTLY CHART FACTORIES
# =============================================================================

PLOTLY_TEMPLATE = "plotly_white"


def _light_layout(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=56, b=20), legend_title_text="",
    )
    fig.update_xaxes(showgrid=False, zerolinecolor="rgba(128,128,128,0.2)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.2)", zerolinecolor="rgba(128,128,128,0.2)")
    return fig


def violation_bar_chart(summary_df: pd.DataFrame) -> go.Figure:
    fig = px.bar(summary_df.head(8), x="violation_type", y="count", color="count",
                 color_continuous_scale=["#fee2e2","#dc2626"], template=PLOTLY_TEMPLATE, title="Top Violation Types")
    fig.update_layout(xaxis_title="Violation Type", yaxis_title="Count", coloraxis_showscale=False)
    return _light_layout(fig)


def violations_line_chart(time_df: pd.DataFrame) -> go.Figure:
    fig = px.area(time_df, x="date", y="count", template=PLOTLY_TEMPLATE, title="Violation Trend Over Time")
    fig.update_traces(line=dict(color="#dc2626", width=3), fillcolor="rgba(220,38,38,0.12)")
    fig.update_layout(xaxis_title="Date", yaxis_title="Violations")
    return _light_layout(fig)


def vehicle_pie_chart(vehicle_df: pd.DataFrame) -> go.Figure:
    fig = px.pie(vehicle_df, names="vehicle_type", values="count", hole=0.62,
                 template=PLOTLY_TEMPLATE, title="Vehicle Distribution",
                 color_discrete_sequence=["#2563eb","#16a34a","#f97316","#7c3aed","#e11d48","#0891b2"])
    fig.update_traces(textinfo="percent")
    return _light_layout(fig)


def location_heatmap(heatmap_df: pd.DataFrame) -> go.Figure:
    fig = px.imshow(heatmap_df, aspect="auto",
                    color_continuous_scale=["#eff6ff","#93c5fd","#3b82f6","#1d4ed8"],
                    template=PLOTLY_TEMPLATE, title="Location Heatmap",
                    labels=dict(x="Violation Type", y="Location", color="Count"))
    return _light_layout(fig)


def risk_cluster_chart(risk_df: pd.DataFrame) -> go.Figure:
    color_map = {"High": "#ef4444", "Medium": "#f59e0b", "Low": "#10b981"}
    fig = px.scatter(risk_df, x="violations", y="avg_speed", size="violations",
                     color="risk_level", hover_name="location", template=PLOTLY_TEMPLATE,
                     title="High Risk Locations", color_discrete_map=color_map)
    fig.update_traces(marker=dict(line=dict(color="white", width=1), opacity=0.85))
    fig.update_layout(xaxis_title="Violation Count", yaxis_title="Average Recorded Speed")
    return _light_layout(fig)


def categorical_bar_chart(summary_df: pd.DataFrame, label_col: str, value_col: str,
                           title: str, color_scale: list[str] | None = None) -> go.Figure:
    colors = color_scale or ["#dbeafe", "#2563eb"]
    fig = px.bar(summary_df, x=label_col, y=value_col, color=value_col,
                 color_continuous_scale=colors, template=PLOTLY_TEMPLATE, title=title)
    fig.update_layout(xaxis_title=label_col.replace("_"," ").title(),
                      yaxis_title=value_col.replace("_"," ").title(), coloraxis_showscale=False)
    return _light_layout(fig)


def categorical_donut_chart(summary_df: pd.DataFrame, label_col: str,
                             value_col: str, title: str) -> go.Figure:
    fig = px.pie(summary_df, names=label_col, values=value_col, hole=0.58,
                 template=PLOTLY_TEMPLATE, title=title,
                 color_discrete_sequence=["#2563eb","#10b981","#f97316","#7c3aed","#e11d48","#0891b2","#f59e0b"])
    fig.update_traces(textinfo="percent+label")
    return _light_layout(fig)


def histogram_chart(df: pd.DataFrame, column: str, title: str,
                    nbins: int = 24, color: str = "#2563eb") -> go.Figure:
    plot_df = df.copy()
    plot_df[column] = pd.to_numeric(plot_df[column], errors="coerce")
    plot_df = plot_df.dropna(subset=[column])
    fig = px.histogram(plot_df, x=column, nbins=nbins, template=PLOTLY_TEMPLATE, title=title)
    fig.update_traces(marker_color=color, marker_line_color="white", marker_line_width=1)
    fig.update_layout(xaxis_title=column.replace("_"," ").title(), yaxis_title="Count")
    return _light_layout(fig)
