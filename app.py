# =============================================================================
# app.py — All Pages & Main Entry Point
# Smart Traffic Violation Pattern Detector
#
# Run with:  streamlit run app.py
# =============================================================================

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.express as px

from config import (
    APP_ICON, APP_TITLE,
    PAGE_HOME, PAGE_DASHBOARD, PAGE_ADVANCED, PAGE_PREDICTIONS,
    PAGE_REPORTS, PAGE_UPLOAD, PAGE_VISUALIZATION, PAGE_TRENDS,
    MONTH_ORDER,
)
from data import (
    load_data_from_path, load_data_from_upload, get_payment_model,
    build_context, summarize_counts, dataframe_to_csv_bytes, build_pdf_report,
)
from analysis import (
    apply_filters, compute_kpis, detect_patterns,
    violation_type_summary, location_summary, vehicle_type_summary,
    violations_over_time, heatmap_summary, predict_risk_clusters,
    build_report_table, predict_fine_payment_status,
)
from ui import (
    inject_styles, render_home_page,
    render_sidebar_nav, render_dashboard_header, render_analytics_header,
    render_dashboard_filters, render_metric_card, render_chart_panel,
    render_section_header, render_analysis_cards, render_analysis_text_panel,
    render_recent_violations,
    # charts
    violation_bar_chart, violations_line_chart, vehicle_pie_chart,
    location_heatmap, risk_cluster_chart, categorical_bar_chart,
    categorical_donut_chart, histogram_chart, _light_layout,
)


# =============================================================================
# Streamlit page config (must be first st call)
# =============================================================================
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# PAGE: DASHBOARD
# =============================================================================

def render_dashboard(filtered_df: pd.DataFrame) -> None:
    kpis        = compute_kpis(filtered_df)
    patterns    = detect_patterns(filtered_df)
    type_df     = violation_type_summary(filtered_df)
    vehicle_df  = vehicle_type_summary(filtered_df)
    risk_df     = predict_risk_clusters(filtered_df)
    location_df = location_summary(filtered_df).head(10)
    context     = build_context(filtered_df, risk_df)

    day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    day_df    = summarize_counts(filtered_df, "day_name")
    if not day_df.empty:
        day_df["day_name"] = pd.Categorical(day_df["day_name"], categories=day_order, ordered=True)
        day_df = day_df.sort_values("day_name")

    hourly_df        = patterns["hourly_distribution"].copy()
    weather_df       = summarize_counts(filtered_df, "Weather_Condition", top_n=8)
    road_df          = summarize_counts(filtered_df, "Road_Condition")
    state_df         = summarize_counts(filtered_df, "Registration_State", top_n=8)
    gender_df        = summarize_counts(filtered_df, "Driver_Gender")
    license_df       = summarize_counts(filtered_df, "License_Type")
    payment_df       = summarize_counts(filtered_df, "Fine_Paid")
    payment_method_df = summarize_counts(filtered_df, "Payment_Method")
    signal_df        = summarize_counts(filtered_df, "signal_status")
    helmet_df        = summarize_counts(filtered_df, "helmet_detected")
    vehicle_color_df = summarize_counts(filtered_df, "Vehicle_Color", top_n=8)
    breathalyzer_df  = summarize_counts(filtered_df, "Breathalyzer_Result")
    seatbelt_df      = summarize_counts(filtered_df, "Seatbelt_Worn")

    # KPI row
    c1, c2, c3, c4 = st.columns(4)
    with c1: render_metric_card("total",   "Total Violations",      f"{context['total']:,}",      f"Top type: {kpis['top_violation']}")
    with c2: render_metric_card("risk",    "High Risk Locations",   str(context["risk"]),          f"Peak day: {patterns['peak_day']}")
    with c3: render_metric_card("pending", "Pending Fine Payments", f"{context['pending']:,}",     "Current filtered view")
    with c4: render_metric_card("speed",   "Avg. Recorded Speed",   f"{context['speed']:.1f} km/h", f"Peak hour: {patterns['peak_hour']}")

    render_section_header("Core Pattern",  "&#128202;", "Violation Type Distribution",    "Compare the most common offense categories.")
    s1c1, s1c2 = st.columns([1.2, 1])
    with s1c1: render_chart_panel(categorical_bar_chart(type_df.head(10), "violation_type", "count", "Violation Type Distribution", ["#fee2e2","#dc2626"]))
    with s1c2: render_chart_panel(categorical_donut_chart(type_df.head(6), "violation_type", "count", "Violation Type Share"))

    render_section_header("Spatial Risk",  "&#128205;", "Violations by Location",         "Review where violations are concentrated.")
    s2c1, s2c2 = st.columns(2)
    with s2c1:
        if not location_df.empty: render_chart_panel(categorical_bar_chart(location_df, "location", "count", "Violations by Location", ["#e0f2fe","#0891b2"]))
    with s2c2:
        render_chart_panel(risk_cluster_chart(risk_df) if not risk_df.empty else location_heatmap(heatmap_summary(filtered_df)))

    render_section_header("Fleet View",    "&#128663;", "Vehicle-Based Analysis",          "Break down by vehicle mix, color, and registration.")
    s3c1, s3c2, s3c3 = st.columns(3)
    with s3c1: render_chart_panel(vehicle_pie_chart(vehicle_df))
    with s3c2:
        if not vehicle_color_df.empty: render_chart_panel(categorical_bar_chart(vehicle_color_df, "Vehicle_Color", "count", "Vehicle Color Distribution", ["#dbeafe","#2563eb"]))
    with s3c3:
        if not state_df.empty: render_chart_panel(categorical_bar_chart(state_df, "Registration_State", "count", "Registration State Analysis", ["#ede9fe","#7c3aed"]))

    render_section_header("Behavior",      "&#128100;", "Driver Behavior Analysis",        "Explore behavioral indicators and demographic distribution.")
    s4c1, s4c2, s4c3 = st.columns(3)
    with s4c1:
        if not gender_df.empty: render_chart_panel(categorical_donut_chart(gender_df, "Driver_Gender", "count", "Driver Gender Distribution"))
        elif not license_df.empty: render_chart_panel(categorical_donut_chart(license_df, "License_Type", "count", "License Type Distribution"))
    with s4c2:
        if "Previous_Violations" in filtered_df.columns: render_chart_panel(histogram_chart(filtered_df, "Previous_Violations", "Previous Violations History", nbins=15, color="#16a34a"))
    with s4c3:
        if not seatbelt_df.empty: render_chart_panel(categorical_donut_chart(seatbelt_df, "Seatbelt_Worn", "count", "Seatbelt Usage"))
        elif not helmet_df.empty: render_chart_panel(categorical_donut_chart(helmet_df, "helmet_detected", "count", "Helmet Detection Status"))

    render_section_header("Speed",         "&#9889;",   "Speed Analysis",                  "Inspect speed distributions and time-based intensity.")
    s5c1, s5c2, s5c3 = st.columns(3)
    with s5c1: render_chart_panel(histogram_chart(filtered_df, "speed", "Recorded Speed Distribution", nbins=24, color="#7c3aed"))
    with s5c2:
        if not hourly_df.empty: render_chart_panel(categorical_bar_chart(hourly_df, "hour", "count", "Hourly Speed-Linked Activity", ["#dbeafe","#2563eb"]))
    with s5c3:
        if "Speed_Limit" in filtered_df.columns: render_chart_panel(histogram_chart(filtered_df, "Speed_Limit", "Speed Limit Distribution", nbins=18, color="#f97316"))

    render_section_header("Safety Risk",   "&#127863;", "Drunk Driving",                   "Track alcohol-related signals.")
    s6c1, s6c2 = st.columns(2)
    with s6c1:
        if "Alcohol_Level" in filtered_df.columns: render_chart_panel(histogram_chart(filtered_df, "Alcohol_Level", "Alcohol Level Distribution", nbins=20, color="#ef4444"))
    with s6c2:
        if not breathalyzer_df.empty: render_chart_panel(categorical_donut_chart(breathalyzer_df, "Breathalyzer_Result", "count", "Breathalyzer Results"))

    render_section_header("Revenue",       "&#128176;", "Fine & Payment Insights",          "Evaluate payment completion and distribution.")
    s7c1, s7c2, s7c3 = st.columns(3)
    with s7c1:
        if not payment_df.empty: render_chart_panel(categorical_donut_chart(payment_df, "Fine_Paid", "count", "Fine Payment Status"))
    with s7c2:
        if "Fine_Amount" in filtered_df.columns: render_chart_panel(histogram_chart(filtered_df, "Fine_Amount", "Fine Amount Distribution", nbins=20, color="#dc2626"))
    with s7c3:
        if not payment_method_df.empty: render_chart_panel(categorical_bar_chart(payment_method_df, "Payment_Method", "count", "Payment Method Analysis", ["#dcfce7","#16a34a"]))

    render_section_header("Environment",   "&#127774;", "Environmental Impact",             "Measure how weather and road conditions relate to violations.")
    s8c1, s8c2, s8c3 = st.columns(3)
    with s8c1:
        if not weather_df.empty: render_chart_panel(categorical_bar_chart(weather_df, "Weather_Condition", "count", "Weather Condition Impact", ["#fef3c7","#f59e0b"]))
    with s8c2:
        if not road_df.empty: render_chart_panel(categorical_donut_chart(road_df, "Road_Condition", "count", "Road Condition Impact"))
    with s8c3:
        if not signal_df.empty: render_chart_panel(categorical_donut_chart(signal_df, "signal_status", "count", "Traffic Light Status"))
        elif not day_df.empty: render_chart_panel(categorical_bar_chart(day_df, "day_name", "count", "Violations by Day", ["#bfdbfe","#2563eb"]))


# =============================================================================
# PAGE: ADVANCED ANALYTICS
# =============================================================================

def render_advanced_analytics(filtered_df: pd.DataFrame) -> None:
    kpis        = compute_kpis(filtered_df)
    patterns    = detect_patterns(filtered_df)
    type_df     = violation_type_summary(filtered_df)
    location_df = location_summary(filtered_df)
    vehicle_df  = vehicle_type_summary(filtered_df)
    payment_df  = summarize_counts(filtered_df, "Fine_Paid")
    gender_df   = summarize_counts(filtered_df, "Driver_Gender")
    weather_df  = summarize_counts(filtered_df, "Weather_Condition")

    top_violation       = type_df.iloc[0]["violation_type"] if not type_df.empty else "N/A"
    top_violation_count = int(type_df.iloc[0]["count"])     if not type_df.empty else 0
    second_violation    = type_df.iloc[1]["violation_type"] if len(type_df) > 1 else "N/A"
    top_location        = location_df.iloc[0]["location"]   if not location_df.empty else "N/A"
    top_location_count  = int(location_df.iloc[0]["count"]) if not location_df.empty else 0
    top_vehicle         = vehicle_df.iloc[0]["vehicle_type"] if not vehicle_df.empty else "N/A"
    top_vehicle_count   = int(vehicle_df.iloc[0]["count"])   if not vehicle_df.empty else 0
    gender_leader       = gender_df.iloc[0]["Driver_Gender"]     if not gender_df.empty else "N/A"
    weather_leader      = weather_df.iloc[0]["Weather_Condition"] if not weather_df.empty else "N/A"

    paid_pct = 0.0
    if not payment_df.empty and payment_df["count"].sum() > 0:
        paid_count = int(payment_df.loc[payment_df["Fine_Paid"] == "Yes", "count"].sum())
        paid_pct   = (paid_count / int(payment_df["count"].sum())) * 100

    render_analysis_cards([
        {"label": "Leading Violation",  "icon": "&#9888;",  "value": f"{top_violation} ({top_violation_count})",  "note": "This violation currently appears most often in the filtered dataset."},
        {"label": "Primary Hotspot",    "icon": "&#128205;","value": f"{top_location} ({top_location_count})",    "note": "This location is contributing the highest concentration of violations."},
        {"label": "Dominant Vehicle",   "icon": "&#128663;","value": f"{top_vehicle} ({top_vehicle_count})",      "note": "This vehicle category is the strongest contributor to recorded violations."},
    ])
    render_analysis_cards([
        {"label": "Peak Hour",           "icon": "&#9200;",  "value": str(patterns["peak_hour"]), "note": "This time window shows the highest traffic-violation intensity."},
        {"label": "Peak Day",            "icon": "&#128197;","value": str(patterns["peak_day"]),  "note": "This weekday produces the highest overall violation volume."},
        {"label": "Payment Completion",  "icon": "&#128176;","value": f"{paid_pct:.1f}% paid",    "note": "This indicates the current fine-settlement rate."},
    ])

    render_section_header("Narrative Summary", "&#128221;", "Operational Reading", "Short findings for quick briefings and analytical interpretation.")
    render_analysis_text_panel("Key Findings", [
        f"Top enforcement concern: <b>{top_violation}</b> is the leading violation type, while <b>{second_violation}</b> also appears prominently.",
        f"Location pressure is highest in <b>{top_location}</b>, suggesting a concentrated hotspot.",
        f"Vehicle exposure is led by <b>{top_vehicle}</b>.",
    ], icon="&#128161;")
    render_analysis_text_panel("Behavior & Compliance Insights", [
        f"Violation activity peaks during <b>{patterns['peak_hour']}</b>.",
        f"The strongest day-level concentration falls on <b>{patterns['peak_day']}</b>.",
        f"Fine payment completion is currently estimated at <b>{paid_pct:.1f}%</b>.",
    ], icon="&#129504;")
    render_analysis_text_panel("Contextual Interpretation", [
        f"The most represented driver segment is <b>{gender_leader}</b>; most common weather context is <b>{weather_leader}</b>.",
        f"Average recorded speed is <b>{kpis['avg_speed']:.2f} km/h</b>.",
        "This page is text-led so findings can be read quickly before deeper chart exploration.",
    ], icon="&#128269;")

    st.markdown("### Supporting Views")
    sc1, sc2 = st.columns(2)
    with sc1: render_chart_panel(violation_bar_chart(type_df))
    with sc2: render_chart_panel(location_heatmap(heatmap_summary(filtered_df)))

    st.markdown("### Record Preview")
    preview_cols = [c for c in ["date","location","violation_type","vehicle_type","speed","Fine_Paid"] if c in filtered_df.columns]
    st.dataframe(filtered_df[preview_cols].head(20) if preview_cols else filtered_df.head(20), use_container_width=True, hide_index=True)


# =============================================================================
# PAGE: TREND ANALYSIS
# =============================================================================

def render_trend_analysis_page(df: pd.DataFrame) -> None:
    st.markdown("## Trend Analysis")
    work = df.copy()
    work["date"]       = pd.to_datetime(work["date"], errors="coerce")
    work["_year"]      = work["date"].dt.year.astype("Int64")
    work["_month"]     = work["date"].dt.month_name()
    work["_month_num"] = work["date"].dt.month
    work["_day"]       = work["date"].dt.day_name()
    work["_hour"]      = work["hour"] if "hour" in work.columns else work["event_time"].dt.hour
    work["_date_only"] = work["date"].dt.date

    states          = sorted(work["location"].dropna().unique().tolist()) if "location" in work.columns else []
    viol_types      = sorted(work["violation_type"].dropna().unique().tolist()) if "violation_type" in work.columns else []
    years_available = sorted(work["_year"].dropna().unique().tolist())

    # Section 1: Monthly Violations by Type
    st.markdown("### 1) Monthly Violations by Type")
    c1a, c1b = st.columns(2)
    sel_year1  = c1a.selectbox("Year",  ["All"] + years_available, key="t1_year")
    sel_state1 = c1b.selectbox("State", ["All"] + states,          key="t1_state")
    s1 = work.copy()
    if sel_year1  != "All": s1 = s1[s1["_year"]     == int(sel_year1)]
    if sel_state1 != "All": s1 = s1[s1["location"]  == sel_state1]
    if not s1.empty:
        grp1 = s1.groupby(["_month","violation_type"]).size().reset_index(name="Count")
        grp1["_month"] = pd.Categorical(grp1["_month"], categories=MONTH_ORDER, ordered=True)
        fig1 = px.line(grp1.sort_values("_month"), x="_month", y="Count", color="violation_type",
                       title="Monthly Violations by Type", markers=True,
                       labels={"_month": "Month", "violation_type": "Violation Type"})
        st.plotly_chart(_light_layout(fig1), use_container_width=True)
    else:
        st.info("No data for selected filters.")

    # Section 2: Traffic Light Status Heatmap
    st.markdown("---")
    st.markdown("### 2) Monthly Traffic Light Status by Vehicle Type")
    vehicle_types = sorted(work["vehicle_type"].dropna().unique().tolist()) if "vehicle_type" in work.columns else []
    c2a, c2b, c2c = st.columns(3)
    sel_year2    = c2a.selectbox("Year",         ["All"] + years_available, key="t2_year")
    sel_state2   = c2b.selectbox("State",        ["All"] + states,          key="t2_state")
    sel_vehicle2 = c2c.selectbox("Vehicle Type", ["All"] + vehicle_types,   key="t2_vehicle")
    s2 = work.copy()
    if sel_year2    != "All": s2 = s2[s2["_year"]        == int(sel_year2)]
    if sel_state2   != "All": s2 = s2[s2["location"]     == sel_state2]
    if sel_vehicle2 != "All": s2 = s2[s2["vehicle_type"] == sel_vehicle2]
    signal_col = "signal_status" if "signal_status" in s2.columns else None
    if signal_col and not s2.empty:
        grp2 = s2.groupby(["_month", signal_col]).size().reset_index(name="Count")
        grp2["_month"] = pd.Categorical(grp2["_month"], categories=MONTH_ORDER, ordered=True)
        pivot2 = grp2.sort_values("_month").pivot(index=signal_col, columns="_month", values="Count").fillna(0)
        for status in ["Red","Yellow","Green"]:
            if status not in pivot2.index: pivot2.loc[status] = 0
        pivot2 = pivot2.loc[["Red","Yellow","Green"]]
        fig2 = px.imshow(pivot2, aspect="auto", color_continuous_scale=["#eff6ff","#3b82f6","#1e3a8a"],
                         title="Months vs Traffic Light Status", labels=dict(x="Month", y="Traffic Light Status", color="Count"))
        fig2.update_traces(text=pivot2.values, texttemplate="%{text:.0f}")
        st.plotly_chart(_light_layout(fig2), use_container_width=True)
    else:
        st.info("No data for selected filters.")

    # Section 3: Total Fine Amount vs Years
    st.markdown("---")
    st.markdown("### 3) Total Fine Amount vs Years")
    if "Fine_Amount" in work.columns and years_available:
        c3a, c3b = st.columns(2)
        mn, mx = int(min(years_available)), int(max(years_available))
        yr_range  = c3a.slider("Year Range", min_value=mn, max_value=mx, value=(mn, mx), key="t3_yr") if mn < mx else (mn, mx)
        sel_state3 = c3b.selectbox("Location", ["All"] + states, key="t3_state")
        s3 = work.copy()
        s3["Fine_Amount"] = pd.to_numeric(s3["Fine_Amount"], errors="coerce").fillna(0)
        s3 = s3[(s3["_year"] >= yr_range[0]) & (s3["_year"] <= yr_range[1])]
        if sel_state3 != "All": s3 = s3[s3["location"] == sel_state3]
        if not s3.empty:
            grp3 = s3.groupby("_year")["Fine_Amount"].sum().reset_index()
            fig3 = px.line(grp3, x="_year", y="Fine_Amount", title="Total Fine Amount vs Years", markers=True,
                           labels={"_year": "Year", "Fine_Amount": "Total Fine Amount"})
            fig3.update_xaxes(type="category")
            st.plotly_chart(_light_layout(fig3), use_container_width=True)
        else:
            st.info("No data for selected filters.")
    else:
        st.warning("Data lacks 'Fine_Amount' or year information.")

    # Section 4: Custom Trend Exploration
    st.markdown("---")
    st.markdown("### 4) Custom Trend Exploration")
    x_options = {k: v for k, v in {
        "Year": "_year", "Month": "_month", "Day of Week": "_day", "Hour": "_hour",
        "Location": "location", "Violation Type": "violation_type", "Vehicle Type": "vehicle_type",
        "Weather Condition": "Weather_Condition", "Road Condition": "Road_Condition",
        "Registration State": "Registration_State", "License Type": "License_Type", "Driver Gender": "Driver_Gender",
    }.items() if v in work.columns}
    value_options = {k: v for k, v in {
        "Number of Violations (count)": "__count__", "Total Fine Amount": "Fine_Amount",
        "Average Fine Amount": "Fine_Amount__avg", "Average Recorded Speed": "speed",
        "Total Penalty Points": "Penalty_Points", "Average Driver Age": "Driver_Age__avg",
    }.items() if v == "__count__" or v.replace("__avg","") in work.columns}
    color_options = {k: v for k, v in {
        "None": None, "Location": "location", "Violation Type": "violation_type",
        "Vehicle Type": "vehicle_type", "Driver Gender": "Driver_Gender",
        "Registration State": "Registration_State", "Weather Condition": "Weather_Condition",
    }.items() if v is None or v in work.columns}

    r1c1, r1c2 = st.columns(2)
    sel_state4 = r1c1.selectbox("State Filter",          ["All"] + states,               key="t4_state")
    sel_viol4  = r1c2.selectbox("Violation Type Filter", ["All"] + viol_types,            key="t4_viol")
    r2c1, r2c2, r2c3 = st.columns(3)
    x_label     = r2c1.selectbox("X-Axis",           list(x_options.keys()),      key="t4_x")
    val_label   = r2c2.selectbox("Value (Y-Axis)",   list(value_options.keys()),  key="t4_val")
    color_label = r2c3.selectbox("Color / Group By", list(color_options.keys()),  key="t4_color")
    chart_type  = st.selectbox("Graph Type", ["Bar Chart","Line Chart","Pie Chart"], key="t4_chart")

    x_col     = x_options[x_label]
    val_key   = value_options[val_label]
    color_col = color_options[color_label]
    s4 = work.copy()
    if sel_state4 != "All": s4 = s4[s4["location"]     == sel_state4]
    if sel_viol4  != "All": s4 = s4[s4["violation_type"] == sel_viol4]

    if s4.empty:
        st.info("No data for the custom selection.")
    else:
        group_cols = [x_col] + ([color_col] if color_col and color_col != x_col else [])
        if val_key == "__count__":
            agg_df = s4.groupby(group_cols).size().reset_index(name="Count"); y_col_name = "Count"
        elif val_key.endswith("__avg"):
            raw_col = val_key.replace("__avg","")
            s4[raw_col] = pd.to_numeric(s4[raw_col], errors="coerce")
            agg_df = s4.groupby(group_cols)[raw_col].mean().reset_index()
            agg_df = agg_df.rename(columns={raw_col: f"Avg {raw_col}"}); y_col_name = f"Avg {raw_col}"
        else:
            s4[val_key] = pd.to_numeric(s4[val_key], errors="coerce")
            agg_df = s4.groupby(group_cols)[val_key].sum().reset_index(); y_col_name = val_key

        if x_col == "_month":
            agg_df["_month"] = pd.Categorical(agg_df["_month"], categories=MONTH_ORDER, ordered=True)
            agg_df = agg_df.sort_values("_month")
        elif x_col in ("_year","_hour"):
            agg_df = agg_df.sort_values(x_col)

        color_arg  = color_col if color_col else None
        title_text = f"{val_label} by {x_label}" + (f" (grouped by {color_label})" if color_arg else "")

        if chart_type == "Bar Chart":
            fig4 = px.bar(agg_df, x=x_col, y=y_col_name, color=color_arg, title=title_text, barmode="group",
                          labels={x_col: x_label, y_col_name: val_label})
        elif chart_type == "Line Chart":
            fig4 = px.line(agg_df, x=x_col, y=y_col_name, color=color_arg, title=title_text, markers=True,
                           labels={x_col: x_label, y_col_name: val_label})
        else:
            fig4 = px.pie(agg_df, names=color_col or x_col, values=y_col_name, title=title_text)
        st.plotly_chart(_light_layout(fig4), use_container_width=True)


# =============================================================================
# PAGE: DATA VISUALIZATION
# =============================================================================

def render_visualization_page(df: pd.DataFrame) -> None:
    st.title("📊 Data Visualization")
    st.markdown("Deep dive into traffic violation patterns. Each section has its own independent date and state filter.")
    min_date = df["date"].min().date()
    max_date = df["date"].max().date()
    states   = sorted(df["location"].dropna().unique().tolist())

    def get_filters(key_prefix: str, show_state: bool = False):
        col1, col2 = st.columns([1, 1])
        with col1:
            dr = st.date_input("Select Date Range", value=(min_date, max_date),
                               min_value=min_date, max_value=max_date, key=f"{key_prefix}_date")
            start_d, end_d = (dr[0], dr[1]) if isinstance(dr, tuple) and len(dr) == 2 else (min_date, max_date)
        sel_state = "All States"
        if show_state:
            with col2:
                sel_state = st.selectbox("Select State", ["All States"] + states, key=f"{key_prefix}_state")
        return start_d, end_d, sel_state

    # Graph 1: Top 5 Locations
    st.markdown("### 1. Top 5 Locations with Most Violations")
    sd1, ed1, _ = get_filters("g1")
    df1 = df[(df["date"].dt.date >= sd1) & (df["date"].dt.date <= ed1)]
    if not df1.empty:
        lc = df1.groupby("location").size().reset_index(name="count").sort_values("count", ascending=False).head(5)
        fig1 = px.bar(lc, x="location", y="count", color="count", color_continuous_scale="Viridis",
                      labels={"location":"State","count":"Violation Count"}, title=f"Top 5 States ({sd1} to {ed1})")
        st.plotly_chart(_light_layout(fig1), use_container_width=True)

    # Graph 2: Violations by Type
    st.markdown("---"); st.markdown("### 2. Violations by Type")
    sd2, ed2, state2 = get_filters("g2", show_state=True)
    df2 = df[(df["date"].dt.date >= sd2) & (df["date"].dt.date <= ed2)]
    if state2 != "All States": df2 = df2[df2["location"] == state2]
    if not df2.empty:
        v2 = df2.groupby("violation_type").size().reset_index(name="count").sort_values("count", ascending=False)
        fig2 = px.bar(v2, x="violation_type", y="count", color="violation_type",
                      title=f"Violation Distribution: {state2}", labels={"violation_type":"Type","count":"Count"})
        st.plotly_chart(_light_layout(fig2), use_container_width=True)

    # Graph 3: Violation Percentage (Pie)
    st.markdown("---"); st.markdown("### 3. Violation Percentage")
    sd3, ed3, state3 = get_filters("g3", show_state=True)
    df3 = df[(df["date"].dt.date >= sd3) & (df["date"].dt.date <= ed3)]
    if state3 != "All States": df3 = df3[df3["location"] == state3]
    if not df3.empty:
        v3  = df3.groupby("violation_type").size().reset_index(name="count")
        fig3 = px.pie(v3, names="violation_type", values="count", title=f"Category Split: {state3}", hole=0.4)
        fig3.update_traces(textinfo="percent+label")
        st.plotly_chart(_light_layout(fig3), use_container_width=True)

    # Graph 4: Age Group vs Risk Level
    st.markdown("---"); st.markdown("### 4. Age Group vs Risk Level")
    sd4, ed4, _ = get_filters("g4")
    df4 = df[(df["date"].dt.date >= sd4) & (df["date"].dt.date <= ed4)].copy()
    if not df4.empty:
        risk_df  = predict_risk_clusters(df)
        risk_map = dict(zip(risk_df["location"], risk_df["risk_level"]))
        df4["risk_level"] = df4["location"].map(risk_map).fillna("Low")
        df4["age_group"]  = pd.cut(df4["Driver_Age"], bins=[0,18,25,40,60,100], labels=["<18","18-25","26-40","41-60","60+"])
        age_risk = df4.groupby(["age_group","risk_level"], observed=True).size().reset_index(name="count")
        fig4 = px.bar(age_risk, x="age_group", y="count", color="risk_level", barmode="group",
                      title="Risk Level Profile Across Age Groups",
                      color_discrete_map={"High":"#ef4444","Medium":"#f59e0b","Low":"#10b981"})
        st.plotly_chart(_light_layout(fig4), use_container_width=True)

    # Graph 5: Over-speeding by Road Condition
    st.markdown("---"); st.markdown("### 5. Over-speeding by Road Condition")
    sd5, ed5, state5 = get_filters("g5", show_state=True)
    df5 = df[(df["date"].dt.date >= sd5) & (df["date"].dt.date <= ed5)].copy()
    if state5 != "All States": df5 = df5[df5["location"] == state5]
    os_df = df5[df5["violation_type"].str.contains("speeding", case=False, na=False)].copy()
    if not os_df.empty:
        road_mapping = {"Potholes":"Pothole Road","Under Construction":"1-Lane Road",
                        "Slippery":"2-Lane Road","Wet":"3-Lane Road","Dry":"Normal Road"}
        os_df["road_type_mapped"] = os_df["Road_Condition"].map(road_mapping).fillna("Other")
        road_os = os_df.groupby("road_type_mapped").size().reset_index(name="count").sort_values("count")
        fig5 = px.bar(road_os, y="road_type_mapped", x="count", orientation="h", color="count",
                      color_continuous_scale="Reds", title=f"Over-speeding Intensity: {state5}",
                      labels={"road_type_mapped":"Road Conditions","count":"Incident Count"})
        st.plotly_chart(_light_layout(fig5), use_container_width=True)
    else:
        st.info("No over-speeding incidents found.")


# =============================================================================
# PAGE: PREDICTION
# =============================================================================

def render_prediction_module(source_df: pd.DataFrame) -> None:
    st.markdown("### Prediction")
    st.caption("Estimate whether a driver is likely to pay their traffic fine.")

    model_bundle = get_payment_model(source_df)
    if model_bundle is None:
        st.warning("The current dataset does not contain enough valid `Fine_Paid` training data.")
        return

    metrics = model_bundle["metrics"]
    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Model Accuracy", f"{metrics['accuracy']*100:.1f}%")
    mc2.metric("ROC AUC",        f"{metrics['roc_auc']:.3f}")
    mc3.metric("Training Rows",  f"{metrics['train_rows']:,}")

    with st.form("fine_payment_prediction_form"):
        st.markdown("#### Driver & Violation Details")
        options        = model_bundle["choices"]
        numeric_ranges = model_bundle["numeric_ranges"]

        r1 = st.columns(3)
        vehicle_type       = r1[0].selectbox("Vehicle Type",        options["Vehicle_Type"])
        registration_state = r1[1].selectbox("Registration State",  options["Registration_State"])
        driver_age         = r1[2].number_input("Driver Age",        min_value=numeric_ranges["Driver_Age"]["min"],         max_value=numeric_ranges["Driver_Age"]["max"],         value=numeric_ranges["Driver_Age"]["default"],         step=1.0)

        r2 = st.columns(3)
        license_type       = r2[0].selectbox("License Type",        options["License_Type"])
        penalty_points     = r2[1].number_input("Penalty Points",   min_value=numeric_ranges["Penalty_Points"]["min"],      max_value=numeric_ranges["Penalty_Points"]["max"],      value=numeric_ranges["Penalty_Points"]["default"],      step=1.0)
        weather_condition  = r2[2].selectbox("Weather Condition",   options["Weather_Condition"])

        r3 = st.columns(3)
        speed_limit         = r3[0].number_input("Speed Limit",      min_value=numeric_ranges["Speed_Limit"]["min"],         max_value=numeric_ranges["Speed_Limit"]["max"],         value=numeric_ranges["Speed_Limit"]["default"],         step=1.0)
        recorded_speed      = r3[1].number_input("Recorded Speed",   min_value=numeric_ranges["Recorded_Speed"]["min"],      max_value=numeric_ranges["Recorded_Speed"]["max"],      value=numeric_ranges["Recorded_Speed"]["default"],      step=1.0)
        previous_violations = r3[2].number_input("Previous Violations", min_value=numeric_ranges["Previous_Violations"]["min"], max_value=numeric_ranges["Previous_Violations"]["max"], value=numeric_ranges["Previous_Violations"]["default"], step=1.0)

        submitted = st.form_submit_button("Run Prediction", use_container_width=True)

    if submitted:
        payload = {
            "Vehicle_Type": vehicle_type, "Registration_State": registration_state,
            "Driver_Age": driver_age, "License_Type": license_type,
            "Penalty_Points": penalty_points, "Weather_Condition": weather_condition,
            "Speed_Limit": speed_limit, "Recorded_Speed": recorded_speed,
            "Previous_Violations": previous_violations,
        }
        prediction = predict_fine_payment_status(model_bundle, payload)
        rc1, rc2 = st.columns([1.2, 1])
        if prediction["pay_probability"] >= 50:
            rc1.success(f"Prediction: {prediction['label']} ({prediction['pay_probability']:.2f}% probability)")
        else:
            rc1.error(f"Prediction: {prediction['label']} ({prediction['non_pay_probability']:.2f}% non-payment probability)")
        rc2.metric("Pay Probability", f"{prediction['pay_probability']:.2f}%")
        st.progress(min(max(prediction["pay_probability"] / 100, 0.0), 1.0))


# =============================================================================
# PAGE: REPORTS & DOWNLOADS
# =============================================================================

def render_reports_page(filtered_df: pd.DataFrame) -> None:
    st.markdown("## Reports & Downloads")

    current_locations      = st.session_state.get("dashboard_locations", [])
    current_violation_types = st.session_state.get("dashboard_violation_types", [])
    current_filters = (tuple(current_locations), tuple(current_violation_types))

    if "saved_filters"  not in st.session_state: st.session_state.saved_filters  = current_filters
    if "report_saved"   not in st.session_state: st.session_state.report_saved   = False
    if st.session_state.saved_filters != current_filters: st.session_state.report_saved = False

    if st.button("Save", type="primary"):
        st.session_state.report_saved  = True
        st.session_state.saved_filters = current_filters

    if st.session_state.report_saved:
        st.success("Filters saved! You can now download the reports.")
        report_df = build_report_table(filtered_df)
        csv_bytes = dataframe_to_csv_bytes(filtered_df)
        pdf_bytes = build_pdf_report(report_df, filtered_df)
        col1, col2 = st.columns(2)
        col1.download_button("Download filtered data (CSV)", data=csv_bytes,
                             file_name="traffic_violation_filtered_report.csv", mime="text/csv", use_container_width=True)
        col2.download_button("Download summary report (PDF)", data=pdf_bytes,
                             file_name="traffic_violation_summary.pdf", mime="application/pdf", use_container_width=True)
        st.dataframe(report_df.rename(columns={"metric":"Metric","value":"Value"}),
                     use_container_width=True, hide_index=True)
    else:
        st.info("Pending filter changes detected. Please click 'Save' to apply the current filters.")


# =============================================================================
# PAGE: UPLOAD
# =============================================================================

def render_upload_page(active_df: pd.DataFrame, active_source_label: str) -> None:
    st.markdown("""
        <div class="hero-banner">
            <div class="hero-title">Upload Dataset</div>
            <div class="hero-copy">Drag and drop a CSV file to replace the active dataset.</div>
        </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload a CSV dataset", type=["csv"], key="upload_dataset_page")
    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        try:
            uploaded_df = load_data_from_upload(file_bytes)
            st.session_state["uploaded_dataset_bytes"] = file_bytes
            st.session_state["uploaded_dataset_name"]  = uploaded_file.name
            active_df           = uploaded_df
            active_source_label = f"Uploaded dataset: {uploaded_file.name}"
            st.success(f"Dataset uploaded successfully: {uploaded_file.name}")
        except Exception as exc:
            st.error(f"Unable to process uploaded file: {exc}")

    ic1, ic2 = st.columns([1.35, 1])
    with ic1:
        st.markdown(f"**Active Source:** {active_source_label}")
        st.caption("The currently active dataset is used across the dashboard, analytics, predictions, and reports pages.")
    with ic2:
        if st.button("Use Local dataset.csv", use_container_width=True):
            st.session_state.pop("uploaded_dataset_bytes", None)
            st.session_state.pop("uploaded_dataset_name", None)
            st.rerun()

    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Rows",      f"{len(active_df):,}")
    mc2.metric("Columns",   f"{len(active_df.columns):,}")
    mc3.metric("Locations", f"{active_df['location'].nunique():,}" if "location" in active_df.columns else "0")

    st.markdown("### Dataset Preview")
    preview_cols = [c for c in ["date","location","violation_type","vehicle_type","speed"] if c in active_df.columns]
    st.dataframe(active_df[preview_cols].head(15) if preview_cols else active_df.head(15), use_container_width=True, hide_index=True)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main() -> None:
    inject_styles()

    default_path        = Path("dataset.csv")
    active_source_label = "Local dataset: dataset.csv"

    try:
        uploaded_bytes = st.session_state.get("uploaded_dataset_bytes")
        if uploaded_bytes:
            df           = load_data_from_upload(uploaded_bytes)
            uploaded_name = st.session_state.get("uploaded_dataset_name", "uploaded_file.csv")
            active_source_label = f"Uploaded dataset: {uploaded_name}"
            st.sidebar.caption(f"Using uploaded dataset: `{uploaded_name}`")
        else:
            df = load_data_from_path(str(default_path))
            st.sidebar.caption("Using local dataset: `dataset.csv`")
    except Exception as exc:
        st.error(f"Unable to load dataset: {exc}")
        st.stop()

    if df.empty:
        st.warning("No valid records were found in the dataset.")
        st.stop()

    page = render_sidebar_nav()

    filtered_df         = df
    pages_with_filters  = {PAGE_DASHBOARD, PAGE_ADVANCED, PAGE_REPORTS}

    if page in pages_with_filters:
        if page == PAGE_DASHBOARD: render_dashboard_header()
        elif page == PAGE_ADVANCED: render_analytics_header()
        filter_title = "Analytics Filter" if page == PAGE_ADVANCED else "Dashboard Filters"
        start_date, end_date, locations, violation_types = render_dashboard_filters(df, filter_title)
        filtered_df = apply_filters(df, start_date, end_date, locations, violation_types)
        if filtered_df.empty:
            st.warning("No records match the selected filters. Try broadening your selection.")
            st.stop()

    if   page == PAGE_HOME:          render_home_page()
    elif page == PAGE_DASHBOARD:     render_dashboard(filtered_df)
    elif page == PAGE_ADVANCED:      render_advanced_analytics(filtered_df)
    elif page == PAGE_PREDICTIONS:   render_prediction_module(df)
    elif page == PAGE_REPORTS:       render_reports_page(filtered_df)
    elif page == PAGE_UPLOAD:        render_upload_page(df, active_source_label)
    elif page == PAGE_VISUALIZATION: render_visualization_page(df)
    elif page == PAGE_TRENDS:        render_trend_analysis_page(df)


if __name__ == "__main__":
    main()
