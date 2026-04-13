# =============================================================================
# config.py — App Constants & Configuration
# Smart Traffic Violation Pattern Detector
# =============================================================================

from __future__ import annotations

# ---------------------------------------------------------------------------
# App Identity
# ---------------------------------------------------------------------------
APP_TITLE = "Traffic Violation Pattern Detector"
APP_ICON = "\U0001F6A6"

# ---------------------------------------------------------------------------
# Page Names
# ---------------------------------------------------------------------------
PAGE_HOME        = "\U0001F3E0 Home"
PAGE_DASHBOARD   = "\U0001F4CA Dashboard"
PAGE_ADVANCED    = "\U0001F4C8 Analytics"
PAGE_PREDICTIONS = "\U0001F52E Prediction"
PAGE_REPORTS     = "\U0001F4C4 Reports & Downloads"
PAGE_UPLOAD      = "\U0001F4E4 Upload File"
PAGE_VISUALIZATION = "\U0001F5A5\uFE0F Data Visualization"
PAGE_TRENDS      = "\U0001F4C9 Trend Analysis"

NAV_ITEMS = [
    PAGE_HOME,
    PAGE_DASHBOARD,
    PAGE_ADVANCED,
    PAGE_VISUALIZATION,
    PAGE_TRENDS,
    PAGE_PREDICTIONS,
    PAGE_REPORTS,
    PAGE_UPLOAD,
]

# ---------------------------------------------------------------------------
# Metric Card Styles
# ---------------------------------------------------------------------------
METRIC_STYLES = {
    "total":   {"bg": "linear-gradient(135deg, #ef4444 0%, #be123c 100%)", "icon": "\U0001F6A8"},
    "risk":    {"bg": "linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%)", "icon": "\U0001F4CD"},
    "pending": {"bg": "linear-gradient(135deg, #10b981 0%, #047857 100%)", "icon": "\u23F3"},
    "speed":   {"bg": "linear-gradient(135deg, #7c3aed 0%, #4338ca 100%)", "icon": "\u26A1"},
}

# ---------------------------------------------------------------------------
# Dataset Column Mapping (raw CSV → normalized internal names)
# ---------------------------------------------------------------------------
DATASET_COLUMN_MAP = {
    "Date":                 "date",
    "Location":             "location",
    "Violation_Type":       "violation_type",
    "Vehicle_Type":         "vehicle_type",
    "Recorded_Speed":       "speed",
    "Helmet_Worn":          "helmet_detected",
    "Traffic_Light_Status": "signal_status",
    "Time":                 "time",
}

REQUIRED_COLUMNS = [
    "date",
    "location",
    "violation_type",
    "vehicle_type",
    "speed",
    "helmet_detected",
    "signal_status",
]

# ---------------------------------------------------------------------------
# ML Model: Fine Payment Prediction
# ---------------------------------------------------------------------------
PAYMENT_FEATURES = [
    "Vehicle_Type",
    "Registration_State",
    "Driver_Age",
    "License_Type",
    "Penalty_Points",
    "Weather_Condition",
    "Speed_Limit",
    "Recorded_Speed",
    "Previous_Violations",
]
PAYMENT_TARGET = "Fine_Paid"

PAYMENT_COLUMN_ALIASES = {
    "Vehicle_Type":        ["Vehicle_Type", "vehicle_type"],
    "Registration_State":  ["Registration_State"],
    "Driver_Age":          ["Driver_Age"],
    "License_Type":        ["License_Type"],
    "Penalty_Points":      ["Penalty_Points"],
    "Weather_Condition":   ["Weather_Condition"],
    "Speed_Limit":         ["Speed_Limit"],
    "Recorded_Speed":      ["Recorded_Speed", "speed"],
    "Previous_Violations": ["Previous_Violations"],
    "Fine_Paid":           ["Fine_Paid"],
}

# ---------------------------------------------------------------------------
# Trend Analysis
# ---------------------------------------------------------------------------
MONTH_ORDER = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]
