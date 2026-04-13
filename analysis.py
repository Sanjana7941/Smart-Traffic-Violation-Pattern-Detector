# =============================================================================
# analysis.py — Analytics, Pattern Detection, Clustering & ML Model
# Smart Traffic Violation Pattern Detector
# =============================================================================

from __future__ import annotations

import os
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import PAYMENT_FEATURES, PAYMENT_TARGET, PAYMENT_COLUMN_ALIASES


# =============================================================================
# 1. FILTER
# =============================================================================

def apply_filters(
    df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    locations: list[str],
    violation_types: list[str],
) -> pd.DataFrame:
    """Filter DataFrame by date range, locations, and violation types."""
    start_ts = pd.to_datetime(start_date)
    end_ts   = pd.to_datetime(end_date)
    filtered = df[(df["date"] >= start_ts) & (df["date"] <= end_ts)].copy()

    if locations:
        filtered = filtered[filtered["location"].isin(locations)]
    if violation_types:
        filtered = filtered[filtered["violation_type"].isin(violation_types)]

    return filtered.reset_index(drop=True)


# =============================================================================
# 2. KPI COMPUTATIONS
# =============================================================================

def compute_kpis(df: pd.DataFrame) -> dict[str, float | int | str]:
    total_violations = int(len(df))
    avg_speed        = float(df["speed"].mean()) if total_violations else 0.0
    top_violation    = (
        df["violation_type"].mode().iat[0]
        if total_violations and not df["violation_type"].mode().empty
        else "N/A"
    )
    top_location = (
        df["location"].mode().iat[0]
        if total_violations and not df["location"].mode().empty
        else "N/A"
    )
    return {
        "total_violations": total_violations,
        "top_violation":    top_violation,
        "top_location":     top_location,
        "avg_speed":        round(avg_speed, 2),
    }


# =============================================================================
# 3. SUMMARY AGGREGATIONS
# =============================================================================

def violation_type_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("violation_type").size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )


def location_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("location").size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )


def violations_over_time(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("date").size().reset_index(name="count").sort_values("date")


def vehicle_type_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("vehicle_type").size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )


def heatmap_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.pivot_table(
            index="location",
            columns="violation_type",
            values="speed",
            aggfunc="count",
            fill_value=0,
        ).sort_index()
    )


# =============================================================================
# 4. PATTERN DETECTION
# =============================================================================

def detect_patterns(df: pd.DataFrame) -> dict[str, object]:
    if df.empty:
        return {
            "peak_hour":           "N/A",
            "peak_day":            "N/A",
            "high_risk_areas":     pd.DataFrame(columns=["location", "count"]),
            "hourly_distribution": pd.DataFrame(columns=["hour", "count"]),
        }

    hourly    = df.groupby("hour").size().reset_index(name="count").sort_values("hour")
    peak_hour = int(hourly.sort_values("count", ascending=False).iloc[0]["hour"])
    peak_day  = df["day_name"].mode().iat[0]
    high_risk_areas = location_summary(df).head(5)

    return {
        "peak_hour":           f"{peak_hour:02d}:00 - {peak_hour:02d}:59",
        "peak_day":            peak_day,
        "high_risk_areas":     high_risk_areas,
        "hourly_distribution": hourly,
    }


# =============================================================================
# 5. RISK CLUSTERING (KMeans)
# =============================================================================

def predict_risk_clusters(df: pd.DataFrame) -> pd.DataFrame:
    """Cluster locations into Low / Medium / High risk groups."""
    if df.empty or df["location"].nunique() < 3:
        return pd.DataFrame(
            columns=["location", "violations", "avg_speed", "peak_hour", "risk_level"]
        )

    features = (
        df.groupby("location")
        .agg(
            violations=("violation_type", "size"),
            avg_speed=("speed", "mean"),
            peak_hour=("hour", lambda s: s.mode().iat[0] if not s.mode().empty else 0),
        )
        .reset_index()
    )

    scaled  = StandardScaler().fit_transform(features[["violations", "avg_speed", "peak_hour"]])
    model   = KMeans(n_clusters=3, random_state=42, n_init=10)
    features["cluster"] = model.fit_predict(scaled)

    cluster_scores = (
        features.groupby("cluster")[["violations", "avg_speed"]]
        .mean().sum(axis=1).sort_values()
    )
    label_order = {
        cluster: level
        for cluster, level in zip(cluster_scores.index, ["Low", "Medium", "High"])
    }
    features["risk_level"] = features["cluster"].map(label_order)

    risk_rank = {"High": 0, "Medium": 1, "Low": 2}
    features["risk_sort"] = features["risk_level"].map(risk_rank)
    features = features.sort_values(
        ["risk_sort", "violations", "avg_speed"], ascending=[True, False, False]
    )
    return features.drop(columns=["cluster", "risk_sort"]).reset_index(drop=True)


# =============================================================================
# 6. REPORT TABLE
# =============================================================================

def build_report_table(df: pd.DataFrame) -> pd.DataFrame:
    patterns = detect_patterns(df)
    summary_rows = [
        {"metric": "Total Violations",     "value": len(df)},
        {"metric": "Most Common Violation","value": df["violation_type"].mode().iat[0] if not df.empty else "N/A"},
        {
            "metric": "Highest Risk Area",
            "value":  patterns["high_risk_areas"].iloc[0]["location"]
                      if not patterns["high_risk_areas"].empty else "N/A",
        },
        {"metric": "Peak Hour", "value": patterns["peak_hour"]},
        {"metric": "Peak Day",  "value": patterns["peak_day"]},
    ]
    return pd.DataFrame(summary_rows)


# =============================================================================
# 7. ML MODEL — Fine Payment Prediction (Random Forest)
# =============================================================================

def _resolve_column(df: pd.DataFrame, canonical_name: str) -> str | None:
    for candidate in PAYMENT_COLUMN_ALIASES[canonical_name]:
        if candidate in df.columns:
            return candidate
    return None


def _build_payment_training_frame(df: pd.DataFrame) -> pd.DataFrame | None:
    resolved = {name: _resolve_column(df, name) for name in PAYMENT_FEATURES + [PAYMENT_TARGET]}
    if any(v is None for v in resolved.values()):
        return None
    model_df = pd.DataFrame()
    for canonical_name, source_col in resolved.items():
        model_df[canonical_name] = df[source_col]
    return model_df


def train_fine_payment_model(df: pd.DataFrame) -> dict[str, object] | None:
    model_df = _build_payment_training_frame(df)
    if model_df is None:
        return None

    model_df = model_df.dropna(subset=[PAYMENT_TARGET])
    model_df[PAYMENT_TARGET] = model_df[PAYMENT_TARGET].astype(str).str.strip().str.lower()
    model_df = model_df[model_df[PAYMENT_TARGET].isin(["yes", "no"])]

    if model_df.empty or model_df[PAYMENT_TARGET].nunique() < 2:
        return None

    y = model_df[PAYMENT_TARGET].map({"no": 0, "yes": 1})
    X = model_df[PAYMENT_FEATURES].copy()

    categorical_features = ["Vehicle_Type", "Registration_State", "License_Type", "Weather_Condition"]
    numeric_features     = ["Driver_Age", "Penalty_Points", "Speed_Limit", "Recorded_Speed", "Previous_Violations"]

    preprocessor = ColumnTransformer(transformers=[
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_features),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical_features),
    ])

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=250, max_depth=12, min_samples_split=4, random_state=42,
        )),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )
    model.fit(X_train, y_train)

    predictions   = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy":      round(float(accuracy_score(y_test, predictions)), 3),
        "roc_auc":       round(float(roc_auc_score(y_test, probabilities)), 3),
        "train_rows":    int(len(X_train)),
        "test_rows":     int(len(X_test)),
        "positive_rate": round(float(y.mean()), 3),
    }
    choices = {
        feature: sorted(model_df[feature].dropna().astype(str).unique().tolist())
        for feature in categorical_features
    }
    numeric_ranges = {
        feature: {
            "min":     float(model_df[feature].min()),
            "max":     float(model_df[feature].max()),
            "default": float(model_df[feature].median()),
        }
        for feature in numeric_features
    }

    return {
        "model":           model,
        "metrics":         metrics,
        "choices":         choices,
        "numeric_ranges":  numeric_ranges,
        "feature_columns": PAYMENT_FEATURES,
    }


def predict_fine_payment_status(
    model_bundle: dict[str, object], input_payload: dict[str, object]
) -> dict[str, object]:
    input_df    = pd.DataFrame([input_payload], columns=model_bundle["feature_columns"])
    probability = float(model_bundle["model"].predict_proba(input_df)[0][1])
    label       = "Likely to Pay" if probability >= 0.5 else "Likely Not to Pay"
    return {
        "label":               label,
        "pay_probability":     round(probability * 100, 2),
        "non_pay_probability": round((1 - probability) * 100, 2),
    }
