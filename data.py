# =============================================================================
# data.py — Data Loading, Normalization, Caching & Export Utilities
# Smart Traffic Violation Pattern Detector
# =============================================================================

from __future__ import annotations

from io import BytesIO, StringIO
from pathlib import Path
from typing import BinaryIO

import pandas as pd
import streamlit as st
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

from config import DATASET_COLUMN_MAP, REQUIRED_COLUMNS


# =============================================================================
# 1. CSV LOADING & NORMALIZATION
# =============================================================================

def _read_csv(source: str | Path | BytesIO | BinaryIO) -> pd.DataFrame:
    if hasattr(source, "read"):
        raw = source.read()
        if isinstance(raw, bytes):
            return pd.read_csv(BytesIO(raw))
        return pd.read_csv(StringIO(raw))
    return pd.read_csv(source)


def load_dataset(source: str | Path | BytesIO | BinaryIO) -> pd.DataFrame:
    """Load and normalize a traffic violation dataset from a file path or bytes."""
    df = _read_csv(source)
    df = df.rename(columns=DATASET_COLUMN_MAP)

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            "Dataset is missing required columns after normalization: "
            + ", ".join(missing)
        )

    if "time" not in df.columns:
        df["time"] = "00:00"

    normalized = df.copy()
    normalized["date"] = pd.to_datetime(normalized["date"], errors="coerce", dayfirst=True)
    normalized["time"] = normalized["time"].fillna("00:00").astype(str).str.strip()
    normalized["event_time"] = pd.to_datetime(
        normalized["date"].dt.strftime("%Y-%m-%d") + " " + normalized["time"],
        errors="coerce",
    )
    normalized["location"]      = normalized["location"].fillna("Unknown").astype(str).str.strip()
    normalized["violation_type"] = normalized["violation_type"].fillna("Unknown").astype(str).str.strip()
    normalized["vehicle_type"]   = normalized["vehicle_type"].fillna("Unknown").astype(str).str.strip()
    normalized["speed"]          = pd.to_numeric(normalized["speed"], errors="coerce").fillna(0)
    normalized["helmet_detected"] = (
        normalized["helmet_detected"]
        .fillna("Unknown").astype(str).str.strip()
        .replace({"Yes": "Detected", "No": "Not Detected", "N/A": "Unknown"})
    )
    normalized["signal_status"] = normalized["signal_status"].fillna("Unknown").astype(str).str.strip()
    normalized["hour"]     = normalized["event_time"].dt.hour.fillna(0).astype(int)
    normalized["day_name"] = normalized["date"].dt.day_name().fillna("Unknown")
    normalized["month"]    = normalized["date"].dt.to_period("M").astype(str)

    normalized = normalized.dropna(subset=["date"]).reset_index(drop=True)
    return normalized


def validate_dataset(df: pd.DataFrame) -> tuple[bool, list[str]]:
    """Return (is_valid, list_of_missing_columns)."""
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    return not missing, missing


# =============================================================================
# 2. STREAMLIT CACHED DATA SERVICES
# =============================================================================

@st.cache_data(show_spinner=False)
def load_data_from_path(path: str) -> pd.DataFrame:
    return load_dataset(path)


@st.cache_data(show_spinner=False)
def load_data_from_upload(file_bytes: bytes) -> pd.DataFrame:
    return load_dataset(BytesIO(file_bytes))


@st.cache_resource(
    show_spinner=False,
    hash_funcs={pd.DataFrame: lambda df: int(pd.util.hash_pandas_object(df, index=True).sum())},
)
def get_payment_model(df: pd.DataFrame):
    from analysis import train_fine_payment_model
    return train_fine_payment_model(df)


# =============================================================================
# 3. EXPORT UTILITIES
# =============================================================================

def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Serialize a DataFrame to UTF-8 encoded CSV bytes."""
    return df.to_csv(index=False).encode("utf-8")


def build_pdf_report(summary_df: pd.DataFrame, filtered_df: pd.DataFrame) -> bytes:
    """Generate a simple A4 PDF summary report using matplotlib."""
    buffer = BytesIO()

    with PdfPages(buffer) as pdf:
        fig, ax = plt.subplots(figsize=(8.27, 11.69))
        ax.axis("off")
        ax.set_title(
            "Traffic Violation Pattern Detector Report",
            fontsize=16, fontweight="bold", pad=20,
        )

        report_lines = [f"{row['metric']}: {row['value']}" for _, row in summary_df.iterrows()]
        report_lines.extend([
            "",
            f"Filtered records: {len(filtered_df)}",
            f"Locations covered: {filtered_df['location'].nunique() if not filtered_df.empty else 0}",
            f"Violation types covered: {filtered_df['violation_type'].nunique() if not filtered_df.empty else 0}",
        ])

        ax.text(0.05, 0.95, "\n".join(report_lines), va="top", ha="left",
                fontsize=11, family="monospace")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    buffer.seek(0)
    return buffer.read()


# =============================================================================
# 4. PAGE HELPER UTILITIES
# =============================================================================

def build_context(filtered_df: pd.DataFrame, risk_df: pd.DataFrame) -> dict[str, object]:
    """Build KPI context dict for metric cards."""
    payment_series = (
        filtered_df["Fine_Paid"].astype(str).str.strip().str.lower()
        if "Fine_Paid" in filtered_df.columns
        else pd.Series(dtype=str)
    )
    pending_count   = int(payment_series.eq("no").sum()) if not payment_series.empty else 0
    high_risk_count = int(risk_df["risk_level"].eq("High").sum()) if not risk_df.empty else 0
    return {
        "total":   int(len(filtered_df)),
        "risk":    high_risk_count,
        "pending": pending_count,
        "speed":   round(float(filtered_df["speed"].mean()), 1) if not filtered_df.empty else 0.0,
    }


def summarize_counts(
    df: pd.DataFrame, column: str, top_n: int | None = None
) -> pd.DataFrame:
    """Return a value-count summary DataFrame for the given column."""
    if column not in df.columns:
        return pd.DataFrame(columns=[column, "count"])

    summary = (
        df[column]
        .fillna("Unknown").astype(str).str.strip()
        .replace({"": "Unknown"})
        .value_counts()
        .rename_axis(column)
        .reset_index(name="count")
    )
    if top_n is not None:
        summary = summary.head(top_n)
    return summary
