import os
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = [
    "ticket_id",
    "created_at",
    "category",
    "priority",
    "status",
    "response_time_hrs",
    "resolution_time_hrs",
    "agent_id",
    "customer_rating",
    "issue_summary",
]
NUMERIC_COLUMNS = ("response_time_hrs", "resolution_time_hrs", "customer_rating")
ALLOWED_VALUES = {
    "category": {"Billing", "Technical", "General"},
    "priority": {"Low", "Medium", "High", "Critical"},
    "status": {"Open", "Resolved", "Escalated"},
}


def load_tickets(
    path: str | Path | None = None, reference_time: str | None = None
) -> pd.DataFrame:
    """Read the CSV, validate its contract, and add derived columns."""
    source = path or os.getenv("DATA_PATH", "data/support_tickets.csv")
    frame = pd.read_csv(Path(source))
    _validate_columns(frame)

    frame = frame[REQUIRED_COLUMNS].copy()
    _coerce_columns(frame)
    _validate_allowed_values(frame)
    reference = _resolve_reference_time(frame, reference_time)

    frame["is_resolved"] = frame["status"].eq("Resolved")
    frame["age_hrs"] = (reference - frame["created_at"]).dt.total_seconds() / 3600
    frame.attrs["reference_time"] = reference.isoformat(sep=" ")
    return frame


def _validate_columns(frame: pd.DataFrame) -> None:
    missing = set(REQUIRED_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def _coerce_columns(frame: pd.DataFrame) -> None:
    frame["created_at"] = pd.to_datetime(frame["created_at"], errors="coerce")
    if frame["created_at"].isna().any():
        raise ValueError("created_at contains invalid timestamps")

    for column in NUMERIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")


def _validate_allowed_values(frame: pd.DataFrame) -> None:
    for column, values in ALLOWED_VALUES.items():
        invalid = set(frame[column].dropna()) - values
        if invalid:
            raise ValueError(f"Invalid {column} values: {sorted(invalid)}")


def _resolve_reference_time(
    frame: pd.DataFrame,
    reference_time: str | None,
) -> pd.Timestamp:
    configured = reference_time or os.getenv("REFERENCE_TIME")
    return pd.Timestamp(configured) if configured else frame["created_at"].max()
