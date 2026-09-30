import pandas as pd
from typing import Any


OVERDUE_RULE = "unresolved_priority_over_24h"
OUTLIER_RULE = "resolution_time_iqr_outlier"
TIMING_RULE = "resolution_before_response"
HIGH_PRIORITIES = {"High", "Critical"}


def detect_anomalies(
    frame: pd.DataFrame, days: int | None = None
) -> list[dict[str, Any]]:
    """Return one record per ticket/rule, optionally limited to recent tickets."""
    scoped = _scope_by_age(frame, days)
    anomalies = []
    anomalies.extend(_find_overdue_tickets(scoped))
    anomalies.extend(_find_resolution_outliers(scoped))
    anomalies.extend(_find_invalid_timing(scoped))
    return anomalies


def _scope_by_age(frame: pd.DataFrame, days: int | None) -> pd.DataFrame:
    if days is None:
        return frame

    reference = pd.Timestamp(
        frame.attrs.get("reference_time", frame["created_at"].max())
    )
    cutoff = reference - pd.Timedelta(days=days)
    return frame[frame["created_at"] >= cutoff]


def _find_overdue_tickets(frame: pd.DataFrame) -> list[dict[str, Any]]:
    overdue = frame[
        (~frame["is_resolved"])
        & frame["priority"].isin(HIGH_PRIORITIES)
        & (frame["age_hrs"] > 24)
    ]
    return [
        _item(
            row,
            OVERDUE_RULE,
            "high" if row.priority == "High" else "critical",
            f"{row.priority} ticket unresolved for {row.age_hrs:.1f} hours",
        )
        for _, row in overdue.iterrows()
    ]


def _find_resolution_outliers(frame: pd.DataFrame) -> list[dict[str, Any]]:
    resolved = frame[frame["resolution_time_hrs"].notna()]
    anomalies: list[dict[str, Any]] = []

    for priority, group in resolved.groupby("priority"):
        if len(group) < 4:
            continue

        quartiles = group["resolution_time_hrs"].quantile([0.25, 0.75])
        upper_bound = quartiles[0.75] + 1.5 * (quartiles[0.75] - quartiles[0.25])
        outliers = group[group["resolution_time_hrs"] > upper_bound]

        for _, row in outliers.iterrows():
            anomalies.append(
                _item(
                    row,
                    OUTLIER_RULE,
                    "medium",
                    f"{row.resolution_time_hrs:.1f} hours exceeds "
                    f"{priority} priority IQR upper bound of {upper_bound:.1f}",
                )
            )

    return anomalies


def _find_invalid_timing(frame: pd.DataFrame) -> list[dict[str, Any]]:
    invalid_timing = frame[
        frame["resolution_time_hrs"].notna()
        & (frame["resolution_time_hrs"] < frame["response_time_hrs"])
    ]
    return [
        _item(
            row,
            TIMING_RULE,
            "high",
            f"resolution time {row.resolution_time_hrs:.1f}h is below "
            f"response time {row.response_time_hrs:.1f}h",
        )
        for _, row in invalid_timing.iterrows()
    ]


def _item(row: pd.Series, rule: str, severity: str, reason: str) -> dict[str, Any]:
    return {
        "ticket_id": row["ticket_id"],
        "rule": rule,
        "severity": severity,
        "reason": reason,
    }
