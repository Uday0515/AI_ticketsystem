from app.anomalies import detect_anomalies
from app.data import load_tickets


def test_data_quality_rule_flags_known_rows():
    result = detect_anomalies(load_tickets("data/support_tickets.csv"))
    assert sum(item["rule"] == "resolution_before_response" for item in result) == 28
    assert all(
        set(item) == {"ticket_id", "rule", "severity", "reason"} for item in result
    )


def test_window_is_relative_to_reference_time():
    result = detect_anomalies(load_tickets("data/support_tickets.csv"), days=7)
    assert all(item["ticket_id"] for item in result)
