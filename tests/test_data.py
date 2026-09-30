from app.data import load_tickets


def test_loads_and_derives_columns():
    frame = load_tickets("data/support_tickets.csv")
    assert len(frame) == 500
    assert frame.attrs["reference_time"] == "2024-03-30 18:06:00"
    assert frame["is_resolved"].sum() == 327
    assert frame["resolution_time_hrs"].isna().sum() == 173
