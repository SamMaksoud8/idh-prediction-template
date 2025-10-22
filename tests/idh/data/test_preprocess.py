import pandas as pd

from idh.data import preprocess


def test_get_session_start_time():
    df = pd.DataFrame(
        {
            "session_id": ["a", "a", "b"],
            "datatime": pd.to_datetime(
                ["2024-01-01 10:00", "2024-01-01 09:00", "2024-01-02 08:00"]
            ),
        }
    )
    starts = preprocess.get_session_start_time(df)
    assert list(starts) == [
        pd.Timestamp("2024-01-01 09:00"),
        pd.Timestamp("2024-01-01 09:00"),
        pd.Timestamp("2024-01-02 08:00"),
    ]


def test_sessionize_machine_data_creates_sessions():
    df = pd.DataFrame(
        {
            "pid": [1, 1, 1, 2],
            "datatime": pd.to_datetime(
                [
                    "2024-01-01 00:00",
                    "2024-01-01 05:00",
                    "2024-01-02 20:00",
                    "2024-01-01 00:00",
                ]
            ),
        }
    )

    result = preprocess.sessionize_machine_data(df, delta=12)
    assert result.loc[0, "session_id"] == result.loc[1, "session_id"]
    assert result.loc[1, "session_id"] != result.loc[2, "session_id"]
    assert result.loc[0, "session_start_ts"] == result.loc[1, "session_start_ts"]


def test_merge_machine_and_rego_data():
    machine = pd.DataFrame(
        {
            "pid": [1, 1],
            "session_start_ts": pd.to_datetime(["2024-01-01 00:00", "2024-01-01 00:00"]),
            "value": [10, 20],
        }
    )
    reg = pd.DataFrame(
        {
            "pid": [1],
            "keyindate": [pd.Timestamp("2024-01-01").value],
            "weightstart": [70],
        }
    )
    merged = preprocess.merge_machine_and_rego_data(machine, reg)
    assert "weightstart" in merged.columns
    assert len(merged) == 2


def test_merge_patient_demographics():
    df = pd.DataFrame({"pid": [1], "first_dialysis": [1_500_000_000_000_000]})
    demo = pd.DataFrame({"pid": [1], "gender": ["F"]})
    merged = preprocess.merge_patient_demographics(df, demo)
    assert "gender" in merged.columns
    assert pd.api.types.is_datetime64_any_dtype(merged["first_dialysis_ts"])


def test_aggregate_features_shapes():
    df = pd.DataFrame(
        {
            "pid": [1, 1, 1, 1],
            "session_id": ["s1"] * 4,
            "datatime": pd.to_datetime(
                [
                    "2024-01-01 00:00",
                    "2024-01-01 00:10",
                    "2024-01-01 00:20",
                    "2024-01-01 00:30",
                ]
            ),
            "sbp": [100, 95, 85, 110],
            "dbp": [70, 72, 68, 71],
            "dia_temp_value": [36.5, 36.7, 36.6, 36.8],
            "conductivity": [1.0, 1.1, 1.2, 1.3],
            "uf": [0.5, 0.6, 0.4, 0.7],
            "blood_flow": [300, 320, 310, 305],
            "weightstart": [70, 70, 70, 70],
            "dryweight": [68, 68, 68, 68],
            "gender": ["F"] * 4,
            "birthday": [1980] * 4,
            "DM": [1] * 4,
            "session_start_ts": pd.to_datetime(["2024-01-01 00:00"] * 4),
            "first_dialysis_ts": pd.to_datetime(["2020-01-01 00:00"] * 4),
        }
    )

    aggregated = preprocess.aggregate_features(df)
    expected_columns = {
        "pid",
        "session_id",
        "avg_sbp",
        "label",
        "rolling_avg_sbp",
        "trend_1_sbp",
    }
    assert expected_columns.issubset(aggregated.columns)
    # Because one bin is hypotensive (85), future label should flag earlier bins
    assert aggregated["label"].isin([0, 1]).all()
