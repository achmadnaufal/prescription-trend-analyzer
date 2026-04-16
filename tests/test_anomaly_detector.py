"""
Tests for src/anomaly_detector.py.

Covers happy path, edge cases (empty, single-point, flat series, all-zero),
determinism, method variants, grouping behaviour, and parametrized inputs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.anomaly_detector import AnomalyDetectorError, detect_anomalies


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_df(volumes: list[float], drug: str = "DrugA") -> pd.DataFrame:
    """Build a minimal DataFrame with one drug and the given volumes."""
    return pd.DataFrame(
        {
            "drug_name": [drug] * len(volumes),
            "prescriptions_count": volumes,
        }
    )


def _normal_series() -> list[float]:
    """12-point series with one obvious spike at the end."""
    return [1000, 1020, 980, 1010, 990, 1030, 1005, 995, 1015, 1025, 975, 8000]


# ---------------------------------------------------------------------------
# 1. Happy path — spike detected
# ---------------------------------------------------------------------------


def test_spike_detected_both_method():
    df = _make_df(_normal_series())
    result = detect_anomalies(df, method="both", z_threshold=3.0, iqr_k=1.5)

    flagged = result[result["is_anomaly"]]
    assert len(flagged) >= 1, "Expected the 8 000 spike to be flagged."
    assert flagged["prescriptions_count"].max() == 8000


# ---------------------------------------------------------------------------
# 2. Output columns always present
# ---------------------------------------------------------------------------


def test_output_columns_present():
    df = _make_df(_normal_series())
    result = detect_anomalies(df)
    for col in ("is_anomaly", "anomaly_score", "anomaly_rationale"):
        assert col in result.columns, f"Column '{col}' missing from output."


# ---------------------------------------------------------------------------
# 3. Immutability — caller's DataFrame is not mutated
# ---------------------------------------------------------------------------


def test_immutability():
    df = _make_df(_normal_series())
    original_cols = list(df.columns)
    original_len = len(df)
    detect_anomalies(df)
    assert list(df.columns) == original_cols
    assert len(df) == original_len
    assert "is_anomaly" not in df.columns


# ---------------------------------------------------------------------------
# 4. Determinism — same input → same output every time
# ---------------------------------------------------------------------------


def test_determinism():
    df = _make_df(_normal_series())
    r1 = detect_anomalies(df)
    r2 = detect_anomalies(df)
    pd.testing.assert_frame_equal(r1, r2)


# ---------------------------------------------------------------------------
# 5. Flat series — no anomalies (all values identical)
# ---------------------------------------------------------------------------


def test_flat_series_no_anomalies():
    df = _make_df([500.0] * 12)
    result = detect_anomalies(df)
    assert result["is_anomaly"].sum() == 0, "Flat series should produce no anomalies."


# ---------------------------------------------------------------------------
# 6. All-zero series — no anomalies
# ---------------------------------------------------------------------------


def test_all_zero_series():
    df = _make_df([0.0] * 10)
    result = detect_anomalies(df)
    assert result["is_anomaly"].sum() == 0


# ---------------------------------------------------------------------------
# 7. Fewer rows than min_periods — no flags, NaN scores
# ---------------------------------------------------------------------------


def test_insufficient_data_no_flags():
    df = _make_df([100.0, 200.0, 900.0], drug="Small")  # 3 rows, min_periods=4
    result = detect_anomalies(df, min_periods=4)
    assert result["is_anomaly"].sum() == 0
    assert result["anomaly_score"].isna().all()


# ---------------------------------------------------------------------------
# 8. Empty DataFrame raises AnomalyDetectorError
# ---------------------------------------------------------------------------


def test_empty_dataframe_raises():
    df = pd.DataFrame(columns=["drug_name", "prescriptions_count"])
    with pytest.raises(AnomalyDetectorError, match="empty"):
        detect_anomalies(df)


# ---------------------------------------------------------------------------
# 9. Missing value_col raises AnomalyDetectorError
# ---------------------------------------------------------------------------


def test_missing_value_col_raises():
    df = pd.DataFrame({"drug_name": ["A"], "volume": [100]})
    with pytest.raises(AnomalyDetectorError, match="prescriptions_count"):
        detect_anomalies(df)


# ---------------------------------------------------------------------------
# 10. Invalid method raises AnomalyDetectorError
# ---------------------------------------------------------------------------


def test_invalid_method_raises():
    df = _make_df(_normal_series())
    with pytest.raises(AnomalyDetectorError, match="method"):
        detect_anomalies(df, method="median")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 11. Parametrized method variants all flag the spike
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["zscore", "iqr", "both"])
def test_all_methods_flag_spike(method: str):
    df = _make_df(_normal_series())
    result = detect_anomalies(df, method=method, z_threshold=3.0, iqr_k=1.5)
    flagged = result[result["is_anomaly"]]
    assert 8000 in flagged["prescriptions_count"].values, (
        f"Method '{method}' failed to flag the 8 000 spike."
    )


# ---------------------------------------------------------------------------
# 12. Multi-drug grouping — each drug analysed independently
# ---------------------------------------------------------------------------


def test_multi_drug_grouping_independent():
    """DrugA has a spike; DrugB is flat.  Only DrugA rows should be flagged."""
    drug_a = _normal_series()  # has spike at 8 000
    drug_b = [500.0] * 12  # perfectly flat

    df = pd.DataFrame(
        {
            "drug_name": ["DrugA"] * 12 + ["DrugB"] * 12,
            "prescriptions_count": drug_a + drug_b,
        }
    )

    result = detect_anomalies(df, method="both", z_threshold=3.0, iqr_k=1.5)

    drug_b_flagged = result[result["drug_name"] == "DrugB"]["is_anomaly"].sum()
    assert drug_b_flagged == 0, "Flat DrugB should have no anomalies."

    drug_a_flagged = result[result["drug_name"] == "DrugA"]["is_anomaly"].sum()
    assert drug_a_flagged >= 1, "DrugA spike should be detected."


# ---------------------------------------------------------------------------
# 13. Rationale string populated for flagged rows only
# ---------------------------------------------------------------------------


def test_rationale_populated_for_flagged_only():
    df = _make_df(_normal_series())
    result = detect_anomalies(df)
    for _, row in result.iterrows():
        if row["is_anomaly"]:
            assert isinstance(row["anomaly_rationale"], str)
            assert len(row["anomaly_rationale"]) > 0
        else:
            assert row["anomaly_rationale"] is None


# ---------------------------------------------------------------------------
# 14. z_threshold=0 raises AnomalyDetectorError
# ---------------------------------------------------------------------------


def test_zero_z_threshold_raises():
    df = _make_df(_normal_series())
    with pytest.raises(AnomalyDetectorError, match="z_threshold"):
        detect_anomalies(df, z_threshold=0.0)


# ---------------------------------------------------------------------------
# 15. No-group mode (group_col disabled) treats entire frame as one series
# ---------------------------------------------------------------------------


def test_no_group_mode_single_series():
    """When group_col is empty string, the whole DataFrame is one series."""
    df = pd.DataFrame(
        {
            "drug_name": ["DrugA"] * 6 + ["DrugB"] * 6,
            "prescriptions_count": [1000] * 11 + [9000],  # one big spike
        }
    )
    result = detect_anomalies(df, group_col="", method="both", z_threshold=3.0)
    flagged = result[result["is_anomaly"]]
    assert 9000 in flagged["prescriptions_count"].values
