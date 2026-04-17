"""
Tests for src/changepoint_detector.py.

Covers:
* Happy-path detection of an obvious trend break.
* Edge cases — single point, all zeros, NaN gaps, non-monotonic dates,
  insufficient data, perfectly linear series.
* Multi-group DataFrame helper, including independent per-drug analysis.
* Input validation guards.
* Immutability of caller's DataFrame.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.changepoint_detector import (
    ChangePointDetectorError,
    ChangePointResult,
    detect_change_point,
    detect_change_points,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _series_with_break(
    n_before: int = 8,
    n_after: int = 8,
    slope_before: float = 1.0,
    slope_after: float = 5.0,
    intercept_before: float = 100.0,
) -> np.ndarray:
    """Build a deterministic two-segment series with a clear slope shift."""
    pre = intercept_before + slope_before * np.arange(n_before)
    # Continuity at the break point
    intercept_after = pre[-1] - slope_after * (n_before - 1)
    post = intercept_after + slope_after * np.arange(n_before, n_before + n_after)
    return np.concatenate([pre, post])


def _frame(
    drug: str,
    values: list[float],
    start: str = "2024-01-01",
) -> pd.DataFrame:
    dates = pd.date_range(start=start, periods=len(values), freq="MS")
    return pd.DataFrame(
        {
            "drug_name": [drug] * len(values),
            "date": dates,
            "prescriptions_count": values,
        }
    )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_detect_change_point_finds_known_break() -> None:
    series = _series_with_break(n_before=8, n_after=8)
    result = detect_change_point(series, min_segment=3)

    assert isinstance(result, ChangePointResult)
    # The break is at index 8; allow ±1 tolerance for OLS rounding.
    assert result.index is not None and abs(result.index - 8) <= 1
    assert result.slope_before is not None
    assert result.slope_after is not None
    assert result.slope_after > result.slope_before
    assert result.is_significant is True
    assert result.improvement_ratio is not None
    assert 0.0 < result.improvement_ratio <= 1.0


def test_detect_change_point_returns_dict_view() -> None:
    series = _series_with_break()
    result = detect_change_point(series, min_segment=3)
    payload = result.to_dict()
    assert set(payload.keys()) == {
        "index",
        "improvement_ratio",
        "slope_before",
        "slope_after",
        "intercept_before",
        "intercept_after",
        "is_significant",
    }


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_single_point_returns_null_result() -> None:
    result = detect_change_point([42.0], min_segment=3)
    assert result.index is None
    assert result.improvement_ratio is None
    assert result.is_significant is False


def test_too_short_for_split_returns_null_result() -> None:
    # min_segment=3 means we need >=6 points; supply 5.
    result = detect_change_point([1.0, 2.0, 3.0, 4.0, 5.0], min_segment=3)
    assert result.index is None
    assert result.is_significant is False


def test_all_zeros_returns_non_significant() -> None:
    series = [0.0] * 12
    result = detect_change_point(series, min_segment=3)
    # A flat zero line has SSR_single == 0 so improvement is defined as 0.
    assert result.is_significant is False
    assert result.improvement_ratio == 0.0
    # Slopes on each segment must be (close to) zero.
    assert result.slope_before == pytest.approx(0.0, abs=1e-9)
    assert result.slope_after == pytest.approx(0.0, abs=1e-9)


def test_perfectly_linear_series_is_not_significant() -> None:
    # SSR_single == 0, so no split can improve on it.
    series = list(np.arange(12, dtype=float) * 3.0 + 5.0)
    result = detect_change_point(series, min_segment=3)
    assert result.is_significant is False
    assert result.improvement_ratio == 0.0


def test_nan_in_series_raises() -> None:
    series = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0])
    with pytest.raises(ChangePointDetectorError, match="NaN"):
        detect_change_point(series, min_segment=3)


def test_empty_series_raises() -> None:
    with pytest.raises(ChangePointDetectorError, match="empty"):
        detect_change_point([], min_segment=3)


def test_invalid_min_segment_raises() -> None:
    with pytest.raises(ChangePointDetectorError, match="min_segment"):
        detect_change_point([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], min_segment=1)


@pytest.mark.parametrize("bad", [-0.1, 1.0, 2.5])
def test_invalid_min_improvement_raises(bad: float) -> None:
    with pytest.raises(ChangePointDetectorError, match="min_improvement"):
        detect_change_point([1.0] * 10, min_improvement=bad)


# ---------------------------------------------------------------------------
# DataFrame helper — multi-group
# ---------------------------------------------------------------------------


def test_detect_change_points_multi_drug_independent() -> None:
    drug_a = _series_with_break(n_before=6, n_after=6, slope_after=10.0).tolist()
    drug_b = list(np.arange(12, dtype=float) * 2.0 + 50.0)  # perfectly linear
    df = pd.concat(
        [_frame("DrugA", drug_a), _frame("DrugB", drug_b, start="2024-01-01")],
        ignore_index=True,
    )

    report = detect_change_points(df, min_segment=3, date_col="date")

    assert set(report["group"]) == {"DrugA", "DrugB"}
    row_a = report.loc[report["group"] == "DrugA"].iloc[0]
    row_b = report.loc[report["group"] == "DrugB"].iloc[0]

    assert bool(row_a["is_significant"]) is True
    assert row_a["index"] is not None
    assert bool(row_b["is_significant"]) is False


def test_detect_change_points_handles_non_monotonic_dates() -> None:
    series = _series_with_break(n_before=6, n_after=6).tolist()
    df = _frame("DrugA", series)
    # Shuffle the rows; helper must restore order via date_col before scanning.
    shuffled = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    report = detect_change_points(shuffled, min_segment=3, date_col="date")
    row = report.iloc[0]
    assert bool(row["is_significant"]) is True
    assert row["index"] is not None and abs(int(row["index"]) - 6) <= 1


def test_detect_change_points_does_not_mutate_input() -> None:
    series = _series_with_break(n_before=6, n_after=6).tolist()
    df = _frame("DrugA", series)
    snapshot = df.copy(deep=True)
    _ = detect_change_points(df, min_segment=3, date_col="date")
    pd.testing.assert_frame_equal(df, snapshot)


def test_detect_change_points_drops_nan_rows_per_group() -> None:
    # NaN gaps within a series are dropped before detection so callers with
    # sparse data still get a usable answer.
    series = _series_with_break(n_before=6, n_after=6).tolist()
    series[4] = float("nan")  # introduce a single NaN gap
    df = _frame("DrugA", series)
    report = detect_change_points(df, min_segment=3, date_col="date")
    row = report.iloc[0]
    # 11 finite points remain, which still allows a valid split.
    assert row["index"] is not None


def test_detect_change_points_empty_df_raises() -> None:
    with pytest.raises(ChangePointDetectorError, match="empty"):
        detect_change_points(pd.DataFrame())


def test_detect_change_points_missing_value_col_raises() -> None:
    df = pd.DataFrame({"drug_name": ["A"], "other": [1]})
    with pytest.raises(ChangePointDetectorError, match="not found"):
        detect_change_points(df, value_col="prescriptions_count")


def test_detect_change_points_missing_date_col_raises() -> None:
    df = _frame("DrugA", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    with pytest.raises(ChangePointDetectorError, match="not found"):
        detect_change_points(df, date_col="ts_missing")


def test_detect_change_points_single_series_no_grouping() -> None:
    series = _series_with_break(n_before=5, n_after=5).tolist()
    df = pd.DataFrame({"prescriptions_count": series})
    report = detect_change_points(df, group_col="", min_segment=3)
    assert len(report) == 1
    assert report.iloc[0]["group"] == "_all_"
