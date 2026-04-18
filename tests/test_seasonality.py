"""Tests for :mod:`src.seasonality`.

Covers seasonal decomposition wrapper and period-over-period growth
helpers.  The tests are intentionally structural: they avoid asserting
on exact statsmodels numbers and instead verify shape, column names,
edge-case handling, immutability, and group isolation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.seasonality import (
    SeasonalityError,
    period_over_period_growth,
    seasonal_decompose_series,
)


# ---------------------------------------------------------------------------
# seasonal_decompose_series
# ---------------------------------------------------------------------------


def _monthly_series(n: int, base: float = 1000.0) -> pd.Series:
    """Build a deterministic monthly series with a mild seasonal pattern."""
    idx = pd.date_range("2022-01-01", periods=n, freq="MS")
    trend = np.linspace(0, 100, n)
    season = 50 * np.sin(2 * np.pi * np.arange(n) / 12)
    return pd.Series(base + trend + season, index=idx, name="rx")


def test_seasonal_decompose_returns_expected_columns():
    s = _monthly_series(36)
    out = seasonal_decompose_series(s, period=12)
    assert list(out.columns) == ["observed", "trend", "seasonal", "resid"]
    assert len(out) == len(s)
    assert (out.index == s.index).all()


def test_seasonal_decompose_does_not_mutate_input():
    s = _monthly_series(30)
    snapshot = s.copy()
    _ = seasonal_decompose_series(s, period=12)
    pd.testing.assert_series_equal(s, snapshot)


def test_seasonal_decompose_fills_isolated_nans():
    s = _monthly_series(30)
    s.iloc[5] = np.nan
    out = seasonal_decompose_series(s, period=12)
    # The observed column should be fully populated after ffill/bfill.
    assert not out["observed"].isna().any()


def test_seasonal_decompose_rejects_short_series():
    s = _monthly_series(10)
    with pytest.raises(SeasonalityError, match="at least"):
        seasonal_decompose_series(s, period=12)


def test_seasonal_decompose_rejects_bad_period():
    s = _monthly_series(24)
    with pytest.raises(SeasonalityError, match="period"):
        seasonal_decompose_series(s, period=1)


def test_seasonal_decompose_rejects_bad_model():
    s = _monthly_series(24)
    with pytest.raises(SeasonalityError, match="model"):
        seasonal_decompose_series(s, period=12, model="weird")


def test_seasonal_decompose_multiplicative_rejects_non_positive():
    s = _monthly_series(24)
    s.iloc[0] = 0.0
    with pytest.raises(SeasonalityError, match="positive"):
        seasonal_decompose_series(s, period=12, model="multiplicative")


def test_seasonal_decompose_rejects_non_series():
    with pytest.raises(SeasonalityError, match="pandas Series"):
        seasonal_decompose_series([1, 2, 3], period=2)


# ---------------------------------------------------------------------------
# period_over_period_growth
# ---------------------------------------------------------------------------


def _two_drug_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2024-01-01", "2024-02-01", "2024-03-01",
                    "2024-01-01", "2024-02-01", "2024-03-01",
                ]
            ),
            "drug": ["A", "A", "A", "B", "B", "B"],
            "rx": [100.0, 110.0, 121.0, 200.0, 180.0, 90.0],
        }
    )


def test_pop_pct_monthly():
    df = _two_drug_df()
    out = period_over_period_growth(
        df, value_col="rx", group_col="drug", date_col="date", lag=1
    )
    a = out[out["drug"] == "A"].reset_index(drop=True)
    assert np.isnan(a.loc[0, "rx_pct_lag1"])
    assert a.loc[1, "rx_pct_lag1"] == pytest.approx(10.0)
    assert a.loc[2, "rx_pct_lag1"] == pytest.approx(10.0)


def test_pop_abs_output():
    df = _two_drug_df()
    out = period_over_period_growth(
        df,
        value_col="rx",
        group_col="drug",
        date_col="date",
        lag=1,
        output="abs",
    )
    b = out[out["drug"] == "B"].reset_index(drop=True)
    assert b.loc[1, "rx_abs_lag1"] == pytest.approx(-20.0)
    assert b.loc[2, "rx_abs_lag1"] == pytest.approx(-90.0)


def test_pop_ratio_output_and_custom_col():
    df = _two_drug_df()
    out = period_over_period_growth(
        df,
        value_col="rx",
        group_col="drug",
        date_col="date",
        lag=1,
        output="ratio",
        output_col="growth",
    )
    assert "growth" in out.columns
    a = out[out["drug"] == "A"].reset_index(drop=True)
    assert a.loc[1, "growth"] == pytest.approx(1.1)


def test_pop_handles_zero_previous_as_nan():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-02-01"]),
            "drug": ["A", "A"],
            "rx": [0.0, 10.0],
        }
    )
    out = period_over_period_growth(
        df, value_col="rx", group_col="drug", date_col="date", lag=1
    )
    assert np.isnan(out["rx_pct_lag1"].iloc[1])


def test_pop_groups_are_independent():
    df = _two_drug_df()
    out = period_over_period_growth(
        df, value_col="rx", group_col="drug", date_col="date", lag=1
    )
    # First row of each group must be NaN (no prior value within group).
    ordered = out.sort_values(["drug", "date"]).reset_index(drop=True)
    firsts = ordered.groupby("drug", sort=False).nth(0)["rx_pct_lag1"]
    assert firsts.isna().all()


def test_pop_alias_yoy_requires_12_rows_per_group():
    idx = pd.date_range("2023-01-01", periods=14, freq="MS")
    df = pd.DataFrame(
        {"date": idx, "drug": "A", "rx": np.arange(1, 15, dtype=float)}
    )
    out = period_over_period_growth(
        df, value_col="rx", group_col="drug", date_col="date", lag="yoy"
    )
    # Row 12 compares value 13 vs value 1 -> 1200% growth.
    assert out["rx_pct_lag12"].iloc[12] == pytest.approx(1200.0)
    # Rows 0..11 should all be NaN (no yoy history yet).
    assert out["rx_pct_lag12"].iloc[:12].isna().all()


def test_pop_does_not_mutate_input():
    df = _two_drug_df()
    snapshot = df.copy()
    _ = period_over_period_growth(
        df, value_col="rx", group_col="drug", date_col="date"
    )
    pd.testing.assert_frame_equal(df, snapshot)


def test_pop_without_group_col():
    df = pd.DataFrame({"rx": [100.0, 110.0, 121.0]})
    out = period_over_period_growth(df, value_col="rx", lag=1)
    assert out["rx_pct_lag1"].iloc[1] == pytest.approx(10.0)
    assert out["rx_pct_lag1"].iloc[2] == pytest.approx(10.0)


def test_pop_rejects_unknown_alias():
    df = _two_drug_df()
    with pytest.raises(SeasonalityError, match="alias"):
        period_over_period_growth(
            df, value_col="rx", group_col="drug", lag="daily"
        )


def test_pop_rejects_missing_columns():
    df = _two_drug_df()
    with pytest.raises(SeasonalityError, match="value_col"):
        period_over_period_growth(df, value_col="missing")
    with pytest.raises(SeasonalityError, match="group_col"):
        period_over_period_growth(df, value_col="rx", group_col="nope")


def test_pop_rejects_empty_df():
    df = pd.DataFrame({"rx": []})
    with pytest.raises(SeasonalityError, match="empty"):
        period_over_period_growth(df, value_col="rx")


def test_pop_rejects_bad_output():
    df = _two_drug_df()
    with pytest.raises(SeasonalityError, match="output"):
        period_over_period_growth(df, value_col="rx", output="wat")


def test_pop_sorts_non_monotonic_dates_within_group():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2024-03-01", "2024-01-01", "2024-02-01"]
            ),
            "drug": ["A", "A", "A"],
            "rx": [121.0, 100.0, 110.0],
        }
    )
    out = period_over_period_growth(
        df, value_col="rx", group_col="drug", date_col="date", lag=1
    )
    out_sorted = out.sort_values("date").reset_index(drop=True)
    assert np.isnan(out_sorted.loc[0, "rx_pct_lag1"])
    assert out_sorted.loc[1, "rx_pct_lag1"] == pytest.approx(10.0)
    assert out_sorted.loc[2, "rx_pct_lag1"] == pytest.approx(10.0)
