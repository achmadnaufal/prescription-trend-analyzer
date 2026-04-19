"""Tests for :mod:`src.mat`.

Covers Moving Annual Total (MAT), MAT growth (YoY change of MAT), and
MAT share (market-share equivalent on the MAT basis).

The tests lean heavily on closed-form inputs so the expected values are
obvious (e.g. a constant series of 100 yields MAT=1200 once the
12-month window fills).  Edge cases cover the requirements:

* single data point
* missing months / gaps in the series
* all-zero volumes
* non-datetime index / unsorted rows
* empty DataFrame
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.mat import (
    MATError,
    mat_growth,
    mat_share,
    moving_annual_total,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _constant_series(n: int, value: float = 100.0, product: str = "A") -> pd.DataFrame:
    """Return a monthly DataFrame with a constant ``rx_volume``."""
    return pd.DataFrame(
        {
            "month": pd.date_range("2023-01-01", periods=n, freq="MS"),
            "product": [product] * n,
            "rx_volume": [float(value)] * n,
        }
    )


def _two_product_df(n: int = 24) -> pd.DataFrame:
    """Return a two-product monthly DataFrame with distinct flat levels."""
    a = _constant_series(n, 100.0, "A")
    b = _constant_series(n, 200.0, "B")
    return pd.concat([a, b], ignore_index=True)


# ---------------------------------------------------------------------------
# moving_annual_total - happy path
# ---------------------------------------------------------------------------


def test_mat_constant_series_sums_to_window_times_value():
    df = _constant_series(18, value=100.0)
    out = moving_annual_total(
        df, value_col="rx_volume", group_col="product", date_col="month"
    )
    # First 11 rows: window not yet full -> NaN
    assert out["rx_volume_mat12"].iloc[:11].isna().all()
    # Rows 11..17: full 12-month window of 100 -> 1200
    assert out["rx_volume_mat12"].iloc[11:].tolist() == pytest.approx(
        [1200.0] * 7
    )


def test_mat_custom_window_and_output_col():
    df = _constant_series(6, value=50.0)
    out = moving_annual_total(
        df,
        value_col="rx_volume",
        group_col="product",
        date_col="month",
        window=3,
        output_col="mat3",
    )
    assert "mat3" in out.columns
    # First two rows NaN (min_periods defaults to window=3), then 150 thereafter.
    assert out["mat3"].iloc[:2].isna().all()
    assert out["mat3"].iloc[2:].tolist() == pytest.approx([150.0] * 4)


def test_mat_min_periods_allows_partial_window():
    df = _constant_series(5, value=10.0)
    out = moving_annual_total(
        df,
        value_col="rx_volume",
        group_col="product",
        date_col="month",
        window=12,
        min_periods=1,
    )
    # Cumulative partial sums: 10, 20, 30, 40, 50.
    assert list(out["rx_volume_mat12"]) == pytest.approx([10, 20, 30, 40, 50])


def test_mat_groups_are_independent():
    df = _two_product_df(n=14)
    out = moving_annual_total(
        df, value_col="rx_volume", group_col="product", date_col="month"
    )
    a = out[out["product"] == "A"].reset_index(drop=True)
    b = out[out["product"] == "B"].reset_index(drop=True)
    # Row index 11 is the first with a full 12-month window within the group.
    assert a.loc[11, "rx_volume_mat12"] == pytest.approx(1200.0)
    assert b.loc[11, "rx_volume_mat12"] == pytest.approx(2400.0)


def test_mat_without_group_col():
    df = _constant_series(12, value=100.0).drop(columns=["product"])
    out = moving_annual_total(df, value_col="rx_volume", date_col="month")
    # Row 11 (12th row) is the first full window.
    assert out["rx_volume_mat12"].iloc[11] == pytest.approx(1200.0)
    assert out["rx_volume_mat12"].iloc[:11].isna().all()


def test_mat_sorts_unsorted_input_by_date():
    df = _constant_series(12, value=100.0).sample(
        frac=1.0, random_state=7
    ).reset_index(drop=True)
    out = moving_annual_total(
        df, value_col="rx_volume", group_col="product", date_col="month"
    )
    # After sorting internally, the last chronological row has the full window.
    out_sorted = out.sort_values("month").reset_index(drop=True)
    assert out_sorted["rx_volume_mat12"].iloc[11] == pytest.approx(1200.0)


def test_mat_does_not_mutate_input():
    df = _constant_series(15)
    snapshot = df.copy()
    _ = moving_annual_total(
        df, value_col="rx_volume", group_col="product", date_col="month"
    )
    pd.testing.assert_frame_equal(df, snapshot)


# ---------------------------------------------------------------------------
# moving_annual_total - edge cases
# ---------------------------------------------------------------------------


def test_mat_single_data_point_returns_nan_by_default():
    df = _constant_series(1)
    out = moving_annual_total(
        df, value_col="rx_volume", group_col="product", date_col="month"
    )
    assert len(out) == 1
    assert np.isnan(out["rx_volume_mat12"].iloc[0])


def test_mat_single_data_point_with_min_periods_1():
    df = _constant_series(1, value=42.0)
    out = moving_annual_total(
        df,
        value_col="rx_volume",
        group_col="product",
        date_col="month",
        min_periods=1,
    )
    assert out["rx_volume_mat12"].iloc[0] == pytest.approx(42.0)


def test_mat_all_zero_volumes():
    df = _constant_series(14, value=0.0)
    out = moving_annual_total(
        df, value_col="rx_volume", group_col="product", date_col="month"
    )
    # Full windows exist but every sum is 0.0 (not NaN).
    filled = out["rx_volume_mat12"].iloc[11:]
    assert not filled.isna().any()
    assert filled.tolist() == pytest.approx([0.0] * len(filled))


def test_mat_handles_missing_month_gaps_as_positional_rolling():
    # A gap in the calendar (Feb missing) — we document that MAT is
    # *positional*, not *calendar-aware*, so 11 observations with a gap
    # will not yet trigger a full window at row index 10.
    dates = pd.to_datetime(
        [
            "2024-01-01", "2024-03-01", "2024-04-01", "2024-05-01",
            "2024-06-01", "2024-07-01", "2024-08-01", "2024-09-01",
            "2024-10-01", "2024-11-01", "2024-12-01",  # 11 rows
        ]
    )
    df = pd.DataFrame(
        {"month": dates, "product": "A", "rx_volume": [100.0] * 11}
    )
    out = moving_annual_total(
        df, value_col="rx_volume", group_col="product", date_col="month"
    )
    # Only 11 rows: the 12-window never fills -> all NaN.
    assert out["rx_volume_mat12"].isna().all()


def test_mat_non_datetime_date_column_is_accepted_if_sortable():
    # String month labels sort lexicographically the same as chronologically
    # for ISO-formatted YYYY-MM values, so the rolling should still work.
    df = pd.DataFrame(
        {
            "month": [f"2024-{m:02d}" for m in range(1, 13)],
            "product": ["A"] * 12,
            "rx_volume": [10.0] * 12,
        }
    )
    out = moving_annual_total(
        df, value_col="rx_volume", group_col="product", date_col="month"
    )
    assert out["rx_volume_mat12"].iloc[11] == pytest.approx(120.0)


def test_mat_empty_dataframe_raises():
    df = pd.DataFrame({"rx_volume": [], "product": [], "month": []})
    with pytest.raises(MATError, match="empty"):
        moving_annual_total(df, value_col="rx_volume")


def test_mat_rejects_missing_value_col():
    df = _constant_series(12)
    with pytest.raises(MATError, match="value_col"):
        moving_annual_total(df, value_col="missing")


def test_mat_rejects_missing_group_col():
    df = _constant_series(12)
    with pytest.raises(MATError, match="group_col"):
        moving_annual_total(df, value_col="rx_volume", group_col="nope")


def test_mat_rejects_missing_date_col():
    df = _constant_series(12)
    with pytest.raises(MATError, match="date_col"):
        moving_annual_total(df, value_col="rx_volume", date_col="nope")


def test_mat_rejects_bad_window():
    df = _constant_series(12)
    with pytest.raises(MATError, match="window"):
        moving_annual_total(df, value_col="rx_volume", window=0)


def test_mat_rejects_bad_min_periods():
    df = _constant_series(12)
    with pytest.raises(MATError, match="min_periods"):
        moving_annual_total(
            df, value_col="rx_volume", window=12, min_periods=13
        )


# ---------------------------------------------------------------------------
# mat_growth
# ---------------------------------------------------------------------------


def test_mat_growth_zero_when_volumes_are_flat():
    # 24 months of constant 100 -> MAT is 1200 from month 12 onwards,
    # YoY growth on MAT from month 24 onward is 0.0%.
    df = _constant_series(24, value=100.0)
    out = mat_growth(
        df, value_col="rx_volume", group_col="product", date_col="month"
    )
    # Row index 23 is the first with a valid MAT YoY lag (needs 24 rows).
    assert out["rx_volume_mat12_growth_pct"].iloc[23] == pytest.approx(0.0)


def test_mat_growth_positive_when_volumes_double():
    # Year-1 volume = 100; Year-2 volume = 200 -> MAT grows 1200 -> 2400,
    # YoY MAT growth at the 24th month = +100%.
    volumes = [100.0] * 12 + [200.0] * 12
    df = pd.DataFrame(
        {
            "month": pd.date_range("2023-01-01", periods=24, freq="MS"),
            "product": "A",
            "rx_volume": volumes,
        }
    )
    out = mat_growth(
        df, value_col="rx_volume", group_col="product", date_col="month"
    )
    assert out["rx_volume_mat12_growth_pct"].iloc[23] == pytest.approx(100.0)


def test_mat_growth_nan_for_incomplete_lag_rows():
    df = _constant_series(20, value=100.0)
    out = mat_growth(
        df, value_col="rx_volume", group_col="product", date_col="month"
    )
    # No row has a full 24-month history, so every growth value is NaN.
    assert out["rx_volume_mat12_growth_pct"].isna().all()


def test_mat_growth_zero_previous_yields_nan():
    # First 12 months all-zero -> MAT[11]=0 -> MAT growth at row 23 is
    # undefined (division by zero) -> NaN.
    volumes = [0.0] * 12 + [100.0] * 12
    df = pd.DataFrame(
        {
            "month": pd.date_range("2023-01-01", periods=24, freq="MS"),
            "product": "A",
            "rx_volume": volumes,
        }
    )
    out = mat_growth(
        df, value_col="rx_volume", group_col="product", date_col="month"
    )
    assert np.isnan(out["rx_volume_mat12_growth_pct"].iloc[23])


def test_mat_growth_does_not_mutate_input():
    df = _constant_series(24)
    snapshot = df.copy()
    _ = mat_growth(
        df, value_col="rx_volume", group_col="product", date_col="month"
    )
    pd.testing.assert_frame_equal(df, snapshot)


# ---------------------------------------------------------------------------
# mat_share
# ---------------------------------------------------------------------------


def test_mat_share_sums_to_100_per_date():
    df = _two_product_df(n=14)
    out = mat_share(
        df, value_col="rx_volume", group_col="product", date_col="month"
    )
    # From row index 11 onwards, full windows exist for both products.
    full_window_dates = out.dropna(subset=["rx_volume_mat12_share_pct"])[
        "month"
    ].unique()
    for d in full_window_dates:
        total = out.loc[
            out["month"] == d, "rx_volume_mat12_share_pct"
        ].sum()
        assert total == pytest.approx(100.0)


def test_mat_share_proportional_to_group_total():
    # Product A at 100/month, B at 300/month -> expected shares 25% / 75%.
    a = _constant_series(14, 100.0, "A")
    b = _constant_series(14, 300.0, "B")
    df = pd.concat([a, b], ignore_index=True)
    out = mat_share(
        df, value_col="rx_volume", group_col="product", date_col="month"
    )
    last_month = out["month"].max()
    a_share = out.loc[
        (out["month"] == last_month) & (out["product"] == "A"),
        "rx_volume_mat12_share_pct",
    ].iloc[0]
    b_share = out.loc[
        (out["month"] == last_month) & (out["product"] == "B"),
        "rx_volume_mat12_share_pct",
    ].iloc[0]
    assert a_share == pytest.approx(25.0)
    assert b_share == pytest.approx(75.0)


def test_mat_share_requires_group_and_date():
    df = _constant_series(14)
    with pytest.raises(MATError, match="group_col"):
        mat_share(df, value_col="rx_volume", group_col=None, date_col="month")
    with pytest.raises(MATError, match="date_col"):
        mat_share(df, value_col="rx_volume", group_col="product", date_col=None)


def test_mat_share_all_zero_volumes_yield_nan():
    a = _constant_series(14, 0.0, "A")
    b = _constant_series(14, 0.0, "B")
    df = pd.concat([a, b], ignore_index=True)
    out = mat_share(
        df, value_col="rx_volume", group_col="product", date_col="month"
    )
    # Once windows fill, MAT=0 on both products, totals=0 -> share=NaN.
    assert out["rx_volume_mat12_share_pct"].iloc[-1] is not None
    assert np.isnan(out["rx_volume_mat12_share_pct"].iloc[-1])


def test_mat_share_does_not_mutate_input():
    df = _two_product_df(n=14)
    snapshot = df.copy()
    _ = mat_share(
        df, value_col="rx_volume", group_col="product", date_col="month"
    )
    pd.testing.assert_frame_equal(df, snapshot)


# ---------------------------------------------------------------------------
# Public API re-export
# ---------------------------------------------------------------------------


def test_public_api_reexports():
    import src

    for name in ("moving_annual_total", "mat_growth", "mat_share", "MATError"):
        assert hasattr(src, name), f"{name} not re-exported from src/__init__.py"
