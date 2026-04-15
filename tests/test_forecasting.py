"""
Unit tests for forecasting functionality in RxTrendAnalyzer.

Covers:
- Linear forecast helper (_linear_forecast)
- RxTrendAnalyzer.forecast() happy path and edge cases
- Month arithmetic helper (_advance_months)

Run with:
    pytest tests/test_forecasting.py -v
"""

import math
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.main import RxTrendAnalyzer, _linear_forecast, _advance_months


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def analyzer() -> RxTrendAnalyzer:
    """Return a default RxTrendAnalyzer instance."""
    return RxTrendAnalyzer()


@pytest.fixture()
def multi_month_df() -> pd.DataFrame:
    """Two drugs each with 6 monthly observations — enough for forecasting."""
    return pd.DataFrame(
        {
            "month": list(range(1, 7)) * 2,
            "year": [2024] * 12,
            "drug_name": ["DrugA"] * 6 + ["DrugB"] * 6,
            "prescriptions_count": [
                1000, 1050, 1100, 1150, 1200, 1250,  # DrugA: linear +50/month
                500, 490, 480, 470, 460, 450,         # DrugB: linear -10/month
            ],
        }
    )


# ---------------------------------------------------------------------------
# 1. _linear_forecast helper
# ---------------------------------------------------------------------------


class TestLinearForecastHelper:
    def test_perfect_linear_sequence(self) -> None:
        """Forecast of a perfect linear trend must extrapolate exactly."""
        values = np.array([100.0, 200.0, 300.0, 400.0])
        projected = _linear_forecast(values, n_periods=2)
        assert math.isclose(projected[0], 500.0, rel_tol=1e-6)
        assert math.isclose(projected[1], 600.0, rel_tol=1e-6)

    def test_flat_series_projects_same_value(self) -> None:
        """Flat historical values must forecast the same constant value."""
        values = np.array([300.0, 300.0, 300.0])
        projected = _linear_forecast(values, n_periods=3)
        for p in projected:
            assert math.isclose(p, 300.0, abs_tol=1e-6)

    def test_n_periods_length(self) -> None:
        """Returned array must have exactly n_periods elements."""
        values = np.array([10.0, 20.0, 30.0])
        for n in (1, 5, 10):
            assert len(_linear_forecast(values, n_periods=n)) == n

    def test_raises_for_single_observation(self) -> None:
        """Fewer than 2 observations must raise ValueError."""
        with pytest.raises(ValueError, match="At least 2"):
            _linear_forecast(np.array([100.0]), n_periods=1)

    def test_raises_for_zero_periods(self) -> None:
        """n_periods=0 must raise ValueError."""
        with pytest.raises(ValueError, match="n_periods"):
            _linear_forecast(np.array([100.0, 200.0]), n_periods=0)

    def test_negative_trend_decreasing_forecast(self) -> None:
        """Declining historical values must yield declining forecasts."""
        values = np.array([500.0, 400.0, 300.0])
        projected = _linear_forecast(values, n_periods=2)
        assert projected[0] < 300.0
        assert projected[1] < projected[0]


# ---------------------------------------------------------------------------
# 2. _advance_months helper
# ---------------------------------------------------------------------------


class TestAdvanceMonths:
    def test_within_same_year(self) -> None:
        """Advancing 2 months from January should stay within the same year."""
        result = _advance_months(2024, 1, 2)
        assert result == [(2024, 2), (2024, 3)]

    def test_crosses_year_boundary(self) -> None:
        """Advancing past December must increment the year correctly."""
        result = _advance_months(2024, 11, 3)
        assert result == [(2024, 12), (2025, 1), (2025, 2)]

    def test_december_wraps_to_january(self) -> None:
        """Advancing 1 month from December must produce January of next year."""
        result = _advance_months(2024, 12, 1)
        assert result == [(2025, 1)]

    def test_multiple_year_crossings(self) -> None:
        """14 months from November 2024 should cross into 2026."""
        # Nov 2024 + 1 = Dec 2024, + 2 = Jan 2025, ..., + 14 = Jan 2026
        result = _advance_months(2024, 11, 14)
        assert result[0] == (2024, 12)
        assert result[1] == (2025, 1)
        assert result[-1] == (2026, 1)
        assert len(result) == 14

    def test_zero_periods_returns_empty_list(self) -> None:
        """Advancing 0 periods must return an empty list."""
        assert _advance_months(2024, 6, 0) == []


# ---------------------------------------------------------------------------
# 3. RxTrendAnalyzer.forecast()
# ---------------------------------------------------------------------------


class TestForecast:
    def test_forecast_returns_dataframe(self, analyzer: RxTrendAnalyzer, multi_month_df: pd.DataFrame) -> None:
        """forecast() must return a non-empty DataFrame."""
        result = analyzer.forecast(multi_month_df, n_periods=3)
        assert isinstance(result, pd.DataFrame)
        assert not result.empty

    def test_forecast_row_count(self, analyzer: RxTrendAnalyzer, multi_month_df: pd.DataFrame) -> None:
        """Result must have n_periods rows per drug."""
        n_periods = 3
        n_drugs = multi_month_df["drug_name"].nunique()
        result = analyzer.forecast(multi_month_df, n_periods=n_periods)
        assert len(result) == n_periods * n_drugs

    def test_forecast_is_forecast_column_all_true(
        self, analyzer: RxTrendAnalyzer, multi_month_df: pd.DataFrame
    ) -> None:
        """All rows in the forecast DataFrame must have is_forecast=True."""
        result = analyzer.forecast(multi_month_df, n_periods=2)
        assert result["is_forecast"].all()

    def test_forecast_does_not_mutate_input(
        self, analyzer: RxTrendAnalyzer, multi_month_df: pd.DataFrame
    ) -> None:
        """forecast() must not add columns to the caller's DataFrame."""
        cols_before = set(multi_month_df.columns)
        _ = analyzer.forecast(multi_month_df, n_periods=2)
        assert set(multi_month_df.columns) == cols_before

    def test_forecast_increasing_drug_trend(
        self, analyzer: RxTrendAnalyzer, multi_month_df: pd.DataFrame
    ) -> None:
        """For DrugA (rising trend) each projected value must exceed the last actual."""
        last_actual = multi_month_df.loc[
            multi_month_df["drug_name"] == "DrugA", "prescriptions_count"
        ].max()
        result = analyzer.forecast(multi_month_df, n_periods=3)
        drug_a_forecast = result[result["drug_name"] == "DrugA"]["prescriptions_count"]
        assert (drug_a_forecast > last_actual).all()

    def test_forecast_decreasing_drug_trend(
        self, analyzer: RxTrendAnalyzer, multi_month_df: pd.DataFrame
    ) -> None:
        """For DrugB (declining trend) each projected value must be below the last actual."""
        last_actual = multi_month_df.loc[
            multi_month_df["drug_name"] == "DrugB", "prescriptions_count"
        ].min()
        result = analyzer.forecast(multi_month_df, n_periods=3)
        drug_b_forecast = result[result["drug_name"] == "DrugB"]["prescriptions_count"]
        assert (drug_b_forecast < last_actual).all()

    def test_forecast_raises_for_missing_value_col(
        self, analyzer: RxTrendAnalyzer, multi_month_df: pd.DataFrame
    ) -> None:
        """forecast() must raise ValueError when value_col is absent."""
        with pytest.raises(ValueError, match="not found"):
            analyzer.forecast(multi_month_df, value_col="nonexistent")

    def test_forecast_raises_for_zero_periods(
        self, analyzer: RxTrendAnalyzer, multi_month_df: pd.DataFrame
    ) -> None:
        """n_periods=0 must raise ValueError."""
        with pytest.raises(ValueError, match="n_periods"):
            analyzer.forecast(multi_month_df, n_periods=0)

    def test_forecast_raises_when_all_drugs_insufficient(
        self, analyzer: RxTrendAnalyzer
    ) -> None:
        """If every drug has only one row, forecast() must raise ValueError."""
        df = pd.DataFrame(
            {
                "month": [1],
                "year": [2024],
                "drug_name": ["OnlyOne"],
                "prescriptions_count": [1000],
            }
        )
        with pytest.raises(ValueError, match="sufficient historical data"):
            analyzer.forecast(df, n_periods=3)

    def test_forecast_skips_drug_with_single_observation(
        self, analyzer: RxTrendAnalyzer
    ) -> None:
        """Drug with only 1 row is skipped; drug with 2+ rows is still forecast."""
        df = pd.DataFrame(
            {
                "month":  [1,    1, 2],
                "year":   [2024, 2024, 2024],
                "drug_name": ["Sparse", "Enough", "Enough"],
                "prescriptions_count": [500, 1000, 1100],
            }
        )
        result = analyzer.forecast(df, n_periods=2)
        # Only 'Enough' should appear in forecasts
        assert set(result["drug_name"].unique()) == {"Enough"}

    def test_forecast_future_periods_are_after_last_actual(
        self, analyzer: RxTrendAnalyzer, multi_month_df: pd.DataFrame
    ) -> None:
        """All forecast (year, month) pairs must come after the last observed period."""
        result = analyzer.forecast(multi_month_df, n_periods=3)
        last_year = int(multi_month_df["year"].max())
        last_month = int(
            multi_month_df.loc[multi_month_df["year"] == last_year, "month"].max()
        )
        for _, row in result.iterrows():
            fy, fm = int(row["year"]), int(row["month"])
            assert (fy, fm) > (last_year, last_month), (
                f"Forecast period {fy}-{fm:02d} is not after last actual {last_year}-{last_month:02d}"
            )

    def test_forecast_columns_present(
        self, analyzer: RxTrendAnalyzer, multi_month_df: pd.DataFrame
    ) -> None:
        """Result must contain drug_name, year, month, prescriptions_count, is_forecast."""
        result = analyzer.forecast(multi_month_df, n_periods=1)
        for col in ("drug_name", "year", "month", "prescriptions_count", "is_forecast"):
            assert col in result.columns, f"Expected column '{col}' missing from forecast result."
