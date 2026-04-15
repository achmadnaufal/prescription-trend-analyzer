"""
Unit tests for visualization-preparation helpers in RxTrendAnalyzer.

Covers:
- prepare_trend_chart_data()
- prepare_market_share_chart_data()
- summary_by_drug()
- filter_by_date_range()

Run with:
    pytest tests/test_visualization.py -v
"""

import math
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.main import RxTrendAnalyzer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def analyzer() -> RxTrendAnalyzer:
    """Return a default RxTrendAnalyzer instance."""
    return RxTrendAnalyzer()


@pytest.fixture()
def chart_df() -> pd.DataFrame:
    """Multi-drug, multi-month DataFrame suitable for chart-data tests."""
    return pd.DataFrame(
        {
            "month": [1, 2, 3, 1, 2, 3],
            "year":  [2024, 2024, 2024, 2024, 2024, 2024],
            "drug_name": ["Alpha", "Alpha", "Alpha", "Beta", "Beta", "Beta"],
            "prescriptions_count": [1000, 1100, 1200, 400, 420, 440],
            "region": ["North"] * 6,
        }
    )


@pytest.fixture()
def multi_year_df() -> pd.DataFrame:
    """DataFrame spanning two years for date-range filtering tests."""
    return pd.DataFrame(
        {
            "month": [10, 11, 12, 1, 2, 3] * 2,
            "year":  [2023, 2023, 2023, 2024, 2024, 2024] * 2,
            "drug_name": ["DrugX"] * 6 + ["DrugY"] * 6,
            "prescriptions_count": list(range(100, 106)) + list(range(200, 206)),
        }
    )


# ---------------------------------------------------------------------------
# 1. prepare_trend_chart_data
# ---------------------------------------------------------------------------


class TestPrepareTrendChartData:
    def test_returns_expected_keys(self, analyzer: RxTrendAnalyzer, chart_df: pd.DataFrame) -> None:
        """Result must contain 'labels', 'series', and 'value_col' keys."""
        result = analyzer.prepare_trend_chart_data(chart_df)
        assert set(result.keys()) >= {"labels", "series", "value_col"}

    def test_labels_are_sorted_chronologically(
        self, analyzer: RxTrendAnalyzer, chart_df: pd.DataFrame
    ) -> None:
        """Labels must be in ascending chronological order."""
        result = analyzer.prepare_trend_chart_data(chart_df)
        assert result["labels"] == sorted(result["labels"])

    def test_label_format_is_yyyy_mm(self, analyzer: RxTrendAnalyzer, chart_df: pd.DataFrame) -> None:
        """Every label must match the 'YYYY-MM' format."""
        result = analyzer.prepare_trend_chart_data(chart_df)
        for label in result["labels"]:
            parts = label.split("-")
            assert len(parts) == 2, f"Label '{label}' does not match YYYY-MM format."
            assert len(parts[0]) == 4
            assert len(parts[1]) == 2

    def test_series_length_matches_drug_count(
        self, analyzer: RxTrendAnalyzer, chart_df: pd.DataFrame
    ) -> None:
        """Number of series must equal number of unique drugs."""
        result = analyzer.prepare_trend_chart_data(chart_df)
        assert len(result["series"]) == chart_df["drug_name"].nunique()

    def test_series_data_length_matches_labels(
        self, analyzer: RxTrendAnalyzer, chart_df: pd.DataFrame
    ) -> None:
        """Each series data array must have the same length as labels."""
        result = analyzer.prepare_trend_chart_data(chart_df)
        n_labels = len(result["labels"])
        for s in result["series"]:
            assert len(s["data"]) == n_labels, (
                f"Series '{s['name']}' has {len(s['data'])} points but "
                f"there are {n_labels} labels."
            )

    def test_filter_by_drug_name(self, analyzer: RxTrendAnalyzer, chart_df: pd.DataFrame) -> None:
        """Passing drug_name must restrict result to a single series."""
        result = analyzer.prepare_trend_chart_data(chart_df, drug_name="Alpha")
        assert len(result["series"]) == 1
        assert result["series"][0]["name"] == "Alpha"

    def test_filter_by_drug_name_case_insensitive(
        self, analyzer: RxTrendAnalyzer, chart_df: pd.DataFrame
    ) -> None:
        """Drug name filter must be case-insensitive."""
        result_lower = analyzer.prepare_trend_chart_data(chart_df, drug_name="alpha")
        result_upper = analyzer.prepare_trend_chart_data(chart_df, drug_name="ALPHA")
        assert result_lower["series"][0]["name"] == result_upper["series"][0]["name"]

    def test_values_are_floats_or_none(self, analyzer: RxTrendAnalyzer, chart_df: pd.DataFrame) -> None:
        """All data points must be float or None (never NaN)."""
        result = analyzer.prepare_trend_chart_data(chart_df)
        for s in result["series"]:
            for v in s["data"]:
                assert v is None or isinstance(v, float), (
                    f"Expected float or None, got {type(v)} in series '{s['name']}'."
                )

    def test_raises_for_missing_value_col(
        self, analyzer: RxTrendAnalyzer, chart_df: pd.DataFrame
    ) -> None:
        """ValueError must be raised when value_col is absent."""
        with pytest.raises(ValueError, match="not found"):
            analyzer.prepare_trend_chart_data(chart_df, value_col="nonexistent")

    def test_raises_for_unknown_drug_name(
        self, analyzer: RxTrendAnalyzer, chart_df: pd.DataFrame
    ) -> None:
        """ValueError must be raised when the specified drug has no rows."""
        with pytest.raises(ValueError):
            analyzer.prepare_trend_chart_data(chart_df, drug_name="NoSuchDrug")

    def test_does_not_mutate_input(self, analyzer: RxTrendAnalyzer, chart_df: pd.DataFrame) -> None:
        """prepare_trend_chart_data() must leave the input DataFrame unchanged."""
        cols_before = set(chart_df.columns)
        _ = analyzer.prepare_trend_chart_data(chart_df)
        assert set(chart_df.columns) == cols_before


# ---------------------------------------------------------------------------
# 2. prepare_market_share_chart_data
# ---------------------------------------------------------------------------


class TestPrepareMarketShareChartData:
    def test_returns_expected_keys(self, analyzer: RxTrendAnalyzer, chart_df: pd.DataFrame) -> None:
        """Result must contain 'period', 'labels', 'values', 'shares'."""
        result = analyzer.prepare_market_share_chart_data(chart_df, year=2024, month=1)
        assert set(result.keys()) >= {"period", "labels", "values", "shares"}

    def test_period_format(self, analyzer: RxTrendAnalyzer, chart_df: pd.DataFrame) -> None:
        """Period string must be in 'YYYY-MM' format."""
        result = analyzer.prepare_market_share_chart_data(chart_df, year=2024, month=1)
        assert result["period"] == "2024-01"

    def test_shares_sum_to_100(self, analyzer: RxTrendAnalyzer, chart_df: pd.DataFrame) -> None:
        """Market shares must sum to approximately 100 %."""
        result = analyzer.prepare_market_share_chart_data(chart_df, year=2024, month=1)
        assert math.isclose(sum(result["shares"]), 100.0, rel_tol=1e-3)

    def test_labels_match_drug_names(self, analyzer: RxTrendAnalyzer, chart_df: pd.DataFrame) -> None:
        """Labels must be the drug names present in the requested period."""
        result = analyzer.prepare_market_share_chart_data(chart_df, year=2024, month=2)
        assert set(result["labels"]) == {"Alpha", "Beta"}

    def test_lengths_consistent(self, analyzer: RxTrendAnalyzer, chart_df: pd.DataFrame) -> None:
        """labels, values, and shares must all have the same length."""
        result = analyzer.prepare_market_share_chart_data(chart_df, year=2024, month=3)
        assert len(result["labels"]) == len(result["values"]) == len(result["shares"])

    def test_zero_volume_period_produces_zero_shares(
        self, analyzer: RxTrendAnalyzer
    ) -> None:
        """A period with all-zero volumes must yield 0 % shares (no ZeroDivision)."""
        df = pd.DataFrame(
            {
                "month": [5, 5],
                "year":  [2024, 2024],
                "drug_name": ["A", "B"],
                "prescriptions_count": [0, 0],
            }
        )
        result = analyzer.prepare_market_share_chart_data(df, year=2024, month=5)
        assert all(s == 0.0 for s in result["shares"])

    def test_raises_for_missing_period(
        self, analyzer: RxTrendAnalyzer, chart_df: pd.DataFrame
    ) -> None:
        """Requesting a period not in the data must raise ValueError."""
        with pytest.raises(ValueError, match="No data found for period"):
            analyzer.prepare_market_share_chart_data(chart_df, year=2099, month=1)

    def test_raises_for_missing_value_col(
        self, analyzer: RxTrendAnalyzer, chart_df: pd.DataFrame
    ) -> None:
        """ValueError must be raised when value_col is absent."""
        with pytest.raises(ValueError, match="not found"):
            analyzer.prepare_market_share_chart_data(
                chart_df, year=2024, month=1, value_col="nonexistent"
            )

    def test_does_not_mutate_input(self, analyzer: RxTrendAnalyzer, chart_df: pd.DataFrame) -> None:
        """prepare_market_share_chart_data() must not alter the input DataFrame."""
        rows_before = len(chart_df)
        _ = analyzer.prepare_market_share_chart_data(chart_df, year=2024, month=1)
        assert len(chart_df) == rows_before


# ---------------------------------------------------------------------------
# 3. summary_by_drug
# ---------------------------------------------------------------------------


class TestSummaryByDrug:
    def test_returns_one_row_per_drug(self, analyzer: RxTrendAnalyzer, chart_df: pd.DataFrame) -> None:
        """summary_by_drug() must return exactly one row per unique drug."""
        result = analyzer.summary_by_drug(chart_df)
        assert len(result) == chart_df["drug_name"].nunique()

    def test_expected_columns(self, analyzer: RxTrendAnalyzer, chart_df: pd.DataFrame) -> None:
        """Result must contain drug_name, total, mean, min, max, periods."""
        result = analyzer.summary_by_drug(chart_df)
        for col in ("drug_name", "total", "mean", "min", "max", "periods"):
            assert col in result.columns, f"Expected column '{col}' missing."

    def test_total_equals_sum(self, analyzer: RxTrendAnalyzer, chart_df: pd.DataFrame) -> None:
        """'total' for Alpha must equal the sum of its prescriptions_count rows."""
        result = analyzer.summary_by_drug(chart_df)
        alpha_total = result.loc[result["drug_name"] == "Alpha", "total"].iloc[0]
        expected = chart_df.loc[
            chart_df["drug_name"] == "Alpha", "prescriptions_count"
        ].sum()
        assert math.isclose(float(alpha_total), float(expected), rel_tol=1e-6)

    def test_periods_counts_rows(self, analyzer: RxTrendAnalyzer, chart_df: pd.DataFrame) -> None:
        """'periods' must match the number of rows for each drug."""
        result = analyzer.summary_by_drug(chart_df)
        for drug in chart_df["drug_name"].unique():
            expected_count = int((chart_df["drug_name"] == drug).sum())
            actual = int(result.loc[result["drug_name"] == drug, "periods"].iloc[0])
            assert actual == expected_count

    def test_raises_for_missing_value_col(
        self, analyzer: RxTrendAnalyzer, chart_df: pd.DataFrame
    ) -> None:
        """ValueError must be raised when value_col is absent."""
        with pytest.raises(ValueError, match="not found"):
            analyzer.summary_by_drug(chart_df, value_col="nonexistent")

    def test_does_not_mutate_input(self, analyzer: RxTrendAnalyzer, chart_df: pd.DataFrame) -> None:
        """summary_by_drug() must not modify the caller's DataFrame."""
        cols_before = set(chart_df.columns)
        _ = analyzer.summary_by_drug(chart_df)
        assert set(chart_df.columns) == cols_before


# ---------------------------------------------------------------------------
# 4. filter_by_date_range
# ---------------------------------------------------------------------------


class TestFilterByDateRange:
    def test_filters_to_inclusive_range(
        self, analyzer: RxTrendAnalyzer, multi_year_df: pd.DataFrame
    ) -> None:
        """Rows outside the requested (year, month) range must be excluded."""
        result = analyzer.filter_by_date_range(
            multi_year_df,
            start_year=2023, start_month=11,
            end_year=2024,   end_month=2,
        )
        for _, row in result.iterrows():
            period = (int(row["year"]), int(row["month"]))
            assert (2023, 11) <= period <= (2024, 2), (
                f"Row with period {period} is outside [(2023, 11), (2024, 2)]."
            )

    def test_includes_boundary_months(
        self, analyzer: RxTrendAnalyzer, multi_year_df: pd.DataFrame
    ) -> None:
        """Start and end boundary periods must be included in the result."""
        result = analyzer.filter_by_date_range(
            multi_year_df,
            start_year=2023, start_month=10,
            end_year=2024,   end_month=3,
        )
        periods = set(zip(result["year"].tolist(), result["month"].tolist()))
        assert (2023, 10) in periods
        assert (2024, 3) in periods

    def test_returns_empty_when_no_data_in_range(
        self, analyzer: RxTrendAnalyzer, multi_year_df: pd.DataFrame
    ) -> None:
        """A range that contains no data must return an empty DataFrame."""
        result = analyzer.filter_by_date_range(
            multi_year_df,
            start_year=2020, start_month=1,
            end_year=2020,   end_month=12,
        )
        assert result.empty

    def test_raises_when_start_after_end(
        self, analyzer: RxTrendAnalyzer, multi_year_df: pd.DataFrame
    ) -> None:
        """Start period after end period must raise ValueError."""
        with pytest.raises(ValueError, match="after"):
            analyzer.filter_by_date_range(
                multi_year_df,
                start_year=2025, start_month=1,
                end_year=2024,   end_month=6,
            )

    def test_does_not_mutate_input(
        self, analyzer: RxTrendAnalyzer, multi_year_df: pd.DataFrame
    ) -> None:
        """filter_by_date_range() must not alter the original DataFrame."""
        rows_before = len(multi_year_df)
        _ = analyzer.filter_by_date_range(
            multi_year_df, 2023, 10, 2024, 3
        )
        assert len(multi_year_df) == rows_before
