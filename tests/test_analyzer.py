"""
Unit tests for RxTrendAnalyzer.

Run with:
    pytest tests/ -v
"""
import math
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import pytest

# Allow imports from project root without installation
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.main import RxTrendAnalyzer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def analyzer():
    """Return a default RxTrendAnalyzer instance."""
    return RxTrendAnalyzer()


@pytest.fixture()
def sample_df():
    """Return a minimal, well-formed DataFrame covering two drugs / four months."""
    return pd.DataFrame(
        {
            "month": [1, 2, 3, 4, 1, 2, 3, 4],
            "year": [2024] * 8,
            "drug_name": ["DrugA"] * 4 + ["DrugB"] * 4,
            "generic_name": ["genA"] * 4 + ["genB"] * 4,
            "therapeutic_class": ["ClassX"] * 8,
            "region": ["North"] * 4 + ["South"] * 4,
            "prescriptions_count": [1000, 1100, 1200, 1300, 500, 520, 540, 560],
            "total_units": [30000, 33000, 36000, 39000, 15000, 15600, 16200, 16800],
            "avg_days_supply": [30] * 8,
            "new_rx_count": [200, 220, 240, 260, 100, 104, 108, 112],
            "refill_count": [800, 880, 960, 1040, 400, 416, 432, 448],
            "market_share_pct": [66.7, 67.9, 69.0, 69.9, 33.3, 32.1, 31.0, 30.1],
        }
    )


@pytest.fixture()
def sample_csv(tmp_path, sample_df):
    """Write the sample DataFrame to a temporary CSV and return its path."""
    path = tmp_path / "data.csv"
    sample_df.to_csv(path, index=False)
    return str(path)


# ---------------------------------------------------------------------------
# 1. Validation tests
# ---------------------------------------------------------------------------


class TestValidation:
    def test_validate_passes_for_valid_df(self, analyzer, sample_df):
        """validate() should return True for a well-formed DataFrame."""
        assert analyzer.validate(sample_df) is True

    def test_validate_raises_for_empty_df(self, analyzer):
        """validate() must raise ValueError when the DataFrame is empty."""
        with pytest.raises(ValueError, match="empty"):
            analyzer.validate(pd.DataFrame())

    def test_validate_raises_for_missing_required_columns(self, analyzer):
        """validate() must raise ValueError when required columns are absent."""
        bad_df = pd.DataFrame({"foo": [1, 2], "bar": [3, 4]})
        with pytest.raises(ValueError, match="missing required columns"):
            analyzer.validate(bad_df)

    def test_load_data_raises_for_missing_file(self, analyzer):
        """load_data() must raise FileNotFoundError for a non-existent path."""
        with pytest.raises(FileNotFoundError):
            analyzer.load_data("/nonexistent/path/data.csv")

    def test_load_data_raises_for_unsupported_extension(self, analyzer, tmp_path):
        """load_data() must raise ValueError for unsupported file types."""
        p = tmp_path / "data.json"
        p.write_text("{}")
        with pytest.raises(ValueError, match="Unsupported file type"):
            analyzer.load_data(str(p))


# ---------------------------------------------------------------------------
# 2. Preprocessing tests
# ---------------------------------------------------------------------------


class TestPreprocess:
    def test_preprocess_normalises_column_names(self, analyzer):
        """preprocess() must convert column names to lowercase snake_case."""
        df = pd.DataFrame({"Drug Name": [1], "Month": [1], "Year": [2024]})
        result = analyzer.preprocess(df)
        assert "drug_name" in result.columns
        assert "month" in result.columns

    def test_preprocess_fills_missing_prescriptions_with_zero(self, analyzer):
        """preprocess() must coerce NaN prescriptions_count to 0."""
        df = pd.DataFrame(
            {
                "month": [1, 2],
                "year": [2024, 2024],
                "drug_name": ["X", "X"],
                "prescriptions_count": [100, np.nan],
            }
        )
        result = analyzer.preprocess(df)
        assert result["prescriptions_count"].iloc[1] == 0.0

    def test_preprocess_does_not_mutate_input(self, analyzer, sample_df):
        """preprocess() must return a new object and leave the original intact."""
        original_cols = list(sample_df.columns)
        _ = analyzer.preprocess(sample_df)
        assert list(sample_df.columns) == original_cols


# ---------------------------------------------------------------------------
# 3. MoM growth tests
# ---------------------------------------------------------------------------


class TestMoMGrowth:
    def test_mom_growth_expected_values(self, analyzer, sample_df):
        """calculate_mom_growth() must produce correct MoM % values."""
        processed = analyzer.preprocess(sample_df)
        result = analyzer.calculate_mom_growth(processed)

        drug_a = result[result["drug_name"] == "DrugA"].sort_values("month")
        # Month 1 → 2: (1100 - 1000) / 1000 * 100 = 10 %
        assert math.isclose(drug_a.iloc[1]["mom_growth_pct"], 10.0, rel_tol=1e-3)
        # Month 2 → 3: (1200 - 1100) / 1100 * 100 ≈ 9.09 %
        assert math.isclose(
            drug_a.iloc[2]["mom_growth_pct"], 9.09, rel_tol=1e-2
        )

    def test_mom_growth_first_row_is_nan(self, analyzer, sample_df):
        """First month for each drug must have NaN MoM growth."""
        processed = analyzer.preprocess(sample_df)
        result = analyzer.calculate_mom_growth(processed)

        for drug in ("DrugA", "DrugB"):
            drug_rows = result[result["drug_name"] == drug].sort_values("month")
            assert math.isnan(drug_rows.iloc[0]["mom_growth_pct"])

    def test_mom_growth_single_point_is_nan(self, analyzer):
        """Single data point per drug must yield NaN growth (no prior period)."""
        df = pd.DataFrame(
            {
                "month": [1],
                "year": [2024],
                "drug_name": ["OnlyOne"],
                "prescriptions_count": [500],
            }
        )
        result = analyzer.calculate_mom_growth(df)
        assert math.isnan(result.iloc[0]["mom_growth_pct"])

    def test_mom_growth_raises_for_missing_column(self, analyzer, sample_df):
        """calculate_mom_growth() must raise ValueError for unknown value_col."""
        processed = analyzer.preprocess(sample_df)
        with pytest.raises(ValueError, match="not found"):
            analyzer.calculate_mom_growth(processed, value_col="nonexistent")

    def test_mom_growth_does_not_mutate_input(self, analyzer, sample_df):
        """calculate_mom_growth() must not modify the caller's DataFrame."""
        processed = analyzer.preprocess(sample_df)
        cols_before = set(processed.columns)
        _ = analyzer.calculate_mom_growth(processed)
        assert set(processed.columns) == cols_before


# ---------------------------------------------------------------------------
# 4. YoY growth tests
# ---------------------------------------------------------------------------


class TestYoYGrowth:
    def test_yoy_growth_is_nan_without_prior_year(self, analyzer, sample_df):
        """YoY growth must be NaN when only one year of data exists."""
        processed = analyzer.preprocess(sample_df)
        result = analyzer.calculate_yoy_growth(processed)
        # All data is 2024 — no 2023 rows to compare against
        assert result["yoy_growth_pct"].isna().all()

    def test_yoy_growth_correct_value_with_two_years(self, analyzer):
        """calculate_yoy_growth() must produce the correct YoY % with two years."""
        df = pd.DataFrame(
            {
                "month": [1, 1],
                "year": [2023, 2024],
                "drug_name": ["DrugA", "DrugA"],
                "prescriptions_count": [1000, 1200],
            }
        )
        result = analyzer.calculate_yoy_growth(df)
        row_2024 = result[(result["year"] == 2024) & (result["drug_name"] == "DrugA")]
        assert math.isclose(row_2024.iloc[0]["yoy_growth_pct"], 20.0, rel_tol=1e-3)


# ---------------------------------------------------------------------------
# 5. Market share tests
# ---------------------------------------------------------------------------


class TestMarketShare:
    def test_market_share_sums_to_100_per_period(self, analyzer, sample_df):
        """Computed market shares must sum to 100 % within each (year, month)."""
        processed = analyzer.preprocess(sample_df)
        result = analyzer.compute_market_share(processed)

        for (year, month), group in result.groupby(["year", "month"]):
            total = group["computed_market_share_pct"].sum()
            assert math.isclose(total, 100.0, rel_tol=1e-3), (
                f"Market share for {year}-{month:02d} sums to {total}, expected 100"
            )

    def test_market_share_raises_for_missing_column(self, analyzer, sample_df):
        """compute_market_share() must raise ValueError for unknown value_col."""
        processed = analyzer.preprocess(sample_df)
        with pytest.raises(ValueError, match="not found"):
            analyzer.compute_market_share(processed, value_col="nonexistent")

    def test_market_share_zero_prescriptions_handled(self, analyzer):
        """Periods with zero total prescriptions must produce 0 % share (no ZeroDivision)."""
        df = pd.DataFrame(
            {
                "month": [1, 1],
                "year": [2024, 2024],
                "drug_name": ["DrugA", "DrugB"],
                "prescriptions_count": [0, 0],
            }
        )
        result = analyzer.compute_market_share(df)
        assert (result["computed_market_share_pct"] == 0.0).all()

    def test_market_share_does_not_mutate_input(self, analyzer, sample_df):
        """compute_market_share() must not add columns to the caller's DataFrame."""
        processed = analyzer.preprocess(sample_df)
        cols_before = set(processed.columns)
        _ = analyzer.compute_market_share(processed)
        assert set(processed.columns) == cols_before


# ---------------------------------------------------------------------------
# 6. Moving average tests
# ---------------------------------------------------------------------------


class TestMovingAverage:
    def test_moving_average_window_3(self, analyzer, sample_df):
        """3-period MA for DrugA month 3 should equal mean of months 1-3."""
        processed = analyzer.preprocess(sample_df)
        result = analyzer.moving_average(processed, window=3)

        drug_a = result[result["drug_name"] == "DrugA"].sort_values("month")
        expected = (1000 + 1100 + 1200) / 3
        assert math.isclose(
            drug_a[drug_a["month"] == 3].iloc[0]["prescriptions_count_ma3"],
            expected,
            rel_tol=1e-3,
        )

    def test_moving_average_single_point(self, analyzer):
        """MA with a single row must equal the row's own value (min_periods=1)."""
        df = pd.DataFrame(
            {
                "month": [1],
                "year": [2024],
                "drug_name": ["Solo"],
                "prescriptions_count": [750],
            }
        )
        result = analyzer.moving_average(df, window=3)
        assert result.iloc[0]["prescriptions_count_ma3"] == 750.0

    def test_moving_average_invalid_window_raises(self, analyzer, sample_df):
        """window < 1 must raise ValueError."""
        processed = analyzer.preprocess(sample_df)
        with pytest.raises(ValueError, match="window"):
            analyzer.moving_average(processed, window=0)

    def test_moving_average_does_not_mutate_input(self, analyzer, sample_df):
        """moving_average() must return a new DataFrame without altering input."""
        processed = analyzer.preprocess(sample_df)
        cols_before = set(processed.columns)
        _ = analyzer.moving_average(processed, window=2)
        assert set(processed.columns) == cols_before


# ---------------------------------------------------------------------------
# 7. Filtering tests
# ---------------------------------------------------------------------------


class TestFiltering:
    def test_filter_by_drug_returns_correct_rows(self, analyzer, sample_df):
        """filter_by_drug() must return only rows for the requested drug."""
        processed = analyzer.preprocess(sample_df)
        result = analyzer.filter_by_drug(processed, "DrugA")
        assert (result["drug_name"] == "DrugA").all()
        assert len(result) == 4

    def test_filter_by_drug_case_insensitive(self, analyzer, sample_df):
        """filter_by_drug() must match regardless of case."""
        processed = analyzer.preprocess(sample_df)
        assert len(analyzer.filter_by_drug(processed, "druga")) == 4
        assert len(analyzer.filter_by_drug(processed, "DRUGA")) == 4

    def test_filter_by_drug_returns_empty_for_unknown(self, analyzer, sample_df):
        """filter_by_drug() must return an empty DataFrame for unknown drugs."""
        processed = analyzer.preprocess(sample_df)
        result = analyzer.filter_by_drug(processed, "NonExistentDrug")
        assert result.empty

    def test_filter_by_region_returns_correct_rows(self, analyzer, sample_df):
        """filter_by_region() must return only rows for the requested region."""
        processed = analyzer.preprocess(sample_df)
        result = analyzer.filter_by_region(processed, "North")
        assert (result["region"].str.lower() == "north").all()
        assert len(result) == 4

    def test_filter_by_region_raises_without_region_column(self, analyzer):
        """filter_by_region() must raise ValueError if 'region' column is absent."""
        df = pd.DataFrame(
            {
                "month": [1],
                "year": [2024],
                "drug_name": ["X"],
                "prescriptions_count": [100],
            }
        )
        with pytest.raises(ValueError, match="region"):
            analyzer.filter_by_region(df, "North")

    def test_filter_does_not_mutate_input(self, analyzer, sample_df):
        """filter_by_drug() must not alter the original DataFrame."""
        processed = analyzer.preprocess(sample_df)
        original_len = len(processed)
        _ = analyzer.filter_by_drug(processed, "DrugA")
        assert len(processed) == original_len


# ---------------------------------------------------------------------------
# 8. End-to-end pipeline test
# ---------------------------------------------------------------------------


class TestPipeline:
    def test_run_pipeline_with_sample_csv(self, analyzer, sample_csv):
        """run() must complete the full pipeline and return expected keys."""
        result = analyzer.run(sample_csv)
        assert result["total_records"] == 8
        assert "summary_stats" in result
        assert "totals" in result
        assert "means" in result

    def test_to_dataframe_flattens_nested_result(self, analyzer, sample_df):
        """to_dataframe() must produce a two-column DataFrame from nested results."""
        result = analyzer.analyze(sample_df)
        df = analyzer.to_dataframe(result)
        assert set(df.columns) == {"metric", "value"}
        assert len(df) > 0

    def test_analyze_missing_months_handled(self, analyzer):
        """analyze() must not raise when some months have zero prescriptions."""
        df = pd.DataFrame(
            {
                "month": [1, 2, 3],
                "year": [2024, 2024, 2024],
                "drug_name": ["X", "X", "X"],
                "prescriptions_count": [1000, 0, 1200],
            }
        )
        result = analyzer.analyze(df)
        assert result["total_records"] == 3

    def test_full_chain_mom_then_market_share(self, analyzer, sample_df):
        """Chaining MoM growth then market share must not raise and must add both cols."""
        processed = analyzer.preprocess(sample_df)
        with_mom = analyzer.calculate_mom_growth(processed)
        with_share = analyzer.compute_market_share(with_mom)
        assert "mom_growth_pct" in with_share.columns
        assert "computed_market_share_pct" in with_share.columns
