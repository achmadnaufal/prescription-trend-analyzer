"""
Prescription volume trend analysis, forecasting, and visualization.

Author: github.com/achmadnaufal
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List


class RxTrendAnalyzer:
    """Prescription trend analysis tool.

    Provides data loading, validation, preprocessing, and a suite of
    analytical methods covering trend calculation, market share, moving
    averages, and region/drug filtering.

    Attributes:
        config: Optional configuration dictionary for future extensibility.
    """

    REQUIRED_COLUMNS = {
        "month",
        "year",
        "drug_name",
        "prescriptions_count",
    }

    def __init__(self, config: Optional[Dict] = None) -> None:
        """Initialise the analyser.

        Args:
            config: Optional dictionary of configuration overrides.
        """
        self.config = config or {}

    # ------------------------------------------------------------------
    # I/O helpers
    # ------------------------------------------------------------------

    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load prescription data from a CSV or Excel file.

        Args:
            filepath: Absolute or relative path to the input file.

        Returns:
            Raw DataFrame exactly as read from disk.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file extension is not supported.
        """
        p = Path(filepath)
        if not p.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")
        if p.suffix in (".xlsx", ".xls"):
            return pd.read_excel(filepath)
        if p.suffix == ".csv":
            return pd.read_csv(filepath)
        raise ValueError(
            f"Unsupported file type '{p.suffix}'. Use .csv, .xlsx, or .xls."
        )

    # ------------------------------------------------------------------
    # Validation & preprocessing
    # ------------------------------------------------------------------

    def validate(self, df: pd.DataFrame) -> bool:
        """Validate that the DataFrame meets minimum requirements.

        Checks for non-empty data and the presence of required columns
        (``month``, ``year``, ``drug_name``, ``prescriptions_count``).

        Args:
            df: DataFrame to validate.

        Returns:
            ``True`` when validation passes.

        Raises:
            ValueError: If the DataFrame is empty or missing required columns.
        """
        if df.empty:
            raise ValueError("Input DataFrame is empty.")

        lowered = {c.lower().strip().replace(" ", "_") for c in df.columns}
        missing = self.REQUIRED_COLUMNS - lowered
        if missing:
            raise ValueError(
                f"DataFrame is missing required columns: {sorted(missing)}"
            )
        return True

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and normalise the input DataFrame.

        Operations performed (all immutable — a new DataFrame is returned):
        - Drop fully-empty rows.
        - Normalise column names to snake_case lowercase.
        - Coerce ``month`` and ``year`` to integers where possible.
        - Coerce ``prescriptions_count`` to numeric, filling NaN with 0.

        Args:
            df: Raw input DataFrame.

        Returns:
            A new, cleaned DataFrame.
        """
        cleaned = df.copy()
        cleaned = cleaned.dropna(how="all")
        cleaned.columns = [
            c.lower().strip().replace(" ", "_") for c in cleaned.columns
        ]

        for col in ("month", "year"):
            if col in cleaned.columns:
                cleaned = cleaned.assign(
                    **{col: pd.to_numeric(cleaned[col], errors="coerce")}
                )

        if "prescriptions_count" in cleaned.columns:
            cleaned = cleaned.assign(
                prescriptions_count=pd.to_numeric(
                    cleaned["prescriptions_count"], errors="coerce"
                ).fillna(0)
            )

        return cleaned

    # ------------------------------------------------------------------
    # Core analysis
    # ------------------------------------------------------------------

    def analyze(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Run core analysis and return summary metrics.

        Preprocesses the data, then computes record count, column list,
        missing-value percentages, and (if numeric columns exist) summary
        statistics, column totals, and column means.

        Args:
            df: Input DataFrame (will be preprocessed internally).

        Returns:
            Dictionary with keys: ``total_records``, ``columns``,
            ``missing_pct``, and optionally ``summary_stats``, ``totals``,
            ``means``.
        """
        df = self.preprocess(df)
        result: Dict[str, Any] = {
            "total_records": len(df),
            "columns": list(df.columns),
            "missing_pct": (df.isnull().sum() / len(df) * 100).round(1).to_dict(),
        }
        numeric_df = df.select_dtypes(include="number")
        if not numeric_df.empty:
            result["summary_stats"] = numeric_df.describe().round(3).to_dict()
            result["totals"] = numeric_df.sum().round(2).to_dict()
            result["means"] = numeric_df.mean().round(3).to_dict()
        return result

    def run(self, filepath: str) -> Dict[str, Any]:
        """Full pipeline: load → validate → analyze.

        Args:
            filepath: Path to the input data file.

        Returns:
            Analysis result dictionary (see :meth:`analyze`).
        """
        df = self.load_data(filepath)
        self.validate(df)
        return self.analyze(df)

    # ------------------------------------------------------------------
    # Trend calculations
    # ------------------------------------------------------------------

    def calculate_mom_growth(
        self, df: pd.DataFrame, value_col: str = "prescriptions_count"
    ) -> pd.DataFrame:
        """Calculate month-over-month (MoM) percentage growth per drug.

        Rows are sorted by ``(drug_name, year, month)`` before computing the
        percentage change.  Returns a new DataFrame; the original is not
        mutated.

        For a single data point per drug the growth value is ``NaN`` (not
        enough history to compute a rate of change).

        Args:
            df: Preprocessed DataFrame containing at minimum ``drug_name``,
                ``year``, ``month``, and ``value_col``.
            value_col: Name of the column to compute growth for.
                Defaults to ``"prescriptions_count"``.

        Returns:
            New DataFrame with an additional ``mom_growth_pct`` column.

        Raises:
            ValueError: If ``value_col`` is not present in the DataFrame.
        """
        if value_col not in df.columns:
            raise ValueError(
                f"Column '{value_col}' not found. Available: {list(df.columns)}"
            )

        sorted_df = (
            df.sort_values(["drug_name", "year", "month"])
            .reset_index(drop=True)
            .copy()
        )

        growth_series = sorted_df.groupby("drug_name")[value_col].pct_change() * 100
        return sorted_df.assign(mom_growth_pct=growth_series.round(2))

    def calculate_yoy_growth(
        self, df: pd.DataFrame, value_col: str = "prescriptions_count"
    ) -> pd.DataFrame:
        """Calculate year-over-year (YoY) percentage growth per drug/month.

        Merges the DataFrame with a lagged copy offset by one year, then
        computes ``(current - prior) / prior * 100``.  Returns a new
        DataFrame; the original is not mutated.

        Args:
            df: Preprocessed DataFrame containing ``drug_name``, ``year``,
                ``month``, and ``value_col``.
            value_col: Column to compute growth for.

        Returns:
            New DataFrame with an additional ``yoy_growth_pct`` column.

        Raises:
            ValueError: If ``value_col`` is not present in the DataFrame.
        """
        if value_col not in df.columns:
            raise ValueError(
                f"Column '{value_col}' not found. Available: {list(df.columns)}"
            )

        lag = df.assign(year=df["year"] + 1)[
            ["drug_name", "year", "month", value_col]
        ].rename(columns={value_col: f"{value_col}_prior"})

        merged = df.merge(lag, on=["drug_name", "year", "month"], how="left")

        yoy = np.where(
            merged[f"{value_col}_prior"].notna() & (merged[f"{value_col}_prior"] != 0),
            (merged[value_col] - merged[f"{value_col}_prior"])
            / merged[f"{value_col}_prior"]
            * 100,
            np.nan,
        )

        return merged.drop(columns=[f"{value_col}_prior"]).assign(
            yoy_growth_pct=np.round(yoy, 2)
        )

    # ------------------------------------------------------------------
    # Market share
    # ------------------------------------------------------------------

    def compute_market_share(
        self,
        df: pd.DataFrame,
        group_cols: Optional[List[str]] = None,
        value_col: str = "prescriptions_count",
    ) -> pd.DataFrame:
        """Compute market share percentage within each time period.

        Market share is calculated as each drug's ``value_col`` divided by
        the total across all drugs for the same ``(year, month)`` (and any
        additional ``group_cols``), multiplied by 100.

        Args:
            df: Preprocessed DataFrame.
            group_cols: Additional columns to group by alongside ``year``
                and ``month``.  Defaults to ``None`` (time period only).
            value_col: Numerics column to base share calculation on.

        Returns:
            New DataFrame with an added ``computed_market_share_pct`` column.

        Raises:
            ValueError: If ``value_col`` is not present.
        """
        if value_col not in df.columns:
            raise ValueError(
                f"Column '{value_col}' not found. Available: {list(df.columns)}"
            )

        period_cols = ["year", "month"] + (group_cols or [])
        totals = (
            df.groupby(period_cols)[value_col]
            .transform("sum")
        )

        share = np.where(
            totals != 0,
            df[value_col] / totals * 100,
            0.0,
        )

        return df.assign(computed_market_share_pct=np.round(share, 2))

    # ------------------------------------------------------------------
    # Moving average / smoothing
    # ------------------------------------------------------------------

    def moving_average(
        self,
        df: pd.DataFrame,
        window: int = 3,
        value_col: str = "prescriptions_count",
    ) -> pd.DataFrame:
        """Apply a rolling moving average to smooth prescription counts.

        Rows are sorted by ``(drug_name, year, month)`` before the window is
        applied per drug group so that values from different drugs never bleed
        into each other.

        Args:
            df: Preprocessed DataFrame.
            window: Number of periods in the rolling window.  Must be >= 1.
            value_col: Column to smooth.

        Returns:
            New DataFrame with a ``{value_col}_ma{window}`` column.

        Raises:
            ValueError: If ``window`` < 1 or ``value_col`` is absent.
        """
        if window < 1:
            raise ValueError(f"'window' must be >= 1, got {window}.")
        if value_col not in df.columns:
            raise ValueError(
                f"Column '{value_col}' not found. Available: {list(df.columns)}"
            )

        sorted_df = (
            df.sort_values(["drug_name", "year", "month"])
            .reset_index(drop=True)
            .copy()
        )

        ma_col = f"{value_col}_ma{window}"
        ma_values = (
            sorted_df.groupby("drug_name")[value_col]
            .transform(lambda s: s.rolling(window, min_periods=1).mean())
            .round(2)
        )
        return sorted_df.assign(**{ma_col: ma_values})

    # ------------------------------------------------------------------
    # Filtering helpers
    # ------------------------------------------------------------------

    def filter_by_drug(self, df: pd.DataFrame, drug_name: str) -> pd.DataFrame:
        """Return rows matching a specific drug name (case-insensitive).

        Args:
            df: Preprocessed DataFrame containing a ``drug_name`` column.
            drug_name: Drug name to filter for.

        Returns:
            New filtered DataFrame.  Empty if no match is found.
        """
        mask = df["drug_name"].str.lower() == drug_name.lower()
        return df.loc[mask].reset_index(drop=True).copy()

    def filter_by_region(self, df: pd.DataFrame, region: str) -> pd.DataFrame:
        """Return rows matching a specific region (case-insensitive).

        Args:
            df: Preprocessed DataFrame containing a ``region`` column.
            region: Region name to filter for.

        Returns:
            New filtered DataFrame.  Empty if no match is found.

        Raises:
            ValueError: If the DataFrame has no ``region`` column.
        """
        if "region" not in df.columns:
            raise ValueError("DataFrame does not contain a 'region' column.")
        mask = df["region"].str.lower() == region.lower()
        return df.loc[mask].reset_index(drop=True).copy()

    # ------------------------------------------------------------------
    # Export helpers
    # ------------------------------------------------------------------

    def to_dataframe(self, result: Dict) -> pd.DataFrame:
        """Convert a flat or nested analysis result dictionary to a DataFrame.

        Nested dicts are flattened with dot-separated keys (e.g.
        ``"totals.prescriptions_count"``).

        Args:
            result: Dictionary returned by :meth:`analyze`.

        Returns:
            Two-column DataFrame with columns ``metric`` and ``value``.
        """
        rows = []
        for k, v in result.items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    rows.append({"metric": f"{k}.{kk}", "value": vv})
            else:
                rows.append({"metric": k, "value": v})
        return pd.DataFrame(rows)
