"""
Prescription volume trend analysis, forecasting, and visualization prep.

This module is the primary entry point for the prescription-trend-analyzer
library.  It exposes :class:`RxTrendAnalyzer`, a high-level façade that
orchestrates data loading, validation, preprocessing, KPI calculation, and
simple time-series forecasting for pharmaceutical prescription data.

Typical usage::

    from src.main import RxTrendAnalyzer

    analyzer = RxTrendAnalyzer()
    result   = analyzer.run("demo/sample_data.csv")
    print(result["total_records"])

Author: github.com/achmadnaufal
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: tiny linear-regression forecast (no statsmodels dependency required
# for the core path)
# ---------------------------------------------------------------------------


def _linear_forecast(values: np.ndarray, n_periods: int) -> np.ndarray:
    """Fit an OLS trend line and extrapolate ``n_periods`` steps ahead.

    Args:
        values: 1-D array of historical observations (chronological order).
        n_periods: Number of future periods to project.

    Returns:
        1-D array of length ``n_periods`` containing forecast values.

    Raises:
        ValueError: If ``values`` has fewer than 2 observations or
            ``n_periods`` is not a positive integer.
    """
    if len(values) < 2:
        raise ValueError(
            f"At least 2 historical observations are required for forecasting; "
            f"got {len(values)}."
        )
    if n_periods < 1:
        raise ValueError(
            f"'n_periods' must be a positive integer; got {n_periods}."
        )

    x = np.arange(len(values), dtype=float)
    slope, intercept = np.polyfit(x, values.astype(float), 1)
    future_x = np.arange(len(values), len(values) + n_periods, dtype=float)
    return slope * future_x + intercept


# ---------------------------------------------------------------------------
# Main analyser class
# ---------------------------------------------------------------------------


class RxTrendAnalyzer:
    """High-level façade for prescription trend analysis.

    Provides data loading, validation, preprocessing, and a full suite of
    analytical methods covering trend calculation, market share computation,
    rolling moving-averages, simple linear forecasting, and helpers that
    transform results into chart-ready data structures.

    All transformation methods follow an *immutable* pattern: they return new
    :class:`~pandas.DataFrame` objects and never modify the caller's data.

    Attributes:
        config: Optional configuration dictionary for future extensibility.

    Example::

        analyzer = RxTrendAnalyzer()
        df       = analyzer.load_data("demo/sample_data.csv")
        analyzer.validate(df)
        df = analyzer.preprocess(df)
        df_mom = analyzer.calculate_mom_growth(df)
    """

    REQUIRED_COLUMNS: frozenset = frozenset(
        {"month", "year", "drug_name", "prescriptions_count"}
    )

    MIN_FORECAST_PERIODS: int = 2
    """Minimum number of historical rows (per drug) needed to produce a forecast."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialise the analyser.

        Args:
            config: Optional dictionary of configuration overrides.
                Supported keys (all optional):

                - ``"forecast_periods"`` (*int*, default ``3``): number of
                  future months projected by :meth:`forecast`.
                - ``"ma_window"`` (*int*, default ``3``): default rolling-
                  average window used by :meth:`moving_average`.
        """
        self.config: Dict[str, Any] = config or {}

    # ------------------------------------------------------------------
    # I/O helpers
    # ------------------------------------------------------------------

    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load prescription data from a CSV or Excel file.

        Args:
            filepath: Absolute or relative path to the input file.
                Supported extensions: ``.csv``, ``.xlsx``, ``.xls``.

        Returns:
            Raw :class:`~pandas.DataFrame` exactly as read from disk.

        Raises:
            FileNotFoundError: If *filepath* does not point to an existing
                file.
            ValueError: If the file extension is not ``.csv``, ``.xlsx``,
                or ``.xls``.
        """
        p = Path(filepath)
        if not p.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")
        if p.suffix in (".xlsx", ".xls"):
            logger.debug("Loading Excel file: %s", filepath)
            return pd.read_excel(filepath)
        if p.suffix == ".csv":
            logger.debug("Loading CSV file: %s", filepath)
            return pd.read_csv(filepath)
        raise ValueError(
            f"Unsupported file type '{p.suffix}'. Use .csv, .xlsx, or .xls."
        )

    # ------------------------------------------------------------------
    # Validation & preprocessing
    # ------------------------------------------------------------------

    def validate(self, df: pd.DataFrame) -> bool:
        """Validate that a DataFrame satisfies minimum structural requirements.

        Checks:
        1. The DataFrame is **not empty**.
        2. All columns in :attr:`REQUIRED_COLUMNS` are present (comparison is
           case-insensitive and strips leading/trailing whitespace).

        Args:
            df: DataFrame to inspect.

        Returns:
            ``True`` when all checks pass.

        Raises:
            ValueError: If the DataFrame is empty or one or more required
                columns are absent.
        """
        if df.empty:
            raise ValueError(
                "Input DataFrame is empty. Ensure the source file contains "
                "at least one data row."
            )

        normalised_cols = {c.lower().strip().replace(" ", "_") for c in df.columns}
        missing = self.REQUIRED_COLUMNS - normalised_cols
        if missing:
            raise ValueError(
                f"DataFrame is missing required columns: {sorted(missing)}. "
                f"Found columns: {sorted(df.columns.tolist())}."
            )
        logger.debug("Validation passed (%d rows, %d columns).", len(df), len(df.columns))
        return True

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and normalise the input DataFrame.

        The following operations are performed (the original DataFrame is
        **never mutated** — a new object is always returned):

        1. Drop fully-empty rows (all values ``NaN``).
        2. Normalise column names to *lowercase snake_case*.
        3. Coerce ``month`` and ``year`` to integers where possible.
        4. Coerce ``prescriptions_count`` to numeric, replacing non-parsable
           values with ``0``.
        5. If a ``date`` column is present, parse it as
           :class:`~pandas.Timestamp`.

        Args:
            df: Raw input DataFrame (typically returned by :meth:`load_data`).

        Returns:
            A new, cleaned :class:`~pandas.DataFrame`.
        """
        cleaned = df.copy()
        cleaned = cleaned.dropna(how="all")

        # Normalise column names
        cleaned.columns = [
            c.lower().strip().replace(" ", "_") for c in cleaned.columns
        ]

        # Coerce integer time columns
        for col in ("month", "year"):
            if col in cleaned.columns:
                cleaned = cleaned.assign(
                    **{col: pd.to_numeric(cleaned[col], errors="coerce")}
                )

        # Coerce prescription volume — default missing to 0
        if "prescriptions_count" in cleaned.columns:
            cleaned = cleaned.assign(
                prescriptions_count=pd.to_numeric(
                    cleaned["prescriptions_count"], errors="coerce"
                ).fillna(0)
            )

        # Optional date parsing
        if "date" in cleaned.columns:
            cleaned = cleaned.assign(
                date=pd.to_datetime(cleaned["date"], errors="coerce")
            )

        logger.debug(
            "Preprocessing complete: %d rows, %d columns.", len(cleaned), len(cleaned.columns)
        )
        return cleaned

    # ------------------------------------------------------------------
    # Core analysis
    # ------------------------------------------------------------------

    def analyze(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Run core descriptive analysis and return a summary metrics dictionary.

        Preprocesses the data, then computes:

        - Total record count.
        - Column list.
        - Per-column missing-value percentages.
        - (when numeric columns exist) ``summary_stats``, ``totals``, and
          ``means``.

        Args:
            df: Input DataFrame (preprocessed internally — the caller's object
                is not modified).

        Returns:
            Dictionary with keys:

            ``total_records`` (*int*)
                Number of rows after preprocessing.

            ``columns`` (*list[str]*)
                Column names after normalisation.

            ``missing_pct`` (*dict[str, float]*)
                Percentage of ``NaN`` values per column.

            ``summary_stats`` (*dict*, optional)
                Output of :meth:`~pandas.DataFrame.describe` on numeric
                columns (only present when at least one numeric column exists).

            ``totals`` (*dict*, optional)
                Column-wise sum for all numeric columns.

            ``means`` (*dict*, optional)
                Column-wise mean for all numeric columns.
        """
        preprocessed = self.preprocess(df)

        result: Dict[str, Any] = {
            "total_records": len(preprocessed),
            "columns": list(preprocessed.columns),
            "missing_pct": (
                preprocessed.isnull().sum() / max(len(preprocessed), 1) * 100
            )
            .round(1)
            .to_dict(),
        }

        numeric_df = preprocessed.select_dtypes(include="number")
        if not numeric_df.empty:
            result["summary_stats"] = numeric_df.describe().round(3).to_dict()
            result["totals"] = numeric_df.sum().round(2).to_dict()
            result["means"] = numeric_df.mean().round(3).to_dict()

        return result

    def run(self, filepath: str) -> Dict[str, Any]:
        """Execute the full analysis pipeline: load → validate → analyze.

        Args:
            filepath: Path to the input data file (see :meth:`load_data` for
                supported formats).

        Returns:
            Analysis result dictionary as returned by :meth:`analyze`.

        Raises:
            FileNotFoundError: Propagated from :meth:`load_data`.
            ValueError: Propagated from :meth:`load_data` or :meth:`validate`.
        """
        df = self.load_data(filepath)
        self.validate(df)
        return self.analyze(df)

    # ------------------------------------------------------------------
    # Trend calculations
    # ------------------------------------------------------------------

    def calculate_mom_growth(
        self,
        df: pd.DataFrame,
        value_col: str = "prescriptions_count",
    ) -> pd.DataFrame:
        """Calculate month-over-month (MoM) percentage growth per drug.

        Rows are sorted by ``(drug_name, year, month)`` before the percentage
        change is computed within each drug group.  The first observation for
        each drug will always have ``NaN`` growth (no prior period exists).

        Args:
            df: Preprocessed DataFrame containing at minimum ``drug_name``,
                ``year``, ``month``, and *value_col*.
            value_col: Name of the numeric column to compute growth for.
                Defaults to ``"prescriptions_count"``.

        Returns:
            New DataFrame (original unmodified) with an additional
            ``mom_growth_pct`` column (``float``, rounded to 2 d.p.).

        Raises:
            ValueError: If *value_col* is not a column in *df*.
        """
        if value_col not in df.columns:
            raise ValueError(
                f"Column '{value_col}' not found. "
                f"Available columns: {list(df.columns)}."
            )

        sorted_df = (
            df.sort_values(["drug_name", "year", "month"])
            .reset_index(drop=True)
            .copy()
        )

        growth = sorted_df.groupby("drug_name")[value_col].pct_change() * 100
        return sorted_df.assign(mom_growth_pct=growth.round(2))

    def calculate_yoy_growth(
        self,
        df: pd.DataFrame,
        value_col: str = "prescriptions_count",
    ) -> pd.DataFrame:
        """Calculate year-over-year (YoY) percentage growth per drug and month.

        A lagged copy of the DataFrame (year + 1) is merged onto the original
        to align prior-year values.  YoY growth is then computed as
        ``(current - prior) / prior × 100``.  Rows without a matching prior-
        year record receive ``NaN``.

        Args:
            df: Preprocessed DataFrame containing ``drug_name``, ``year``,
                ``month``, and *value_col*.
            value_col: Column to compute growth for.
                Defaults to ``"prescriptions_count"``.

        Returns:
            New DataFrame with an additional ``yoy_growth_pct`` column
            (``float``, rounded to 2 d.p.).

        Raises:
            ValueError: If *value_col* is not a column in *df*.
        """
        if value_col not in df.columns:
            raise ValueError(
                f"Column '{value_col}' not found. "
                f"Available columns: {list(df.columns)}."
            )

        prior_col = f"{value_col}_prior"
        lag = df.assign(year=df["year"] + 1)[
            ["drug_name", "year", "month", value_col]
        ].rename(columns={value_col: prior_col})

        merged = df.merge(lag, on=["drug_name", "year", "month"], how="left")

        has_prior = merged[prior_col].notna() & (merged[prior_col] != 0)
        yoy = np.where(
            has_prior,
            (merged[value_col] - merged[prior_col]) / merged[prior_col] * 100,
            np.nan,
        )

        return merged.drop(columns=[prior_col]).assign(
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
        """Compute market share percentage for each drug within each time period.

        Market share is defined as each drug's *value_col* divided by the
        sum across **all** drugs for the same ``(year, month)`` (plus any
        extra columns in *group_cols*), multiplied by 100.  Periods where the
        total is zero receive a share of ``0.0`` (no :class:`ZeroDivisionError`
        is raised).

        Args:
            df: Preprocessed DataFrame.
            group_cols: Extra columns to include in the grouping key alongside
                ``year`` and ``month`` (e.g. ``["region"]``).  Defaults to
                ``None`` (time-period grouping only).
            value_col: Numeric column to use as the base for share calculation.
                Defaults to ``"prescriptions_count"``.

        Returns:
            New DataFrame with an added ``computed_market_share_pct`` column
            (``float``, rounded to 2 d.p.).

        Raises:
            ValueError: If *value_col* is not present in *df*.
        """
        if value_col not in df.columns:
            raise ValueError(
                f"Column '{value_col}' not found. "
                f"Available columns: {list(df.columns)}."
            )

        period_cols = ["year", "month"] + (group_cols or [])
        totals = df.groupby(period_cols)[value_col].transform("sum")

        share = np.where(totals != 0, df[value_col] / totals * 100, 0.0)
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

        Rows are sorted by ``(drug_name, year, month)`` prior to windowing so
        that values from different drugs never bleed into each other.  The
        window uses ``min_periods=1`` so that early rows (fewer than *window*
        predecessors) receive a partial average rather than ``NaN``.

        Args:
            df: Preprocessed DataFrame.
            window: Rolling window width in periods.  Must be >= 1.
                Defaults to ``3``.
            value_col: Column to smooth.
                Defaults to ``"prescriptions_count"``.

        Returns:
            New DataFrame with a ``{value_col}_ma{window}`` column appended.

        Raises:
            ValueError: If *window* < 1 or *value_col* is absent from *df*.
        """
        if window < 1:
            raise ValueError(f"'window' must be >= 1, got {window}.")
        if value_col not in df.columns:
            raise ValueError(
                f"Column '{value_col}' not found. "
                f"Available columns: {list(df.columns)}."
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
    # Forecasting
    # ------------------------------------------------------------------

    def forecast(
        self,
        df: pd.DataFrame,
        n_periods: int = 3,
        value_col: str = "prescriptions_count",
    ) -> pd.DataFrame:
        """Project future prescription volumes using per-drug linear regression.

        For each unique drug in *df* an ordinary least-squares trend line is
        fitted to the chronologically ordered historical observations.  The
        resulting slope and intercept are used to extrapolate *n_periods*
        months beyond the last observed period.

        Drugs with fewer than :attr:`MIN_FORECAST_PERIODS` observations are
        silently skipped (a warning is logged).

        Args:
            df: Preprocessed DataFrame containing ``drug_name``, ``year``,
                ``month``, and *value_col* with **at least two rows per drug**.
            n_periods: Number of future months to project.  Must be >= 1.
                Defaults to ``3``.
            value_col: Column whose values are regressed.
                Defaults to ``"prescriptions_count"``.

        Returns:
            New DataFrame with columns:

            - ``drug_name``
            - ``year``
            - ``month``
            - *value_col* (forecasted value, rounded to 2 d.p.)
            - ``is_forecast`` (``True`` for all rows in this result)

        Raises:
            ValueError: If *value_col* is absent from *df* or *n_periods* < 1.
            ValueError: If **no** drug has enough data to produce a forecast.
        """
        if value_col not in df.columns:
            raise ValueError(
                f"Column '{value_col}' not found. "
                f"Available columns: {list(df.columns)}."
            )
        if n_periods < 1:
            raise ValueError(
                f"'n_periods' must be >= 1, got {n_periods}."
            )

        sorted_df = df.sort_values(["drug_name", "year", "month"]).reset_index(drop=True)
        forecast_rows: List[Dict[str, Any]] = []

        for drug_name, group in sorted_df.groupby("drug_name"):
            if len(group) < self.MIN_FORECAST_PERIODS:
                logger.warning(
                    "Skipping forecast for '%s': only %d observation(s) "
                    "(minimum %d required).",
                    drug_name,
                    len(group),
                    self.MIN_FORECAST_PERIODS,
                )
                continue

            values = group[value_col].to_numpy()
            projected = _linear_forecast(values, n_periods)

            # Determine the starting month/year for forecast rows
            last_year: int = int(group["year"].iloc[-1])
            last_month: int = int(group["month"].iloc[-1])
            future_periods = _advance_months(last_year, last_month, n_periods)

            for (fut_year, fut_month), proj_value in zip(future_periods, projected):
                forecast_rows.append(
                    {
                        "drug_name": drug_name,
                        "year": fut_year,
                        "month": fut_month,
                        value_col: round(float(proj_value), 2),
                        "is_forecast": True,
                    }
                )

        if not forecast_rows:
            raise ValueError(
                "No drug has sufficient historical data to produce a forecast. "
                f"Minimum required: {self.MIN_FORECAST_PERIODS} observations per drug."
            )

        return pd.DataFrame(forecast_rows)

    # ------------------------------------------------------------------
    # Visualization preparation
    # ------------------------------------------------------------------

    def prepare_trend_chart_data(
        self,
        df: pd.DataFrame,
        drug_name: Optional[str] = None,
        value_col: str = "prescriptions_count",
    ) -> Dict[str, Any]:
        """Transform prescription data into a chart-ready dictionary.

        Produces a structure suitable for passing directly to charting
        libraries (Matplotlib, Plotly, Vega-Altair, etc.).  If *drug_name* is
        specified the result is filtered to that drug only; otherwise all
        drugs are included as separate series.

        Args:
            df: Preprocessed DataFrame containing ``drug_name``, ``year``,
                ``month``, and *value_col*.
            drug_name: Optional drug to isolate.  Case-insensitive.  Pass
                ``None`` (the default) to include all drugs.
            value_col: Column containing the metric to plot.
                Defaults to ``"prescriptions_count"``.

        Returns:
            Dictionary with keys:

            ``labels`` (*list[str]*)
                Period labels in ``"YYYY-MM"`` format, chronologically sorted.

            ``series`` (*list[dict]*)
                One entry per drug.  Each entry has:

                - ``"name"`` (*str*): drug name.
                - ``"data"`` (*list[float | None]*): values aligned to
                  *labels* (``None`` where a period is absent for that drug).

            ``value_col`` (*str*)
                The column used as the y-axis metric.

        Raises:
            ValueError: If *value_col* is absent from *df* or the resulting
                dataset (after optional drug filter) is empty.
        """
        if value_col not in df.columns:
            raise ValueError(
                f"Column '{value_col}' not found. "
                f"Available columns: {list(df.columns)}."
            )

        working = df.copy()
        if drug_name is not None:
            mask = working["drug_name"].str.lower() == drug_name.lower()
            working = working.loc[mask].reset_index(drop=True)

        if working.empty:
            raise ValueError(
                f"No data found after filtering"
                + (f" for drug '{drug_name}'." if drug_name else ".")
            )

        working = working.sort_values(["year", "month"]).reset_index(drop=True)
        working = working.assign(
            period=working["year"].astype(str)
            + "-"
            + working["month"].astype(str).str.zfill(2)
        )

        all_labels: List[str] = sorted(working["period"].unique().tolist())
        drug_list: List[str] = sorted(working["drug_name"].unique().tolist())

        series: List[Dict[str, Any]] = []
        for drug in drug_list:
            drug_data = (
                working.loc[working["drug_name"] == drug]
                .set_index("period")[value_col]
                .reindex(all_labels)
            )
            series.append(
                {
                    "name": drug,
                    "data": [
                        None if pd.isna(v) else round(float(v), 2)
                        for v in drug_data
                    ],
                }
            )

        return {
            "labels": all_labels,
            "series": series,
            "value_col": value_col,
        }

    def prepare_market_share_chart_data(
        self,
        df: pd.DataFrame,
        year: int,
        month: int,
        value_col: str = "prescriptions_count",
    ) -> Dict[str, Any]:
        """Build a pie/donut chart payload for market share at a given period.

        Filters *df* to the requested ``(year, month)`` period and computes
        each drug's percentage share of the total *value_col*.

        Args:
            df: Preprocessed DataFrame.
            year: Calendar year to snapshot.
            month: Calendar month (1–12) to snapshot.
            value_col: Numeric column used to calculate shares.
                Defaults to ``"prescriptions_count"``.

        Returns:
            Dictionary with keys:

            ``period`` (*str*)
                Label in ``"YYYY-MM"`` format.

            ``labels`` (*list[str]*)
                Drug names.

            ``values`` (*list[float]*)
                Corresponding prescription volumes.

            ``shares`` (*list[float]*)
                Corresponding market share percentages (sum ≈ 100 %).

        Raises:
            ValueError: If *value_col* is absent or no rows match the period.
        """
        if value_col not in df.columns:
            raise ValueError(
                f"Column '{value_col}' not found. "
                f"Available columns: {list(df.columns)}."
            )

        period_df = df.loc[(df["year"] == year) & (df["month"] == month)].copy()
        if period_df.empty:
            raise ValueError(
                f"No data found for period {year}-{month:02d}. "
                f"Available years: {sorted(df['year'].unique().tolist())}."
            )

        total = period_df[value_col].sum()
        shares = (
            (period_df[value_col] / total * 100).round(2).tolist()
            if total > 0
            else [0.0] * len(period_df)
        )

        return {
            "period": f"{year}-{month:02d}",
            "labels": period_df["drug_name"].tolist(),
            "values": period_df[value_col].round(2).tolist(),
            "shares": shares,
        }

    # ------------------------------------------------------------------
    # Filtering helpers
    # ------------------------------------------------------------------

    def filter_by_drug(self, df: pd.DataFrame, drug_name: str) -> pd.DataFrame:
        """Return only rows that match a specific drug name (case-insensitive).

        Args:
            df: Preprocessed DataFrame with a ``drug_name`` column.
            drug_name: Drug name to filter for.  The comparison ignores case.

        Returns:
            New DataFrame containing only the matching rows.  Returns an empty
            DataFrame (not an error) when no rows match.
        """
        mask = df["drug_name"].str.lower() == drug_name.lower()
        return df.loc[mask].reset_index(drop=True).copy()

    def filter_by_region(self, df: pd.DataFrame, region: str) -> pd.DataFrame:
        """Return only rows matching a specific region (case-insensitive).

        Args:
            df: Preprocessed DataFrame with a ``region`` column.
            region: Region name to filter for.  The comparison ignores case.

        Returns:
            New DataFrame containing only the matching rows.

        Raises:
            ValueError: If the DataFrame has no ``region`` column.
        """
        if "region" not in df.columns:
            raise ValueError(
                "DataFrame does not contain a 'region' column. "
                f"Available columns: {list(df.columns)}."
            )
        mask = df["region"].str.lower() == region.lower()
        return df.loc[mask].reset_index(drop=True).copy()

    def filter_by_date_range(
        self,
        df: pd.DataFrame,
        start_year: int,
        start_month: int,
        end_year: int,
        end_month: int,
    ) -> pd.DataFrame:
        """Return rows within an inclusive ``(year, month)`` range.

        Useful for slicing a multi-year dataset to a specific analysis window
        without the need for a parsed date column.

        Args:
            df: Preprocessed DataFrame with ``year`` and ``month`` columns.
            start_year: First year of the range (inclusive).
            start_month: First month of the range (inclusive, 1–12).
            end_year: Last year of the range (inclusive).
            end_month: Last month of the range (inclusive, 1–12).

        Returns:
            New DataFrame containing only rows within the specified window.

        Raises:
            ValueError: If ``start`` is chronologically after ``end``.
        """
        start = (start_year, start_month)
        end = (end_year, end_month)
        if start > end:
            raise ValueError(
                f"Start period {start_year}-{start_month:02d} is after "
                f"end period {end_year}-{end_month:02d}."
            )

        period = list(zip(df["year"].tolist(), df["month"].tolist()))
        mask = [start <= p <= end for p in period]
        return df.loc[mask].reset_index(drop=True).copy()

    # ------------------------------------------------------------------
    # Export helpers
    # ------------------------------------------------------------------

    def to_dataframe(self, result: Dict[str, Any]) -> pd.DataFrame:
        """Convert a flat or one-level-nested analysis result dict to a DataFrame.

        Nested dictionaries are flattened with dot-separated keys, e.g.
        ``"totals.prescriptions_count"``.  This is intended for exporting
        :meth:`analyze` output to CSV or for display in tabular form.

        Args:
            result: Dictionary returned by :meth:`analyze`.

        Returns:
            Two-column DataFrame with columns ``metric`` and ``value``.
        """
        rows: List[Dict[str, Any]] = []
        for k, v in result.items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    rows.append({"metric": f"{k}.{kk}", "value": vv})
            else:
                rows.append({"metric": k, "value": v})
        return pd.DataFrame(rows)

    def summary_by_drug(
        self,
        df: pd.DataFrame,
        value_col: str = "prescriptions_count",
    ) -> pd.DataFrame:
        """Aggregate prescription data by drug across all time periods.

        Computes total, mean, minimum, and maximum *value_col* for each drug,
        along with the number of recorded periods.

        Args:
            df: Preprocessed DataFrame.
            value_col: Numeric column to summarise.
                Defaults to ``"prescriptions_count"``.

        Returns:
            New aggregated DataFrame indexed by ``drug_name`` with columns:
            ``total``, ``mean``, ``min``, ``max``, ``periods``.

        Raises:
            ValueError: If *value_col* is absent from *df*.
        """
        if value_col not in df.columns:
            raise ValueError(
                f"Column '{value_col}' not found. "
                f"Available columns: {list(df.columns)}."
            )

        agg = (
            df.groupby("drug_name")[value_col]
            .agg(total="sum", mean="mean", min="min", max="max", periods="count")
            .round(2)
            .reset_index()
        )
        return agg


# ---------------------------------------------------------------------------
# Internal utility: calendar month arithmetic
# ---------------------------------------------------------------------------


def _advance_months(
    year: int, month: int, n: int
) -> List[Tuple[int, int]]:
    """Return a list of ``(year, month)`` tuples for the *n* months after the given period.

    Args:
        year: Starting calendar year.
        month: Starting calendar month (1–12).
        n: Number of future periods to generate.

    Returns:
        List of ``(year, month)`` tuples in chronological order.

    Example::

        _advance_months(2024, 11, 3)
        # [(2024, 12), (2025, 1), (2025, 2)]
    """
    result: List[Tuple[int, int]] = []
    cur_year, cur_month = year, month
    for _ in range(n):
        cur_month += 1
        if cur_month > 12:
            cur_month = 1
            cur_year += 1
        result.append((cur_year, cur_month))
    return result
