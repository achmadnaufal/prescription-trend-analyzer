# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] - 2026-04-19

### Added
- **Seasonality module** (`src/seasonality.py`):
  - `seasonal_decompose_series` - validated wrapper around
    `statsmodels.tsa.seasonal.seasonal_decompose` that returns a tidy
    `observed / trend / seasonal / resid` DataFrame, forward/back-fills
    isolated NaNs, enforces `len(series) >= 2 * period`, and rejects
    non-positive values for multiplicative decomposition.
  - `period_over_period_growth` - per-group lag-based growth with
    string aliases (`"mom"`, `"qoq"`, `"yoy"`, `"wow"`) and three
    output modes (`pct`, `abs`, `ratio`).  Handles zero-denominator
    rows as NaN, respects immutability, and sorts non-monotonic
    dates within each group before computing the lag.
  - `SeasonalityError` custom exception for all validation failures.
- **`tests/test_seasonality.py`**: 21 pytest cases covering
  decomposition shape, NaN filling, short-series guards, bad-input
  rejection, MoM/YoY/abs/ratio variants, zero-previous handling,
  multi-drug group isolation, immutability, and non-monotonic date
  sorting.
- README "New: Seasonality & Period-over-Period Growth" section with
  runnable examples.
- Re-exported the new public API from `src/__init__.py`.

## [Unreleased] - 2026-04-18

### Added
- **Trend Change-Point Detector** (`src/changepoint_detector.py`): locates the single most likely trend break in a prescription volume series via a piecewise-linear regression scan, fitting two OLS segments at every legal split index and selecting the one that minimises combined SSR. Reports an SSR-improvement ratio against a single-line baseline so callers can decide significance via the configurable `min_improvement` threshold.
- **`detect_change_points` DataFrame helper**: runs the detector on every group (e.g. per drug), preserving immutability of the caller's DataFrame, sorting each group by an optional `date_col`, and dropping NaN gaps before evaluation.
- **`ChangePointResult` dataclass**: frozen, JSON-friendly result container with `index`, `improvement_ratio`, `slope_before`, `slope_after`, `intercept_before`, `intercept_after`, and `is_significant` fields.
- **`tests/test_changepoint_detector.py`**: 19 pytest tests covering happy-path detection, single-point and too-short series, all-zero and perfectly linear series, NaN-in-series guard, non-monotonic date inputs, multi-drug independence, immutability, and validation guards (`min_segment`, `min_improvement`, missing columns, empty DataFrame).
- README "New: Trend Change-Point Detector" section with step-by-step usage and output column reference.
- Re-exported the new public API from `src/__init__.py` for ergonomic top-level imports.

## [Unreleased] - 2026-04-17

### Added
- **Anomaly Detector** (`src/anomaly_detector.py`): flags unusual prescription volume observations per drug using robust z-score (median + MAD) and Tukey IQR fences, with configurable sensitivity (`z_threshold`, `iqr_k`), per-group independent analysis, human-readable `anomaly_rationale` strings, and graceful handling of flat series, all-zero series, and groups with insufficient data.
- **`tests/test_anomaly_detector.py`**: 17 pytest tests covering spike detection, immutability, determinism, flat/zero/short series edge cases, multi-drug grouping independence, rationale population, all three method variants (`zscore`, `iqr`, `both`), and invalid-input guards.
- README "New: Anomaly Detector" section with step-by-step usage, sensitivity tuning guide, and output column reference.

## [0.2.0] - 2026-04-16

### Added
- **Forecasting** (`RxTrendAnalyzer.forecast`): per-drug linear regression projections for any number of future periods, with graceful handling of insufficient data.
- **Visualization preparation** (`prepare_trend_chart_data`, `prepare_market_share_chart_data`): produces chart-ready dicts (labels + series arrays) compatible with Matplotlib, Plotly, and Vega-Altair.
- **`summary_by_drug`**: aggregates total, mean, min, max, and period count per drug across all time periods.
- **`filter_by_date_range`**: slice datasets to an inclusive `(year, month)` window without requiring a parsed date column.
- **`_linear_forecast` helper**: standalone OLS extrapolation utility.
- **`_advance_months` helper**: calendar-aware month arithmetic that correctly handles year-boundary crossings.
- **`tests/test_forecasting.py`**: 14 focused tests covering the forecast pipeline and its helpers.
- **`tests/test_visualization.py`**: 20 focused tests covering chart data prep, market share snapshot, summary, and date-range filtering.
- Expanded `demo/sample_data.csv` to 54 rows covering 3 drugs across 18 months (Jan 2024 – Jun 2025) with columns: `date`, `product_name`, `therapeutic_area`, `region`, `prescription_volume`, `market_share_pct`, `physician_count`.

### Improved
- **Docstrings**: all public methods now have full Args/Returns/Raises documentation.
- **Type hints**: complete annotations on all functions, including return types.
- **Input validation**: additional guards in `analyze()` (division-by-zero safety when DataFrame has 0 rows after preprocessing), `forecast()`, `filter_by_date_range()`, and both chart-data helpers.
- **Immutability**: all transformation methods confirmed to return new DataFrames without mutating the caller's data; enforced in tests.
- **`REQUIRED_COLUMNS`** changed to `frozenset` for hashability.
- README updated with badges, "Quick Start", example code, "Sample Output", "Project Structure", and "Running Tests" sections.

## [0.1.0] - 2026-04-15

### Added
- Initial release with data loading, validation, preprocessing, MoM/YoY growth, market share, moving average, and filtering helpers.
- Unit tests with pytest.
- Sample data for demo purposes.
