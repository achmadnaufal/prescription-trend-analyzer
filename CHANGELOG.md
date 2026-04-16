# Changelog

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
