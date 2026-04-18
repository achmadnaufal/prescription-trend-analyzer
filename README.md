# Prescription Trend Analyzer

![Python](https://img.shields.io/badge/python-3.9%2B-blue?logo=python&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green)
![Tests](https://img.shields.io/badge/tests-pytest-orange?logo=pytest)
![Coverage](https://img.shields.io/badge/coverage-80%25%2B-brightgreen)
![Code Style](https://img.shields.io/badge/code%20style-immutable-blueviolet)

A Python library for pharmaceutical prescription volume trend analysis, multi-period forecasting, and chart-ready data preparation.

## Features

- Data ingestion from CSV and Excel files
- Automated analysis and KPI calculation
- Month-over-month (MoM) and year-over-year (YoY) growth calculation
- Market share computation per drug and time period
- Rolling moving-average smoothing
- Per-drug linear regression forecasting
- Visualization-ready chart data preparation (trend lines, market share pie/donut)
- Drug and region filtering helpers with date-range support
- Comprehensive input validation and edge-case handling
- Immutable transformation pipeline (no hidden side effects)

## Installation

```bash
git clone https://github.com/achmadnaufal/prescription-trend-analyzer.git
cd prescription-trend-analyzer
pip install -r requirements.txt
```

## Quick Start

```python
from src.main import RxTrendAnalyzer

analyzer = RxTrendAnalyzer()

# Option 1 — full pipeline in one call
result = analyzer.run("demo/sample_data.csv")
print(result["total_records"])

# Option 2 — step by step for more control
df = analyzer.load_data("demo/sample_data.csv")
analyzer.validate(df)
df = analyzer.preprocess(df)

# Month-over-month growth
df_mom = analyzer.calculate_mom_growth(df)
print(df_mom[["drug_name", "month", "prescriptions_count", "mom_growth_pct"]].head())

# Year-over-year growth
df_yoy = analyzer.calculate_yoy_growth(df)

# Market share per time period
df_share = analyzer.compute_market_share(df)

# 3-month rolling moving average
df_ma = analyzer.moving_average(df, window=3)

# Linear regression forecast for the next 6 months
df_forecast = analyzer.forecast(df, n_periods=6)

# Filtering helpers
lipitor_df  = analyzer.filter_by_drug(df, "Lipitor")
south_df    = analyzer.filter_by_region(df, "South")
window_df   = analyzer.filter_by_date_range(df, 2024, 1, 2024, 6)
```

## New: Anomaly Detector

`src/anomaly_detector.py` flags unusual prescription volume observations per drug using two complementary statistical methods — robust z-score (median + MAD) and Tukey IQR fences — both of which are resistant to the outliers they are detecting.

### Step-by-step usage

**1. Load and preprocess data as usual**

```python
from src.main import RxTrendAnalyzer

analyzer = RxTrendAnalyzer()
df = analyzer.load_data("demo/sample_data.csv")
analyzer.validate(df)
df = analyzer.preprocess(df)
```

**2. Rename column if needed** (the detector defaults to `prescriptions_count`)

```python
# demo/sample_data.csv uses "prescription_volume" after preprocessing
df = df.rename(columns={"prescription_volume": "prescriptions_count"})
```

**3. Run anomaly detection**

```python
from src.anomaly_detector import detect_anomalies

result = detect_anomalies(
    df,
    value_col="prescriptions_count",  # numeric column to inspect
    group_col="drug_name",            # analyse each drug independently
    method="both",                    # "zscore", "iqr", or "both"
    z_threshold=3.0,                  # robust z-score cutoff
    iqr_k=1.5,                        # Tukey fence multiplier
    min_periods=4,                    # skip groups with too few rows
)
```

**4. Inspect flagged rows**

```python
anomalies = result[result["is_anomaly"]]
print(anomalies[["drug_name", "date", "prescriptions_count",
                  "anomaly_score", "anomaly_rationale"]])
```

**5. Tune sensitivity**

```python
# More sensitive: flag milder deviations
result_sensitive = detect_anomalies(df, z_threshold=2.0, iqr_k=1.0)

# Less sensitive: only extreme spikes/drops
result_strict = detect_anomalies(df, z_threshold=4.0, iqr_k=3.0)
```

**Added columns in the returned DataFrame:**

| Column | Type | Description |
|---|---|---|
| `is_anomaly` | `bool` | `True` when the row is flagged |
| `anomaly_score` | `float` | Absolute robust z-score (higher = more unusual) |
| `anomaly_rationale` | `str \| None` | Human-readable explanation for flagged rows |

## New: Trend Change-Point Detector

`src/changepoint_detector.py` locates the single most likely point in time at which a prescription volume series changes its underlying trend (e.g. generic launch, new clinical guideline, supply shock). It uses a piecewise-linear regression scan that selects the split index minimising combined sum of squared residuals (SSR) and reports an SSR-improvement ratio against a single-line baseline so callers can decide significance.

### Step-by-step usage

**1. Load and preprocess data as usual**

```python
from src.main import RxTrendAnalyzer

analyzer = RxTrendAnalyzer()
df = analyzer.load_data("demo/sample_data.csv")
analyzer.validate(df)
df = analyzer.preprocess(df)
df = df.rename(columns={"prescription_volume": "prescriptions_count"})
```

**2. Run change-point detection across all drugs**

```python
from src.changepoint_detector import detect_change_points

report = detect_change_points(
    df,
    value_col="prescriptions_count",
    group_col="drug_name",
    date_col="date",
    min_segment=3,        # >=3 points required on each side of the split
    min_improvement=0.10, # split must explain 10% more variance than a line
)
print(report[report["is_significant"]])
```

**3. Inspect a single series directly**

```python
from src.changepoint_detector import detect_change_point

series = [10, 11, 9, 10, 11, 10, 20, 30, 40, 50, 60]
result = detect_change_point(series, min_segment=3)
print(result.index, result.slope_before, result.slope_after,
      result.improvement_ratio, result.is_significant)
```

**Returned columns from `detect_change_points`:**

| Column | Type | Description |
|---|---|---|
| `group` | `str` | Group key (drug name) or `"_all_"` when grouping is disabled |
| `index` | `int \| None` | Position of the first post-change observation |
| `improvement_ratio` | `float \| None` | `1 - SSR_split / SSR_single`; higher = stronger break |
| `slope_before` / `slope_after` | `float` | OLS slopes for the two segments |
| `intercept_before` / `intercept_after` | `float` | OLS intercepts for the two segments |
| `is_significant` | `bool` | `True` when `improvement_ratio >= min_improvement` |

## New: Seasonality & Period-over-Period Growth

Two helpers in `src/seasonality.py` complete the trend toolkit: classical
additive/multiplicative seasonal decomposition and lag-based growth
computation (MoM, QoQ, YoY, or any integer lag).

### Seasonal decomposition

```python
import pandas as pd
from src.seasonality import seasonal_decompose_series

df = pd.read_csv("demo/sample_data.csv", parse_dates=["date"])
lipitor = (
    df[df["product_name"] == "Lipitor"]
    .sort_values("date")
    .set_index("date")["prescription_volume"]
)

decomp = seasonal_decompose_series(lipitor, period=12, model="additive")
print(decomp.head())
#             observed    trend  seasonal       resid
# 2024-01-01   12450.0      NaN  -103.333         NaN
# 2024-02-01   12820.0      NaN   -41.667         NaN
# ...
```

Edge cases handled:

- Series shorter than `2 * period` raises `SeasonalityError`.
- Isolated NaNs are forward/back-filled before decomposition.
- Multiplicative mode rejects non-positive values.

### Period-over-period growth

```python
from src.seasonality import period_over_period_growth

mom = period_over_period_growth(
    df,
    value_col="prescription_volume",
    group_col="product_name",
    date_col="date",
    lag="mom",            # alias for lag=1
    output="pct",         # "pct" | "abs" | "ratio"
)
yoy = period_over_period_growth(
    df, value_col="prescription_volume",
    group_col="product_name", date_col="date",
    lag="yoy",            # alias for lag=12
)
```

The helper groups by `product_name`, sorts each group by `date`, then
computes growth independently per group.  Rows without a valid lag
(first N rows per group, or rows where the previous value is zero)
receive `NaN`.

## Example Code

### Trend Chart Data

```python
from src.main import RxTrendAnalyzer

analyzer = RxTrendAnalyzer()
df = analyzer.load_data("demo/sample_data.csv")
analyzer.validate(df)
df = analyzer.preprocess(df)

# All drugs as separate chart series
chart = analyzer.prepare_trend_chart_data(df)
# chart["labels"]  -> ["2024-01", "2024-02", ..., "2025-06"]
# chart["series"]  -> [{"name": "Lipitor", "data": [...]}, ...]

# Single-drug drill-down
lipitor_chart = analyzer.prepare_trend_chart_data(df, drug_name="Lipitor")
```

### Market Share Snapshot

```python
# Pie/donut chart payload for January 2025
share_data = analyzer.prepare_market_share_chart_data(df, year=2025, month=1)
# share_data["labels"] -> ["Lipitor", "Lisinopril", "Metformin"]
# share_data["shares"] -> [22.1, 32.7, 35.9]  (sums to ~100%)
```

### Forecasting

```python
# Project 6 months beyond last observation for every drug
forecast_df = analyzer.forecast(df, n_periods=6)
print(forecast_df[["drug_name", "year", "month", "prescriptions_count"]].head(6))
#     drug_name  year  month  prescriptions_count
# 0     Lipitor  2025      7             17560.33
# 1     Lipitor  2025      8             17830.67
# ...
```

### Per-Drug Summary

```python
summary = analyzer.summary_by_drug(df)
print(summary)
#   drug_name      total      mean      min      max  periods
# 0   Lipitor  265370.0  14742.78  12450.0  17300.0       18
# 1  Lisinopril ...
```

## Sample Output

Running `analyzer.run("demo/sample_data.csv")` produces a dictionary similar to:

```
{
  "total_records": 54,
  "columns": ["date", "product_name", "therapeutic_area", ...],
  "missing_pct": {"date": 0.0, "prescription_volume": 0.0, ...},
  "summary_stats": {
    "prescription_volume": {"mean": 18937.04, "std": 4612.11, "min": 12450.0, "max": 28400.0, ...}
  },
  "totals":  {"prescription_volume": 1022600.0, ...},
  "means":   {"prescription_volume": 18937.04, ...}
}
```

## Sample Data

A ready-to-use demo dataset lives at `demo/sample_data.csv` (54 rows, 3 drugs across 18 months — Jan 2024 to Jun 2025).

| Column | Description |
|---|---|
| `date` | First day of the observation month (YYYY-MM-DD) |
| `product_name` | Brand name of the drug |
| `therapeutic_area` | Drug class (e.g. Cardiovascular, Endocrinology) |
| `region` | Geographic region |
| `prescription_volume` | Total prescription count for the month |
| `market_share_pct` | Market share percentage within therapeutic area |
| `physician_count` | Number of prescribing physicians |

### Generating Larger Synthetic Data

```python
from src.data_generator import generate_sample

df = generate_sample(n=300, seed=42)
df.to_csv("data/synthetic_300.csv", index=False)
```

## Project Structure

```
prescription-trend-analyzer/
├── src/
│   ├── __init__.py
│   ├── main.py                 # Core analysis, forecasting, and viz-prep logic
│   ├── anomaly_detector.py     # Robust z-score / IQR anomaly flagging
│   ├── changepoint_detector.py # Piecewise-linear trend change-point detector
│   └── data_generator.py       # Synthetic data generator
├── tests/
│   ├── __init__.py
│   ├── test_analyzer.py    # Validation, preprocessing, growth, market share, MA
│   ├── test_forecasting.py # Forecasting and calendar-arithmetic helpers
│   └── test_visualization.py # Chart data prep, summary, date-range filtering
├── demo/
│   └── sample_data.csv     # 54-row demo dataset (18 months, 3 drugs)
├── examples/
│   └── basic_usage.py      # Runnable usage example
├── data/                   # Data directory (gitignored for real data)
├── CHANGELOG.md
├── requirements.txt
└── README.md
```

## Running Tests

```bash
# All tests
pytest tests/ -v

# Specific test modules
pytest tests/test_analyzer.py -v
pytest tests/test_forecasting.py -v
pytest tests/test_visualization.py -v

# With coverage report
pip install pytest-cov
pytest tests/ --cov=src --cov-report=term-missing
```

All tests are organized into focused modules:

| Module | What it covers |
|---|---|
| `test_analyzer.py` | Validation, preprocessing, MoM/YoY growth, market share, moving average, filtering, end-to-end pipeline |
| `test_forecasting.py` | Linear forecast helper, calendar month arithmetic, `forecast()` happy paths and edge cases |
| `test_visualization.py` | Trend chart data prep, market share chart data, `summary_by_drug()`, date-range filtering |
| `test_anomaly_detector.py` | Spike detection, immutability, determinism, flat/zero/short series, multi-drug grouping, all method variants |
| `test_changepoint_detector.py` | Piecewise-linear break detection, perfectly linear series, NaN handling, non-monotonic dates, multi-group grouping, validation guards |

## License

MIT License — free to use, modify, and distribute.
