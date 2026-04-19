# Prescription Trend Analyzer

![Python](https://img.shields.io/badge/python-3.9%2B-blue?logo=python&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green)
![Tests](https://img.shields.io/badge/tests-pytest-orange?logo=pytest)
![Coverage](https://img.shields.io/badge/coverage-80%25%2B-brightgreen)
![Code Style](https://img.shields.io/badge/code%20style-immutable-blueviolet)

A Python library for pharmaceutical prescription volume trend analysis,
multi-period forecasting, seasonality decomposition, change-point and
anomaly detection, and chart-ready data preparation.  Designed for
honest, small-scale commercial analytics: every transformation is
validated, fully typed, immutable, and covered by focused unit tests.

## Scope (Honest)

This library is an **analytical toolkit**, not a forecasting platform.
It does not ship with a database layer, a dashboard, or a scheduler.
It is intentionally limited to deterministic, interpretable pandas /
NumPy / statsmodels transforms so every output can be audited by hand
and reasoned about in a review setting.

## Features

- Data ingestion from CSV and Excel files with schema validation
- Month-over-month (MoM), year-over-year (YoY), and arbitrary-lag
  period-over-period growth
- **Moving Annual Total (MAT), MAT growth %, and MAT share %** — the
  classic pharma rolling-12-month view (`src/mat.py`)
- Market share computation per drug and time period
- Rolling moving-average smoothing
- Per-drug linear regression forecasting
- Classical seasonal decomposition (additive / multiplicative) via
  statsmodels STL wrapper
- Robust-z-score and Tukey-IQR anomaly flagging
- Piecewise-linear trend change-point detection
- Visualization-ready chart data preparation (trend lines, market
  share pie/donut)
- Drug, region, and date-range filtering helpers
- Comprehensive input validation and edge-case handling (empty,
  single-point, all-zero, missing-month, non-datetime index)
- Immutable transformation pipeline — no hidden side effects

## Installation

```bash
git clone https://github.com/achmadnaufal/prescription-trend-analyzer.git
cd prescription-trend-analyzer
pip install -r requirements.txt
```

Python 3.9+ is required.  The only heavy dependencies are pandas,
NumPy, matplotlib (for charting), and statsmodels (for the seasonal
decomposition wrapper).

## Quick Start

The demo dataset at `demo/sample_data.csv` contains 72 rows — 3
products (Atorvastatin/Cardiology, Metformin/Endocrinology,
Amoxicillin/Anti-infective) across 24 months (Jan 2024 – Dec 2025).

```python
import pandas as pd
from src.main import RxTrendAnalyzer
from src.mat import mat_growth, mat_share, moving_annual_total

# Load the demo dataset directly (month is a proper date)
df = pd.read_csv("demo/sample_data.csv", parse_dates=["month"])

# 1) Moving Annual Total — rolling 12-month sum per product
mat = moving_annual_total(
    df,
    value_col="rx_volume",
    group_col="product",
    date_col="month",
)
print(mat.tail(3)[["month", "product", "rx_volume", "rx_volume_mat12"]])

# 2) MAT growth % — YoY change of MAT (classic pharma KPI)
growth = mat_growth(
    df,
    value_col="rx_volume",
    group_col="product",
    date_col="month",
)
print(
    growth.dropna(subset=["rx_volume_mat12_growth_pct"])[
        ["month", "product", "rx_volume_mat12_growth_pct"]
    ]
)

# 3) MAT share — each product's MAT as % of total MAT that month
share = mat_share(
    df,
    value_col="rx_volume",
    group_col="product",
    date_col="month",
)
print(share.tail(3)[["month", "product", "rx_volume_mat12_share_pct"]])

# 4) Descriptive summary via the RxTrendAnalyzer pipeline.
analyzer = RxTrendAnalyzer()
summary = analyzer.run("demo/sample_data.csv")
print(summary["total_records"])  # 72
```

> **Note on column shapes.**  The MAT helpers accept any sortable
> `date_col` (dates, strings, integers).  The legacy `RxTrendAnalyzer`
> pipeline, on the other hand, expects a *numeric* `month` column
> (1-12) and a separate `year` column — `demo/sample_data.csv` ships
> both shapes so everything works out of the box.  To forecast with
> the legacy pipeline, load the CSV without date-parsing:
>
> ```python
> raw = analyzer.load_data("demo/sample_data.csv")
> analyzer.validate(raw)
> # drop the date-string `month` so preprocess() does not try to
> # coerce it — the back-compat numeric month lives elsewhere:
> pre = analyzer.preprocess(raw.drop(columns=["month"]).rename(
>     columns={"year": "year"}
> ).assign(month=pd.to_datetime(raw["month"]).dt.month))
> forecast_df = analyzer.forecast(pre, n_periods=6)
> ```

Additional helpers on `RxTrendAnalyzer` include
`calculate_mom_growth`, `calculate_yoy_growth`, `compute_market_share`,
`moving_average`, `prepare_trend_chart_data`,
`prepare_market_share_chart_data`, `summary_by_drug`, and the
filtering helpers `filter_by_drug`, `filter_by_region`, and
`filter_by_date_range`.

## New: Moving Annual Total (MAT)

`src/mat.py` implements the canonical pharma *Moving Annual Total* — the
trailing 12-month sum — plus its two sibling KPIs used on every
commercial dashboard:

| Helper | Output | Typical Use |
|---|---|---|
| `moving_annual_total(df, …)` | `{value}_mat{W}` | Smoothed annualised view of a brand's volume |
| `mat_growth(df, …)`          | `{value}_mat{W}_growth_pct` | Year-over-year % change of MAT |
| `mat_share(df, …)`           | `{value}_mat{W}_share_pct`  | Each brand's MAT as a % of total MAT per date |

All three helpers are immutable, fully validated, and handle the
standard edge cases honestly:

- **Single data point** — returns `NaN` unless `min_periods=1` is
  requested (then the partial sum is emitted).
- **Missing months / calendar gaps** — the window is *positional*
  (12 observations, not 12 calendar months).  The docstring makes
  this explicit so callers know to reindex to a complete monthly
  grid first if they need strict calendar semantics.
- **All-zero volumes** — MAT returns `0.0` (not `NaN`).  MAT growth
  returns `NaN` when the lagged MAT is zero (division guard).
- **Non-datetime index** — any sortable `date_col` is accepted
  (including ISO-formatted strings).
- **Empty DataFrame** — raises `MATError` immediately with a clear
  message.

### Example

```python
import pandas as pd
from src.mat import moving_annual_total, mat_growth, mat_share

df = pd.read_csv("demo/sample_data.csv", parse_dates=["month"])

mat     = moving_annual_total(df, "rx_volume", "product", "month")
grow    = mat_growth(df,           "rx_volume", "product", "month")
shares  = mat_share(df,            "rx_volume", "product", "month")

# Latest MAT snapshot across all products:
latest = shares["month"].max()
print(shares[shares["month"] == latest]
        .loc[:, ["product", "rx_volume_mat12", "rx_volume_mat12_share_pct"]])
```

Tuning:

- `window=4` for a rolling *quarterly* total.
- `min_periods=1` to emit partial sums before the first full window.
- `output_col="my_name"` to rename the generated column.

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

A ready-to-use demo dataset lives at `demo/sample_data.csv` (72 rows —
3 products across 24 months, Jan 2024 to Dec 2025).

| Column | Description |
|---|---|
| `month` | First day of the observation month (YYYY-MM-DD) |
| `product` | Generic name of the molecule (e.g. `Atorvastatin`) |
| `therapeutic_area` | Drug class (e.g. `Cardiology`, `Endocrinology`, `Anti-infective`) |
| `rx_volume` | Total prescription count for the product in the month |
| `market_volume` | Total prescriptions across the whole market (denominator for share) |
| `channel` | Dispensing channel (e.g. `Retail`, `Hospital`) |
| `drug_name`, `year`, `prescriptions_count` | Back-compat aliases so the legacy `RxTrendAnalyzer` pipeline works unchanged |

The trend is a slight monthly upward drift with realistic pharma ratios
between products.  Useful for demonstrating MAT growth (+15-21% YoY at
Dec 2025), share dynamics, and forecasting without requiring real PHI.

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
│   ├── main.py                 # Core analysis, forecasting, viz-prep
│   ├── mat.py                  # Moving Annual Total, MAT growth, MAT share
│   ├── seasonality.py          # STL decomposition + PoP growth helpers
│   ├── anomaly_detector.py     # Robust z-score / IQR anomaly flagging
│   ├── changepoint_detector.py # Piecewise-linear trend change-point detector
│   └── data_generator.py       # Synthetic data generator
├── tests/
│   ├── __init__.py
│   ├── test_analyzer.py           # Validation, preprocessing, growth, share, MA
│   ├── test_forecasting.py        # Forecasting and calendar-arithmetic helpers
│   ├── test_visualization.py      # Chart data prep, summary, date-range filtering
│   ├── test_mat.py                # Moving Annual Total, MAT growth, MAT share
│   ├── test_seasonality.py        # Decomposition + period-over-period growth
│   ├── test_anomaly_detector.py   # Anomaly flagging
│   └── test_changepoint_detector.py # Trend change-point detection
├── demo/
│   └── sample_data.csv     # 72-row demo dataset (24 months, 3 products)
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
| `test_seasonality.py` | STL decomposition, NaN handling, period validation, PoP growth aliases, immutability, multi-group isolation |
| `test_mat.py` | MAT rolling sums with known inputs, MAT growth YoY, MAT share sums to 100%, single-point / all-zero / calendar-gap / non-datetime / empty edge cases |

## License

MIT License — free to use, modify, and distribute.
