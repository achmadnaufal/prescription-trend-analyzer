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
│   ├── main.py             # Core analysis, forecasting, and viz-prep logic
│   └── data_generator.py   # Synthetic data generator
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

## License

MIT License — free to use, modify, and distribute.
