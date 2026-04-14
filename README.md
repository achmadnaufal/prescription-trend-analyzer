# Prescription Trend Analyzer

Prescription volume trend analysis, forecasting, and visualization.

## Features

- Data ingestion from CSV / Excel input files
- Automated analysis and KPI calculation
- Month-over-month (MoM) and year-over-year (YoY) growth calculation
- Market share computation per drug and time period
- Rolling moving-average smoothing
- Drug and region filtering helpers
- Summary statistics and trend reporting
- Sample data for immediate demo usage

## Installation

```bash
pip install -r requirements.txt
pip install pytest  # for running tests
```

## Quick Start

```python
from src.main import RxTrendAnalyzer

analyzer = RxTrendAnalyzer()

# Option 1 — run the full pipeline in one call
result = analyzer.run("demo/sample_data.csv")
print(result["total_records"])   # 20

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

# Filter helpers
lipitor_df = analyzer.filter_by_drug(df, "Lipitor")
northeast_df = analyzer.filter_by_region(df, "Northeast")
```

## Sample Data

A ready-to-use demo dataset lives at `demo/sample_data.csv` (20 rows, 5 drugs across 4 months).

| Column | Description |
|---|---|
| `month` | Calendar month (1–12) |
| `year` | Calendar year |
| `drug_name` | Brand name of the drug |
| `generic_name` | Generic / INN name |
| `therapeutic_class` | Drug class (e.g. Statins, Antidiabetics) |
| `region` | Geographic region |
| `prescriptions_count` | Total prescription count |
| `total_units` | Total dispensed units |
| `avg_days_supply` | Average days supply per prescription |
| `new_rx_count` | New prescriptions (first-fills) |
| `refill_count` | Refill prescriptions |
| `market_share_pct` | Market share percentage within class |

## Running Tests

```bash
pytest tests/ -v
```

All tests live in `tests/test_analyzer.py` and cover:
- Input validation and edge cases (empty data, missing columns)
- Preprocessing and immutability guarantees
- MoM growth calculation (including single-data-point edge case)
- YoY growth calculation (single year and two-year scenarios)
- Market share computation (sums to 100 %, zero-total guard)
- Moving average (window validation, single point, immutability)
- Drug and region filtering (case-insensitive, unknown values)
- End-to-end pipeline

## Project Structure

```
prescription-trend-analyzer/
├── src/
│   ├── __init__.py
│   ├── main.py           # Core analysis logic
│   └── data_generator.py # Synthetic data generator
├── tests/
│   └── test_analyzer.py  # pytest test suite
├── demo/
│   └── sample_data.csv   # 20-row demo dataset
├── examples/
│   └── basic_usage.py    # Runnable usage example
├── data/                 # Data directory (gitignored for real data)
├── CHANGELOG.md
├── requirements.txt
└── README.md
```

## Generating Larger Synthetic Data

```python
from src.data_generator import generate_sample

df = generate_sample(n=300, seed=42)
df.to_csv("data/synthetic_300.csv", index=False)
```

## License

MIT License — free to use, modify, and distribute.
