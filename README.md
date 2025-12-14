# Prescription Trend Analyzer

Prescription volume trend analysis, forecasting, and visualization

## Features
- Data ingestion from CSV/Excel input files
- Automated analysis and KPI calculation
- Summary statistics and trend reporting
- Sample data generator for testing and development

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from src.main import RxTrendAnalyzer

analyzer = RxTrendAnalyzer()
df = analyzer.load_data("data/sample.csv")
result = analyzer.analyze(df)
print(result)
```

## Data Format

Expected CSV columns: `product, molecule, month, rx_units, nrx_units, trx_units, yoy_growth_pct`

## Project Structure

```
prescription-trend-analyzer/
├── src/
│   ├── main.py          # Core analysis logic
│   └── data_generator.py # Sample data generator
├── data/                # Data directory (gitignored for real data)
├── examples/            # Usage examples
├── requirements.txt
└── README.md
```

## License

MIT License — free to use, modify, and distribute.
