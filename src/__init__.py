"""Package: prescription-trend-analyzer"""

from src.anomaly_detector import AnomalyDetectorError, detect_anomalies
from src.changepoint_detector import (
    ChangePointDetectorError,
    ChangePointResult,
    detect_change_point,
    detect_change_points,
)
from src.mat import (
    MATError,
    mat_growth,
    mat_share,
    moving_annual_total,
)
from src.seasonality import (
    SeasonalityError,
    period_over_period_growth,
    seasonal_decompose_series,
)

__all__ = [
    "AnomalyDetectorError",
    "ChangePointDetectorError",
    "ChangePointResult",
    "MATError",
    "SeasonalityError",
    "detect_anomalies",
    "detect_change_point",
    "detect_change_points",
    "mat_growth",
    "mat_share",
    "moving_annual_total",
    "period_over_period_growth",
    "seasonal_decompose_series",
]
