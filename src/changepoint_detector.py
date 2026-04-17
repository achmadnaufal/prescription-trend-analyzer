"""
Trend change-point detection for prescription volume time series.

Identifies the most likely point in time at which the underlying trend of a
prescription volume series shifts.  This is useful for spotting events such
as a generic-drug launch, a label change, a new clinical guideline, or a
supply disruption.

The detector implements a **piecewise-linear regression** scan that, for
every candidate split index ``k``, fits two independent linear segments
(``[0, k)`` and ``[k, n)``) and records the resulting sum of squared
residuals (SSR).  The split that minimises the combined SSR is reported as
the best candidate.  An additional **F-test** style improvement ratio is
computed against a single-line baseline so callers can decide whether the
change-point is statistically meaningful.

All public functions are pure (inputs are never mutated) and return new
:class:`~pandas.DataFrame` objects or plain dataclass-style dicts.

Author: github.com/achmadnaufal
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

__all__ = [
    "ChangePointDetectorError",
    "ChangePointResult",
    "detect_change_points",
    "detect_change_point",
]


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------


class ChangePointDetectorError(ValueError):
    """Raised when the change-point detector receives invalid inputs.

    Inherits from :class:`ValueError` so callers that already catch
    ``ValueError`` continue to work without changes.
    """


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChangePointResult:
    """Outcome of a single-series change-point search.

    Attributes:
        index: Position (0-based) of the first observation belonging to the
            *post-change* segment.  ``None`` when no valid split exists.
        improvement_ratio: ``1 - (SSR_split / SSR_single)``.  Values close to
            ``0`` mean the split barely helps; values close to ``1`` mean the
            split explains nearly all of the variance left over by a single
            line.  ``None`` when no valid split exists.
        slope_before: OLS slope of the pre-change segment.
        slope_after: OLS slope of the post-change segment.
        intercept_before: OLS intercept of the pre-change segment.
        intercept_after: OLS intercept of the post-change segment.
        is_significant: ``True`` when *improvement_ratio* exceeds the caller's
            *min_improvement* threshold.
    """

    index: Optional[int]
    improvement_ratio: Optional[float]
    slope_before: Optional[float]
    slope_after: Optional[float]
    intercept_before: Optional[float]
    intercept_after: Optional[float]
    is_significant: bool

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain ``dict`` view of the result (JSON-friendly)."""
        return asdict(self)


# ---------------------------------------------------------------------------
# Internal helpers (pure functions)
# ---------------------------------------------------------------------------


_NULL_RESULT = ChangePointResult(
    index=None,
    improvement_ratio=None,
    slope_before=None,
    slope_after=None,
    intercept_before=None,
    intercept_after=None,
    is_significant=False,
)


def _segment_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Fit ``y = slope * x + intercept`` and return slope, intercept, SSR.

    Args:
        x: 1-D array of independent variable values.
        y: 1-D array of dependent variable values, same length as *x*.

    Returns:
        Tuple ``(slope, intercept, ssr)``.  When fewer than two points are
        supplied the slope/intercept are ``0.0`` and SSR is ``0.0``.
    """
    if len(x) < 2:
        return 0.0, float(y[0]) if len(y) else 0.0, 0.0
    slope, intercept = np.polyfit(x, y, 1)
    fitted = slope * x + intercept
    ssr = float(np.sum((y - fitted) ** 2))
    return float(slope), float(intercept), ssr


def _validate_series(values: np.ndarray, min_segment: int) -> None:
    """Raise :class:`ChangePointDetectorError` for unusable input arrays.

    Args:
        values: Numeric series under inspection.
        min_segment: Minimum points required on each side of the split.
    """
    if values.size == 0:
        raise ChangePointDetectorError(
            "Input series is empty. Provide at least one numeric observation."
        )
    if min_segment < 2:
        raise ChangePointDetectorError(
            f"'min_segment' must be >= 2, got {min_segment}."
        )
    if not np.all(np.isfinite(values)):
        raise ChangePointDetectorError(
            "Input series contains NaN or infinite values. "
            "Drop or impute them before calling the detector."
        )


# ---------------------------------------------------------------------------
# Public API — single series
# ---------------------------------------------------------------------------


def detect_change_point(
    values: "pd.Series | np.ndarray | list[float]",
    min_segment: int = 3,
    min_improvement: float = 0.10,
) -> ChangePointResult:
    """Locate the single most likely trend change-point in *values*.

    The algorithm scans every legal split index ``k`` such that both
    segments contain at least *min_segment* observations, fits an OLS line
    to each segment, and selects the ``k`` that minimises the total sum of
    squared residuals.  The returned :class:`ChangePointResult` also
    contains an *improvement_ratio* against a single-line baseline; the
    *is_significant* flag is ``True`` iff the ratio exceeds
    *min_improvement*.

    Args:
        values: Chronologically ordered numeric observations.  Accepts a
            :class:`pandas.Series`, :class:`numpy.ndarray`, or plain ``list``.
        min_segment: Minimum number of points required on each side of the
            split.  Must be >= 2.  Defaults to ``3``.
        min_improvement: SSR-improvement ratio required for the result's
            ``is_significant`` flag to be ``True``.  Must be in ``[0, 1)``.
            Defaults to ``0.10`` (split must explain 10% more variance than
            a single line).

    Returns:
        :class:`ChangePointResult` with the best split index, slopes,
        intercepts, improvement ratio, and significance flag.  When the
        series is too short for any valid split, every numeric attribute is
        ``None`` and *is_significant* is ``False``.

    Raises:
        ChangePointDetectorError: If *values* is empty, contains non-finite
            entries, *min_segment* is < 2, or *min_improvement* is outside
            ``[0, 1)``.

    Example::

        from src.changepoint_detector import detect_change_point

        # Trend shift at index 6: flat then steep climb
        series = [10, 11, 9, 10, 11, 10, 20, 30, 40, 50, 60]
        result = detect_change_point(series, min_segment=3)
        print(result.index, result.improvement_ratio, result.is_significant)
    """
    if not 0.0 <= min_improvement < 1.0:
        raise ChangePointDetectorError(
            f"'min_improvement' must be in [0, 1), got {min_improvement}."
        )

    arr = np.asarray(values, dtype=float)
    _validate_series(arr, min_segment)

    n = arr.size
    # Need at least 2 * min_segment points to perform any split
    if n < 2 * min_segment:
        return _NULL_RESULT

    x_full = np.arange(n, dtype=float)
    _, _, ssr_single = _segment_fit(x_full, arr)

    # Numerical guard: when the single-line fit is "perfect" within
    # floating-point tolerance the improvement ratio is mathematically
    # meaningless (any tiny SSR_split could divide to a large ratio).
    # Compare against the variance scale of the data.
    scale = float(np.var(arr)) * n
    near_perfect = ssr_single <= max(1e-9, 1e-9 * scale)

    best_ssr = np.inf
    best_k = -1
    best_left: tuple[float, float, float] = (0.0, 0.0, 0.0)
    best_right: tuple[float, float, float] = (0.0, 0.0, 0.0)

    for k in range(min_segment, n - min_segment + 1):
        left = _segment_fit(x_full[:k], arr[:k])
        right = _segment_fit(x_full[k:], arr[k:])
        total_ssr = left[2] + right[2]
        if total_ssr < best_ssr:
            best_ssr = total_ssr
            best_k = k
            best_left = left
            best_right = right

    if best_k < 0:
        return _NULL_RESULT

    # Improvement ratio against the single-line baseline
    if near_perfect:
        # Perfect (or numerically perfect) single line — no improvement.
        improvement = 0.0
    else:
        improvement = 1.0 - (best_ssr / ssr_single)

    # Clamp tiny negatives caused by floating-point noise
    improvement = max(0.0, float(improvement))

    return ChangePointResult(
        index=int(best_k),
        improvement_ratio=round(improvement, 6),
        slope_before=round(best_left[0], 6),
        slope_after=round(best_right[0], 6),
        intercept_before=round(best_left[1], 6),
        intercept_after=round(best_right[1], 6),
        is_significant=improvement >= min_improvement,
    )


# ---------------------------------------------------------------------------
# Public API — multi-group DataFrame helper
# ---------------------------------------------------------------------------


def detect_change_points(
    df: pd.DataFrame,
    value_col: str = "prescriptions_count",
    group_col: str = "drug_name",
    date_col: Optional[str] = None,
    min_segment: int = 3,
    min_improvement: float = 0.10,
) -> pd.DataFrame:
    """Run :func:`detect_change_point` on every group in *df*.

    Each group (typically one drug) is analysed independently.  When
    *date_col* is supplied, rows within each group are first sorted by that
    column to ensure chronological order; the original *df* is **never**
    mutated.

    Args:
        df: Input DataFrame.  Must contain *value_col*.  When *group_col*
            is present each group is processed separately; otherwise the
            entire DataFrame is treated as a single series.
        value_col: Numeric column containing the time-series values.
            Defaults to ``"prescriptions_count"``.
        group_col: Column used to split data into independent series.
            Defaults to ``"drug_name"``.  Pass an empty string to disable
            grouping.
        date_col: Optional date/time column used to sort each group before
            detection.  When ``None``, the existing row order is assumed
            chronological.
        min_segment: Minimum number of points required on each side of the
            split.  Forwarded to :func:`detect_change_point`.
        min_improvement: SSR-improvement ratio required for significance.
            Forwarded to :func:`detect_change_point`.

    Returns:
        New :class:`~pandas.DataFrame` with one row per group containing the
        columns ``group``, ``index``, ``improvement_ratio``,
        ``slope_before``, ``slope_after``, ``intercept_before``,
        ``intercept_after``, and ``is_significant``.  When grouping is
        disabled the ``group`` column holds the literal string ``"_all_"``.

    Raises:
        ChangePointDetectorError: If *df* is empty or *value_col* is absent.

    Example::

        from src.changepoint_detector import detect_change_points

        # df has columns: drug_name, date, prescriptions_count
        report = detect_change_points(
            df,
            value_col="prescriptions_count",
            group_col="drug_name",
            date_col="date",
            min_segment=3,
        )
        print(report[report["is_significant"]])
    """
    if df.empty:
        raise ChangePointDetectorError(
            "Input DataFrame is empty. Provide at least one row."
        )
    if value_col not in df.columns:
        raise ChangePointDetectorError(
            f"Column '{value_col}' not found. "
            f"Available columns: {list(df.columns)}."
        )
    if date_col is not None and date_col not in df.columns:
        raise ChangePointDetectorError(
            f"Column '{date_col}' not found. "
            f"Available columns: {list(df.columns)}."
        )

    # Work on an independent copy
    work = df.copy()
    use_groups = bool(group_col) and group_col in work.columns

    if use_groups:
        groups: list[tuple[Any, pd.DataFrame]] = list(
            work.groupby(group_col, sort=False)
        )
    else:
        groups = [("_all_", work)]

    rows: List[Dict[str, Any]] = []
    for name, sub in groups:
        ordered = sub.sort_values(date_col) if date_col is not None else sub
        # Coerce non-numeric values to NaN; let validator surface NaN errors
        numeric = pd.to_numeric(ordered[value_col], errors="coerce")
        # Drop NaN tails / gaps before evaluation so callers with sparse
        # series still get a usable answer.
        cleaned = numeric.dropna().to_numpy(dtype=float)
        if cleaned.size == 0:
            rows.append({"group": name, **_NULL_RESULT.to_dict()})
            continue
        try:
            result = detect_change_point(
                cleaned,
                min_segment=min_segment,
                min_improvement=min_improvement,
            )
        except ChangePointDetectorError:
            # Defensive: cleaned array was validated, but propagate-safe.
            result = _NULL_RESULT
        rows.append({"group": name, **result.to_dict()})

    return pd.DataFrame(rows)
