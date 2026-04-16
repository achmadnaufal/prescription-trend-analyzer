"""
Anomaly detection for prescription volume time series.

Flags unusual week-over-week (or month-over-month) prescription volume
changes using two complementary methods:

* **Robust z-score** — based on median and median absolute deviation (MAD),
  which is resistant to the influence of the anomalies themselves.
* **IQR fence** — flags values outside ``Q1 - k*IQR`` or ``Q3 + k*IQR``
  using Tukey's method.

All public functions are pure (inputs are never mutated) and return new
:class:`~pandas.DataFrame` objects.

Author: github.com/achmadnaufal
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

__all__ = ["detect_anomalies", "AnomalyDetectorError"]

# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------


class AnomalyDetectorError(ValueError):
    """Raised when the detector receives invalid inputs.

    Inherits from :class:`ValueError` so callers that already catch
    ``ValueError`` continue to work without changes.
    """


# ---------------------------------------------------------------------------
# Internal helpers (pure functions)
# ---------------------------------------------------------------------------


def _robust_z_scores(values: np.ndarray) -> np.ndarray:
    """Compute robust z-scores using median and MAD.

    The formula is ``0.6745 * (x - median) / MAD``.  The constant 0.6745 is
    the 75th percentile of the standard normal, making the statistic
    consistent with the classical z-score for Gaussian data.  When MAD == 0
    (all values identical) every z-score is defined as 0.

    Args:
        values: 1-D numeric array with at least one element.

    Returns:
        1-D float array of the same length as *values*.
    """
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    if mad == 0.0:
        return np.zeros_like(values, dtype=float)
    return 0.6745 * (values - median) / mad


def _iqr_flags(values: np.ndarray, k: float) -> np.ndarray:
    """Return a boolean mask for values outside Tukey's IQR fences.

    Args:
        values: 1-D numeric array.
        k: Fence multiplier.  Standard choices are ``1.5`` (mild outliers)
            or ``3.0`` (extreme outliers).

    Returns:
        Boolean array — ``True`` where *values* is outside the fences.
    """
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    return (values < lower) | (values > upper)


def _build_rationale(
    value: float,
    median: float,
    mad: float,
    z: float,
    iqr_flagged: bool,
    z_threshold: float,
) -> str:
    """Produce a human-readable explanation for a flagged anomaly.

    Args:
        value: The observed prescription volume.
        median: Series median.
        mad: Median absolute deviation of the series.
        z: Robust z-score for this observation.
        iqr_flagged: Whether the IQR method also flagged this row.
        z_threshold: The z-score threshold used.

    Returns:
        Single-sentence rationale string.
    """
    direction = "above" if value > median else "below"
    methods: list[str] = []
    if abs(z) >= z_threshold:
        methods.append(f"robust z-score={z:.2f} (threshold ±{z_threshold})")
    if iqr_flagged:
        methods.append("IQR fence")
    method_str = " and ".join(methods) if methods else "combined method"
    return (
        f"Volume {value:,.0f} is {direction} median {median:,.0f} "
        f"(MAD={mad:,.1f}); flagged by {method_str}."
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_anomalies(
    df: pd.DataFrame,
    value_col: str = "prescriptions_count",
    group_col: str = "drug_name",
    method: Literal["zscore", "iqr", "both"] = "both",
    z_threshold: float = 3.0,
    iqr_k: float = 1.5,
    min_periods: int = 4,
) -> pd.DataFrame:
    """Flag anomalous prescription volume observations per drug (or per group).

    Each group (typically one drug) is analysed independently so that a drug
    with naturally high volumes does not suppress anomaly flags for drugs with
    low volumes.

    Two detection methods are supported:

    * ``"zscore"`` — robust z-score >= *z_threshold* (absolute value).
    * ``"iqr"`` — Tukey IQR fence with multiplier *iqr_k*.
    * ``"both"`` — a row is flagged when **either** method triggers.

    Groups with fewer than *min_periods* observations are returned as-is with
    ``is_anomaly=False`` and ``anomaly_rationale=None`` (insufficient data for
    reliable statistics).

    Args:
        df: Input DataFrame.  Must contain *value_col*.  If *group_col*
            is present each group is processed separately; otherwise the
            entire DataFrame is treated as one series.
        value_col: Numeric column containing prescription volumes.
            Defaults to ``"prescriptions_count"``.
        group_col: Column used to split the data into independent series.
            Defaults to ``"drug_name"``.  Pass an empty string ``""`` to
            disable grouping and analyse the whole DataFrame as one series.
        method: Detection algorithm — ``"zscore"``, ``"iqr"``, or ``"both"``.
            Defaults to ``"both"``.
        z_threshold: Minimum absolute robust z-score to flag a row when the
            z-score method is active.  Must be > 0.  Defaults to ``3.0``.
        iqr_k: IQR fence multiplier.  Must be > 0.  Defaults to ``1.5``.
        min_periods: Minimum number of rows in a group for anomaly detection
            to run.  Groups below this threshold are returned un-flagged.
            Must be >= 2.  Defaults to ``4``.

    Returns:
        New :class:`~pandas.DataFrame` (the caller's *df* is **not** mutated)
        with three additional columns:

        ``is_anomaly`` (*bool*)
            ``True`` when the row is considered anomalous.

        ``anomaly_score`` (*float*)
            Absolute robust z-score for the observation (``NaN`` when the
            group had fewer than *min_periods* rows).

        ``anomaly_rationale`` (*str | None*)
            Human-readable explanation for flagged rows; ``None`` otherwise.

    Raises:
        AnomalyDetectorError: If *df* is empty, *value_col* is absent, the
            *method* argument is invalid, *z_threshold* <= 0, *iqr_k* <= 0,
            or *min_periods* < 2.

    Example::

        import pandas as pd
        from src.anomaly_detector import detect_anomalies

        df = pd.DataFrame({
            "drug_name": ["Lipitor"] * 12,
            "prescriptions_count": [
                1000, 1020, 980, 1010, 990, 1030,
                1005, 995, 1015, 1025, 975, 5000,  # last is a spike
            ],
        })
        result = detect_anomalies(df, min_periods=4)
        print(result[result["is_anomaly"]][["prescriptions_count", "anomaly_rationale"]])
    """
    # --- Input validation ---------------------------------------------------
    if df.empty:
        raise AnomalyDetectorError(
            "Input DataFrame is empty. Provide at least one row."
        )
    if value_col not in df.columns:
        raise AnomalyDetectorError(
            f"Column '{value_col}' not found. "
            f"Available columns: {list(df.columns)}."
        )
    if method not in ("zscore", "iqr", "both"):
        raise AnomalyDetectorError(
            f"Invalid method '{method}'. Choose from 'zscore', 'iqr', or 'both'."
        )
    if z_threshold <= 0:
        raise AnomalyDetectorError(
            f"'z_threshold' must be > 0, got {z_threshold}."
        )
    if iqr_k <= 0:
        raise AnomalyDetectorError(
            f"'iqr_k' must be > 0, got {iqr_k}."
        )
    if min_periods < 2:
        raise AnomalyDetectorError(
            f"'min_periods' must be >= 2, got {min_periods}."
        )

    # --- Determine grouping -------------------------------------------------
    use_groups = bool(group_col) and group_col in df.columns

    # Work on an independent copy so the caller's DataFrame is never mutated
    result = df.copy()
    result = result.assign(
        is_anomaly=False,
        anomaly_score=np.nan,
        anomaly_rationale=None,
    )

    # --- Process groups (or the whole frame) --------------------------------
    if use_groups:
        groups = result.groupby(group_col, sort=False)
    else:
        # Treat the entire DataFrame as a single pseudo-group
        groups = [(None, result)]  # type: ignore[assignment]

    processed_parts: list[pd.DataFrame] = []

    for _, group_df in groups:
        part = _flag_group(
            group_df=group_df,
            value_col=value_col,
            method=method,
            z_threshold=z_threshold,
            iqr_k=iqr_k,
            min_periods=min_periods,
        )
        processed_parts.append(part)

    if processed_parts:
        result = pd.concat(processed_parts).sort_index()

    # Ensure boolean dtype (concat may widen to object)
    result = result.assign(is_anomaly=result["is_anomaly"].astype(bool))

    return result


# ---------------------------------------------------------------------------
# Private per-group logic
# ---------------------------------------------------------------------------


def _flag_group(
    group_df: pd.DataFrame,
    value_col: str,
    method: str,
    z_threshold: float,
    iqr_k: float,
    min_periods: int,
) -> pd.DataFrame:
    """Apply anomaly detection to a single group DataFrame.

    Args:
        group_df: Slice of the full DataFrame for one group.
        value_col: Column with numeric volumes.
        method: One of ``"zscore"``, ``"iqr"``, ``"both"``.
        z_threshold: Robust z-score threshold.
        iqr_k: IQR fence multiplier.
        min_periods: Minimum rows for detection to activate.

    Returns:
        A new DataFrame with ``is_anomaly``, ``anomaly_score``, and
        ``anomaly_rationale`` columns filled in for this group.
    """
    part = group_df.copy()

    if len(part) < min_periods:
        # Not enough data — leave defaults (False / NaN / None)
        return part

    values = pd.to_numeric(part[value_col], errors="coerce").to_numpy(dtype=float)
    finite_mask = np.isfinite(values)

    if finite_mask.sum() < min_periods:
        return part

    z_scores = _robust_z_scores(values)
    median = float(np.median(values[finite_mask]))
    mad = float(np.median(np.abs(values[finite_mask] - median)))
    iqr_flag_arr = _iqr_flags(values, iqr_k)

    # Build flags based on chosen method
    if method == "zscore":
        flag_arr = np.abs(z_scores) >= z_threshold
    elif method == "iqr":
        flag_arr = iqr_flag_arr
    else:  # "both"
        flag_arr = (np.abs(z_scores) >= z_threshold) | iqr_flag_arr

    # Assign scores and rationales back to the copy
    part = part.assign(anomaly_score=np.round(np.abs(z_scores), 4))

    rationales: list[str | None] = []
    for i, (flagged, val, z, iqr_f) in enumerate(
        zip(flag_arr, values, z_scores, iqr_flag_arr)
    ):
        if flagged and np.isfinite(val):
            rationales.append(
                _build_rationale(
                    value=val,
                    median=median,
                    mad=mad,
                    z=z,
                    iqr_flagged=bool(iqr_f),
                    z_threshold=z_threshold,
                )
            )
        else:
            rationales.append(None)

    part = part.assign(
        is_anomaly=flag_arr,
        anomaly_rationale=rationales,
    )
    return part
