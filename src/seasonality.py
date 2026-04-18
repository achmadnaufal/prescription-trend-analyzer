"""
Seasonality and period-over-period helpers for prescription volume series.

This module adds two capabilities that complement the core
:class:`RxTrendAnalyzer` pipeline:

* :func:`seasonal_decompose_series` - a thin, validated wrapper around
  ``statsmodels.tsa.seasonal_decompose`` that returns a tidy pandas
  DataFrame with ``observed``, ``trend``, ``seasonal``, and ``resid``
  columns, with graceful degradation when the series is too short.
* :func:`period_over_period_growth` - per-group lag-based growth
  computation (MoM, QoQ, YoY, or any integer lag) with configurable
  denominator guards and a choice of absolute or percentage output.

Both helpers follow the project's immutability contract: the caller's
DataFrame is never mutated; a new DataFrame is returned.

Author: github.com/achmadnaufal
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

__all__ = [
    "SeasonalityError",
    "period_over_period_growth",
    "seasonal_decompose_series",
]


class SeasonalityError(ValueError):
    """Raised when seasonality / PoP input validation fails."""


# ---------------------------------------------------------------------------
# Seasonal decomposition
# ---------------------------------------------------------------------------


def seasonal_decompose_series(
    series: pd.Series,
    period: int,
    model: str = "additive",
) -> pd.DataFrame:
    """Decompose a univariate series into trend, seasonal, and residual parts.

    Wraps :func:`statsmodels.tsa.seasonal.seasonal_decompose` with input
    validation tailored for prescription volume data.  The function is
    deterministic and returns a *new* DataFrame that shares its index with
    ``series``.

    Args:
        series: Chronologically ordered 1-D series of observations.  The
            index is preserved on the output.  NaN values are forward-
            then back-filled internally so the decomposition does not
            explode on isolated gaps, but the returned ``observed``
            column reflects the filled values.
        period: Length of one seasonal cycle in observations (e.g. ``12``
            for monthly data with yearly seasonality, ``4`` for quarterly,
            ``7`` for weekly-over-daily).  Must satisfy ``period >= 2``
            and ``len(series) >= 2 * period``.
        model: ``"additive"`` (default) or ``"multiplicative"``.  For
            multiplicative decomposition all observations must be
            strictly positive.

    Returns:
        A DataFrame indexed like ``series`` with columns
        ``["observed", "trend", "seasonal", "resid"]``.  ``trend`` and
        ``resid`` contain NaN at the edges (this is expected behaviour
        of a centred moving-average decomposition).

    Raises:
        SeasonalityError: If ``series`` is not a pandas Series, the
            series is too short, ``period`` is invalid, or
            multiplicative decomposition is requested on a series that
            contains non-positive values.
    """
    if not isinstance(series, pd.Series):
        raise SeasonalityError("'series' must be a pandas Series.")
    if not isinstance(period, (int, np.integer)) or period < 2:
        raise SeasonalityError(
            f"'period' must be an integer >= 2; got {period!r}."
        )
    if model not in {"additive", "multiplicative"}:
        raise SeasonalityError(
            f"'model' must be 'additive' or 'multiplicative'; got {model!r}."
        )
    if len(series) < 2 * period:
        raise SeasonalityError(
            f"Need at least 2*period={2 * period} observations to decompose; "
            f"got {len(series)}."
        )

    # Fill isolated NaNs so the decomposition is well-defined; do not
    # mutate the caller's series.
    filled = series.astype(float).ffill().bfill()

    if filled.isna().any():
        raise SeasonalityError(
            "'series' is entirely NaN after forward/back fill; cannot decompose."
        )
    if model == "multiplicative" and (filled <= 0).any():
        raise SeasonalityError(
            "Multiplicative decomposition requires strictly positive values."
        )

    # Import lazily so the module stays importable even if statsmodels
    # is unavailable at import time (tests may stub it out).
    from statsmodels.tsa.seasonal import seasonal_decompose

    result = seasonal_decompose(
        filled.values,
        model=model,
        period=int(period),
        extrapolate_trend=0,
    )

    return pd.DataFrame(
        {
            "observed": filled.values,
            "trend": result.trend,
            "seasonal": result.seasonal,
            "resid": result.resid,
        },
        index=series.index,
    )


# ---------------------------------------------------------------------------
# Period-over-period growth
# ---------------------------------------------------------------------------


_LAG_ALIASES = {
    "mom": 1,   # month-over-month (assuming monthly cadence)
    "qoq": 3,   # quarter-over-quarter (monthly cadence) / 1 if quarterly
    "yoy": 12,  # year-over-year (monthly cadence)
    "wow": 1,   # week-over-week (assuming weekly cadence)
}


def _resolve_lag(lag: "int | str") -> int:
    """Translate a string alias (e.g. ``'yoy'``) into an integer lag."""
    if isinstance(lag, str):
        key = lag.lower()
        if key not in _LAG_ALIASES:
            raise SeasonalityError(
                f"Unknown lag alias {lag!r}. "
                f"Valid aliases: {sorted(_LAG_ALIASES)}."
            )
        return _LAG_ALIASES[key]
    if isinstance(lag, (int, np.integer)) and lag >= 1:
        return int(lag)
    raise SeasonalityError(
        f"'lag' must be a positive integer or a known alias; got {lag!r}."
    )


def period_over_period_growth(
    df: pd.DataFrame,
    value_col: str,
    group_col: Optional[str] = None,
    date_col: Optional[str] = None,
    lag: "int | str" = 1,
    output: str = "pct",
    output_col: Optional[str] = None,
) -> pd.DataFrame:
    """Compute period-over-period growth as a new column.

    Supports month-over-month (lag=1 on monthly data), quarter-over-quarter,
    year-over-year, or any integer lag.  When ``group_col`` is provided, the
    lag is computed independently within each group (e.g. per drug), which
    is the typical pharma use-case.

    Args:
        df: Input DataFrame.  Not mutated - a new copy is returned.
        value_col: Column containing the numeric series (e.g.
            ``"prescription_volume"``).
        group_col: Optional grouping column.  If omitted, the entire
            DataFrame is treated as a single series.
        date_col: Optional column used to sort each group chronologically
            before computing the lag.  If omitted, row order is used.
        lag: Positive integer or alias (``"mom"``, ``"qoq"``, ``"yoy"``,
            ``"wow"``).  See :data:`_LAG_ALIASES`.
        output: ``"pct"`` for percentage change (e.g. ``5.0`` = +5%),
            ``"abs"`` for absolute delta (current - previous), or
            ``"ratio"`` for a multiplicative ratio (current / previous).
        output_col: Name of the new column.  Defaults to
            ``f"{value_col}_{output}_lag{lag}"``.

    Returns:
        A new DataFrame equal to ``df`` plus the growth column.  Rows
        without a valid lag (the first ``lag`` rows in each group, or
        rows where the previous value is zero for ``"pct"``/``"ratio"``)
        receive NaN.

    Raises:
        SeasonalityError: If required columns are missing, ``df`` is
            empty, or any argument fails validation.
    """
    if not isinstance(df, pd.DataFrame):
        raise SeasonalityError("'df' must be a pandas DataFrame.")
    if df.empty:
        raise SeasonalityError("'df' is empty; cannot compute growth.")
    if value_col not in df.columns:
        raise SeasonalityError(f"'value_col' {value_col!r} not in DataFrame.")
    if group_col is not None and group_col not in df.columns:
        raise SeasonalityError(f"'group_col' {group_col!r} not in DataFrame.")
    if date_col is not None and date_col not in df.columns:
        raise SeasonalityError(f"'date_col' {date_col!r} not in DataFrame.")
    if output not in {"pct", "abs", "ratio"}:
        raise SeasonalityError(
            f"'output' must be one of 'pct', 'abs', 'ratio'; got {output!r}."
        )

    lag_int = _resolve_lag(lag)
    out_name = output_col or f"{value_col}_{output}_lag{lag_int}"

    # Work on a copy to honour immutability.
    working = df.copy()

    sort_cols = []
    if group_col is not None:
        sort_cols.append(group_col)
    if date_col is not None:
        sort_cols.append(date_col)
    if sort_cols:
        working = working.sort_values(sort_cols, kind="mergesort").reset_index(
            drop=True
        )
    else:
        working = working.reset_index(drop=True)

    values = working[value_col].astype(float)
    if group_col is not None:
        prev = working.groupby(group_col, sort=False)[value_col].shift(lag_int)
    else:
        prev = values.shift(lag_int)
    prev = prev.astype(float)

    if output == "abs":
        growth = values - prev
    elif output == "ratio":
        with np.errstate(divide="ignore", invalid="ignore"):
            growth = np.where(prev == 0, np.nan, values / prev)
        growth = pd.Series(growth, index=working.index)
    else:  # pct
        with np.errstate(divide="ignore", invalid="ignore"):
            growth = np.where(
                prev == 0, np.nan, (values - prev) / prev * 100.0
            )
        growth = pd.Series(growth, index=working.index)

    working[out_name] = growth
    return working
