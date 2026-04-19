"""
Moving Annual Total (MAT) computation for prescription volume series.

In pharmaceutical market analysis a **Moving Annual Total** (MAT) — also
known as a *rolling annual sum* or *12-month trailing total* — is the sum
of the last 12 months of prescription volume at every observation.  It
smooths out monthly seasonality and gives an always-up-to-date view of
"what does this brand look like on a yearly basis?"

Two companion metrics are also provided:

* **MAT growth** – year-over-year change of the MAT itself
  (``(MAT_t - MAT_{t-12}) / MAT_{t-12} * 100``).  This is the standard
  "MAT growth %" reported in IQVIA / IMS pharma dashboards.
* **MAT share** – each drug's MAT as a share of the total MAT across
  all drugs in the same month.  This is the market-share equivalent of
  MAT and is often the metric senior commercial leaders actually track.

All helpers follow the project's immutability contract: the caller's
DataFrame is never mutated; a new DataFrame is returned.

Author: github.com/achmadnaufal
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

__all__ = [
    "MATError",
    "moving_annual_total",
    "mat_growth",
    "mat_share",
]


class MATError(ValueError):
    """Raised when Moving-Annual-Total input validation fails."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_common(
    df: pd.DataFrame,
    value_col: str,
    group_col: Optional[str],
    date_col: Optional[str],
    window: int,
    min_periods: int,
) -> None:
    """Validate the arguments shared by every public helper.

    Args:
        df: Input DataFrame.
        value_col: Numeric column name expected in *df*.
        group_col: Optional grouping column.  When provided, must exist.
        date_col: Optional date column.  When provided, must exist.
        window: Rolling window size (months).  Must be a positive integer.
        min_periods: Minimum number of non-NaN observations in the window
            required to emit a value.  Must satisfy
            ``1 <= min_periods <= window``.

    Raises:
        MATError: When any argument fails validation.
    """
    if not isinstance(df, pd.DataFrame):
        raise MATError("'df' must be a pandas DataFrame.")
    if df.empty:
        raise MATError("'df' is empty; cannot compute Moving Annual Total.")
    if value_col not in df.columns:
        raise MATError(f"'value_col' {value_col!r} not in DataFrame.")
    if group_col is not None and group_col not in df.columns:
        raise MATError(f"'group_col' {group_col!r} not in DataFrame.")
    if date_col is not None and date_col not in df.columns:
        raise MATError(f"'date_col' {date_col!r} not in DataFrame.")
    if not isinstance(window, (int, np.integer)) or window < 1:
        raise MATError(f"'window' must be a positive integer; got {window!r}.")
    if (
        not isinstance(min_periods, (int, np.integer))
        or min_periods < 1
        or min_periods > int(window)
    ):
        raise MATError(
            f"'min_periods' must satisfy 1 <= min_periods <= window; "
            f"got min_periods={min_periods!r}, window={window!r}."
        )


def _sorted_working_copy(
    df: pd.DataFrame,
    group_col: Optional[str],
    date_col: Optional[str],
) -> pd.DataFrame:
    """Return a sorted, re-indexed copy without mutating the input.

    Uses a stable sort (mergesort) so that rows with identical sort keys
    preserve their original order — important for deterministic output
    when the caller supplies ties on ``date_col``.

    Args:
        df: Input DataFrame.
        group_col: Optional grouping column (sorted first when present).
        date_col: Optional date column (sorted second when present).

    Returns:
        A new, row-index-reset DataFrame.
    """
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
    return working


# ---------------------------------------------------------------------------
# Moving Annual Total
# ---------------------------------------------------------------------------


def moving_annual_total(
    df: pd.DataFrame,
    value_col: str,
    group_col: Optional[str] = None,
    date_col: Optional[str] = None,
    window: int = 12,
    min_periods: Optional[int] = None,
    output_col: Optional[str] = None,
) -> pd.DataFrame:
    """Compute the rolling-sum Moving Annual Total (MAT) per group.

    The MAT at observation *t* is defined as the sum of *value_col* over
    the window of the last ``window`` observations (inclusive of *t*).
    With monthly data and ``window=12`` this is the classic 12-month
    trailing total used throughout the pharma industry.

    When ``group_col`` is provided, the rolling sum is computed
    **independently within each group** so volumes from different
    drugs / brands / regions never bleed into each other.

    Args:
        df: Input DataFrame.  Not mutated — a new copy is returned.
        value_col: Numeric column to accumulate (e.g.
            ``"rx_volume"`` or ``"prescriptions_count"``).
        group_col: Optional grouping column (e.g. ``"product"``).
            When ``None``, the full DataFrame is treated as a single
            series.
        date_col: Optional date column used to sort each group in
            chronological order before the rolling sum.  When ``None``
            the existing row order is used.
        window: Window length in observations.  Defaults to ``12``
            (i.e. a true Moving Annual Total when data is monthly).
            Must be a positive integer.
        min_periods: Minimum number of non-NaN observations inside the
            window required to emit a sum.  Defaults to ``window``
            (i.e. only fully-filled windows receive a value; earlier
            rows receive ``NaN``).  Must satisfy
            ``1 <= min_periods <= window``.
        output_col: Name of the added column.  Defaults to
            ``f"{value_col}_mat{window}"``.

    Returns:
        A new DataFrame equal to ``df`` (sorted by ``group_col``,
        ``date_col`` when provided) plus the MAT column.

    Raises:
        MATError: If any argument fails validation.
    """
    resolved_min_periods = int(window) if min_periods is None else int(min_periods)
    _validate_common(df, value_col, group_col, date_col, window, resolved_min_periods)

    out_name = output_col or f"{value_col}_mat{int(window)}"
    working = _sorted_working_copy(df, group_col, date_col)

    values = working[value_col].astype(float)
    if group_col is not None:
        rolled = (
            values.groupby(working[group_col], sort=False)
            .rolling(window=int(window), min_periods=resolved_min_periods)
            .sum()
            .reset_index(level=0, drop=True)
        )
    else:
        rolled = values.rolling(
            window=int(window), min_periods=resolved_min_periods
        ).sum()

    working[out_name] = rolled.astype(float)
    return working


# ---------------------------------------------------------------------------
# MAT growth (year-over-year change of MAT)
# ---------------------------------------------------------------------------


def mat_growth(
    df: pd.DataFrame,
    value_col: str,
    group_col: Optional[str] = None,
    date_col: Optional[str] = None,
    window: int = 12,
    output_col: Optional[str] = None,
) -> pd.DataFrame:
    """Compute MAT together with its year-over-year percentage growth.

    First calls :func:`moving_annual_total` to build the MAT column, then
    appends a second column equal to ``(MAT_t - MAT_{t-window}) / MAT_{t-window} * 100``
    (rows where the lagged MAT is zero or missing receive ``NaN``).

    With the default ``window=12`` this is the canonical "MAT growth %"
    reported on pharma commercial dashboards.

    Args:
        df: Input DataFrame.  Not mutated — a new copy is returned.
        value_col: Numeric column to accumulate.
        group_col: Optional grouping column.  Growth is computed
            independently per group.
        date_col: Optional date column used for chronological sorting.
        window: Window length and growth lag.  Defaults to ``12``.
        output_col: Name of the growth column.  Defaults to
            ``f"{value_col}_mat{window}_growth_pct"``.  The MAT column
            itself is always written as ``f"{value_col}_mat{window}"``.

    Returns:
        A new DataFrame containing both the MAT column and the growth
        column.

    Raises:
        MATError: If any argument fails validation.
    """
    mat_df = moving_annual_total(
        df,
        value_col=value_col,
        group_col=group_col,
        date_col=date_col,
        window=window,
        min_periods=int(window),  # strict windows for a clean YoY lag
    )
    mat_col = f"{value_col}_mat{int(window)}"
    growth_col = output_col or f"{value_col}_mat{int(window)}_growth_pct"

    if group_col is not None:
        prev = mat_df.groupby(group_col, sort=False)[mat_col].shift(int(window))
    else:
        prev = mat_df[mat_col].shift(int(window))

    current = mat_df[mat_col].astype(float)
    prev = prev.astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        growth = np.where(
            (prev == 0) | prev.isna() | current.isna(),
            np.nan,
            (current - prev) / prev * 100.0,
        )
    mat_df[growth_col] = pd.Series(growth, index=mat_df.index)
    return mat_df


# ---------------------------------------------------------------------------
# MAT share (market-share equivalent on the MAT basis)
# ---------------------------------------------------------------------------


def mat_share(
    df: pd.DataFrame,
    value_col: str,
    group_col: str,
    date_col: str,
    window: int = 12,
    output_col: Optional[str] = None,
) -> pd.DataFrame:
    """Compute each group's MAT share of the total MAT per date.

    For every row the MAT is first computed per group (via
    :func:`moving_annual_total`).  Then, within each ``date_col`` value,
    each group's MAT is divided by the sum of MATs for that date and
    multiplied by 100.  Dates where the total MAT is zero or all MATs
    are ``NaN`` receive ``NaN``.

    Args:
        df: Input DataFrame.  Not mutated — a new copy is returned.
        value_col: Numeric column to accumulate.
        group_col: **Required** grouping column (e.g. ``"product"``)
            — MAT share is meaningless without a group.
        date_col: **Required** date column used to align MAT values
            across groups.
        window: Window length.  Defaults to ``12``.
        output_col: Name of the share column.  Defaults to
            ``f"{value_col}_mat{window}_share_pct"``.  The MAT column
            itself is written as ``f"{value_col}_mat{window}"``.

    Returns:
        A new DataFrame containing both the MAT column and the share
        column.  Rows sorted by ``(group_col, date_col)``.

    Raises:
        MATError: If any argument fails validation.  In particular,
            both ``group_col`` and ``date_col`` are required here.
    """
    if group_col is None:
        raise MATError("'group_col' is required for mat_share.")
    if date_col is None:
        raise MATError("'date_col' is required for mat_share.")

    mat_df = moving_annual_total(
        df,
        value_col=value_col,
        group_col=group_col,
        date_col=date_col,
        window=window,
        min_periods=int(window),
    )
    mat_col = f"{value_col}_mat{int(window)}"
    share_col = output_col or f"{value_col}_mat{int(window)}_share_pct"

    totals = mat_df.groupby(date_col)[mat_col].transform("sum")
    with np.errstate(divide="ignore", invalid="ignore"):
        share = np.where(
            (totals == 0) | totals.isna() | mat_df[mat_col].isna(),
            np.nan,
            mat_df[mat_col] / totals * 100.0,
        )
    mat_df[share_col] = pd.Series(share, index=mat_df.index)
    return mat_df
