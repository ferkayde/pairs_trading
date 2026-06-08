# src/pairs.py
"""
GGR (2006) pair formation: normalization, exhaustive SSD ranking, liquidity filter, locked sigma.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def normalize_prices(prices_window: pd.DataFrame) -> pd.DataFrame:
    """P*_{i,t} = P_{i,t} / P_{i,0}.

    Divides each column by its value on the first day of the formation window
    so every series starts at exactly 1.0. This is the GGR cumulative-return
    index normalization (§2 of the paper).
    """
    first_prices = prices_window.apply(
        lambda col: col.dropna().iloc[0] if col.dropna().size > 0 else np.nan
    )
    return prices_window.div(first_prices, axis=1)


def liquidity_filter(
    prices: pd.DataFrame,
    formation_start,
    formation_end,
) -> list:
    """Return tickers with complete price histories during the formation window.

    GGR (2006) §2: "We restrict the sample to stocks with complete return
    histories during the formation period." The criterion is price completeness
    only — any ticker with a single missing price observation is excluded.
    The paper does not apply a volume screen.

    Parameters
    ----------
    prices              : full-history price DataFrame; columns = ticker names.
    formation_start/end : inclusive date bounds of the 12-month window.

    Returns
    -------
    list of tickers that have no NaN prices in the formation window.
    """
    px_w = prices.loc[formation_start:formation_end]
    return [t for t in px_w.columns if px_w[t].notna().all()]


def activity_filter(
    prices_window: pd.DataFrame,
    tickers: list,
    max_zero_frac: float = 0.30,
) -> list:
    """Remove tickers whose daily returns are zero too often (dormant stocks).

    Yahoo Finance repeats the last known close for days when a stock does not
    trade, producing zero daily returns rather than NaN. CRSP — used by GGR —
    records NaN on non-trading days, so dormant stocks were naturally excluded
    from their universe. This filter replicates that effect.

    A ticker is excluded if the fraction of zero daily-return days in the
    formation window exceeds max_zero_frac (default 30 %).

    Parameters
    ----------
    prices_window : price DataFrame for the formation window.
    tickers       : list to screen (typically output of liquidity_filter).
    max_zero_frac : maximum allowed fraction of zero-return days (default 0.30).

    Returns
    -------
    list of tickers that pass the activity screen.
    """
    result = []
    for t in tickers:
        if t not in prices_window.columns:
            continue
        returns = prices_window[t].pct_change().dropna()
        if len(returns) == 0:
            continue
        if (returns == 0).mean() <= max_zero_frac:
            result.append(t)
    return result


def compute_ssd(norm_prices: pd.DataFrame) -> pd.DataFrame:
    """Compute SSD_{A,B} = Σ_t (P*_{A,t} − P*_{B,t})² for all N(N−1)/2 pairs.

    Vectorized via matrix algebra so it scales to large universes (N=200+) without
    Python loops. NaN values are treated as 0 contributions to the dot product,
    which is consistent with nansum semantics.

    Returns pd.DataFrame with columns [ticker1, ticker2, ssd], sorted ascending.
    """
    tickers = norm_prices.columns.tolist()
    arr = norm_prices.values.astype(float)
    arr_filled = np.where(np.isnan(arr), 0.0, arr)

    gram = arr_filled.T @ arr_filled                  # (N, N)
    sq_sum = np.nansum(arr ** 2, axis=0)              # (N,)
    ssd_matrix = sq_sum[:, None] + sq_sum[None, :] - 2 * gram  # (N, N)

    rows, cols = np.triu_indices(len(tickers), k=1)
    ssd_vals = ssd_matrix[rows, cols]
    order = np.argsort(ssd_vals)
    return pd.DataFrame({
        "ticker1": [tickers[rows[i]] for i in order],
        "ticker2": [tickers[cols[i]] for i in order],
        "ssd":     ssd_vals[order],
    }).reset_index(drop=True)


def select_top_pairs(ssd_df: pd.DataFrame, n: int = 20, offset: int = 0) -> list:
    """Return n pairs starting at rank `offset` as (ticker1, ticker2, ssd) tuples.

    offset=0  → ranks 1–20   (Top-20, GGR primary result)
    offset=100 → ranks 101–120 (GGR control group: no longer profitable)
    """
    section = ssd_df.iloc[offset: offset + n]
    return list(zip(section["ticker1"], section["ticker2"], section["ssd"]))


def compute_locked_sigma(norm_prices: pd.DataFrame, pairs: list) -> dict:
    """Compute σ = std(P*_A − P*_B) for each pair over the formation window.

    This value is frozen at the end of formation and used unchanged as the
    divergence threshold throughout the 6-month trading period.

    Returns dict keyed by (ticker1, ticker2) → float σ.
    """
    sigmas = {}
    for item in pairs:
        t1, t2 = item[0], item[1]
        if t1 not in norm_prices.columns or t2 not in norm_prices.columns:
            continue
        spread = norm_prices[t1] - norm_prices[t2]
        sigmas[(t1, t2)] = float(spread.std())
    return sigmas
