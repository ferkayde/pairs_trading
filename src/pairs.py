# src/pairs.py
"""
Pair construction, cointegration testing, and pair screening.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant


def construct_pair(
    price1: pd.Series,
    price2: pd.Series,
    name1: str,
    name2: str,
) -> pd.DataFrame:
    """Construct pair DataFrame with ratio (Close) and individual prices.

    Parameters
    ----------
    price1, price2 : pd.Series  Aligned price series (same index).
    name1, name2   : str         Ticker names.

    Returns
    -------
    pd.DataFrame with columns Close (=price1/price2), Price1, Price2.
    The attrs dict carries 'name' = 'NAME1_NAME2'.
    """
    pair_df = pd.DataFrame(
        {
            "Close": price1 / price2,
            "Price1": price1,
            "Price2": price2,
        },
        index=price1.index,
    )
    pair_df.attrs["name"] = f"{name1}_{name2}"
    return pair_df


def test_cointegration(
    s1: pd.Series,
    s2: pd.Series,
    pvalue_threshold: float = 0.05,
) -> dict:
    """Engle-Granger cointegration test (both directions; take lower p-value).

    Returns
    -------
    dict with keys: t_stat, p_value, is_cointegrated
    """
    score_12, pval_12, _ = coint(s1, s2)
    score_21, pval_21, _ = coint(s2, s1)

    if pval_12 <= pval_21:
        t_stat, p_value = score_12, pval_12
    else:
        t_stat, p_value = score_21, pval_21

    return {
        "t_stat": t_stat,
        "p_value": p_value,
        "is_cointegrated": p_value < pvalue_threshold,
    }


# Prevent pytest from treating test_cointegration as a test function.
test_cointegration.__test__ = False  # type: ignore[attr-defined]


def half_life(spread: pd.Series) -> float:
    """Ornstein-Uhlenbeck half-life of mean reversion (in bars).

    Regresses delta_spread on lagged spread level.
    half_life = -ln(2) / AR_coefficient
    """
    lag = spread.shift(1).dropna()
    delta = spread.diff().dropna()
    common_idx = lag.index.intersection(delta.index)
    reg = OLS(delta.loc[common_idx], add_constant(lag.loc[common_idx])).fit()
    theta = reg.params.iloc[1]  # AR coefficient on lagged level
    if theta >= 0:
        return np.inf  # non-mean-reverting
    return -np.log(2) / theta


def screen_pairs(
    prices: pd.DataFrame,
    candidate_pairs: list,
    pvalue_threshold: float = 0.05,
    min_half_life: float = 0.0,
    max_half_life: float = 120.0,
) -> pd.DataFrame:
    """Screen candidate pairs for cointegration and tradeable half-life.

    Parameters
    ----------
    prices           : pd.DataFrame  Columns are ticker names.
    candidate_pairs  : list of (ticker1, ticker2) tuples.
    pvalue_threshold : float  Cointegration p-value cutoff.
    min_half_life    : float  Min half-life in days (avoid noise).
    max_half_life    : float  Max half-life in days (avoid too slow).

    Returns
    -------
    pd.DataFrame with columns: pair, ticker1, ticker2, p_value, half_life_days
    Only rows that pass both screens are included.
    """
    records = []
    for t1, t2 in candidate_pairs:
        if t1 not in prices.columns or t2 not in prices.columns:
            continue
        s1 = prices[t1].dropna()
        s2 = prices[t2].dropna()
        common = s1.index.intersection(s2.index)
        if len(common) < 252:  # need at least 1 year of data
            continue
        s1, s2 = s1.loc[common], s2.loc[common]

        coint_result = test_cointegration(s1, s2, pvalue_threshold)
        if not coint_result["is_cointegrated"]:
            continue

        spread = s1 / s2
        hl = half_life(spread)
        if not (min_half_life <= hl <= max_half_life):
            continue

        records.append(
            {
                "pair": f"{t1}_{t2}",
                "ticker1": t1,
                "ticker2": t2,
                "p_value": round(coint_result["p_value"], 4),
                "half_life_days": round(hl, 1),
            }
        )

    return pd.DataFrame(records)
