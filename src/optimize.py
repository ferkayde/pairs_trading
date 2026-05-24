# src/optimize.py
"""Grid search and walk-forward optimization for pairs strategies."""

from __future__ import annotations

import numpy as np
import pandas as pd
from src.backtest import run_backtest
from src.strategies import PairsStrategy

# Extra warm-up bars an indicator may require beyond its nominal ``period``
# before it can emit a value.  The cointegration approach builds a rolling
# z-score on top of an OLS residual, so its effective min-period exceeds
# ``period`` by a handful of bars; this margin keeps walk-forward test slices
# safely in-bounds for both the distance and cointegration indicators.
_WFO_WARMUP_MARGIN = 10


def grid_search(
    pair_df: pd.DataFrame,
    param_grid: dict,
    approach: str = "distance",
    min_trades: int = 3,
) -> pd.DataFrame:
    """Exhaustive grid search over formation periods.

    Parameters
    ----------
    pair_df    : pd.DataFrame  Output of construct_pair().
    param_grid : dict          e.g. {"period": [20, 40, 60, 90, 120, 180]}
    approach   : str           'distance' or 'coint'
    min_trades : int           Skip results with fewer trades (avoids overfitting
                               to lucky single trades, as warned in course notes).

    Returns
    -------
    pd.DataFrame sorted by Sharpe descending.
    """
    records = []
    for period in param_grid.get("period", [60]):
        result = run_backtest(pair_df, PairsStrategy, approach=approach, period=period)
        if result["n_trades"] < min_trades:
            continue
        records.append(
            {
                "period": period,
                "sharpe": result["sharpe"],
                "max_drawdown_pct": result["max_drawdown_pct"],
                "total_return_pct": result["total_return_pct"],
                "n_trades": result["n_trades"],
            }
        )
    return pd.DataFrame(records).sort_values("sharpe", ascending=False) if records else pd.DataFrame()


def walk_forward_optimization(
    pair_df: pd.DataFrame,
    param_grid: dict,
    train_days: int = 252,
    test_days: int = 63,
    approach: str = "distance",
    min_trades: int = 3,
) -> pd.DataFrame:
    """Rolling walk-forward: optimize on train window, evaluate on test window.

    Parameters
    ----------
    pair_df    : pair DataFrame (Close = ratio, Price1, Price2)
    param_grid : {"period": [...]}
    train_days : number of days in the training window (default 252 = 1 year)
    test_days  : number of days in the test window (default 63 = 1 quarter)
    approach   : 'distance' or 'coint'
    min_trades : minimum trades to consider a parameter result valid

    Returns
    -------
    pd.DataFrame with columns: window_start, best_period, test_sharpe,
                                test_return_pct, test_trades
    """
    records = []
    n = len(pair_df)
    start = 0

    while start + train_days + test_days <= n:
        train_slice = pair_df.iloc[start : start + train_days].copy()
        test_slice = pair_df.iloc[start + train_days : start + train_days + test_days].copy()

        # Preserve pair name in slices
        for sl in [train_slice, test_slice]:
            sl.attrs["name"] = pair_df.attrs.get("name", "pair")

        # Find best period on training data.  Only consider periods that fit
        # BOTH the train and the test window: a formation period that does not
        # leave enough warm-up bars for the indicator cannot be evaluated and
        # is useless for walk-forward analysis anyway.  Backtrader's vectorised
        # indicators write past their output buffer and raise IndexError when
        # the indicator's effective min-period exceeds the number of available
        # bars.  The cointegration approach (OLS_TransformationN) needs a few
        # extra bars beyond ``period`` for the rolling residual z-score, so a
        # safety margin is applied to keep both approaches in-bounds.
        window_len = min(len(train_slice), len(test_slice))
        max_period = window_len - _WFO_WARMUP_MARGIN
        best_period = None
        best_sharpe = -np.inf
        for period in param_grid.get("period", [60]):
            if period > max_period:
                continue
            res = run_backtest(train_slice, PairsStrategy, approach=approach, period=period)
            if res["n_trades"] >= min_trades and res["sharpe"] > best_sharpe:
                best_sharpe = res["sharpe"]
                best_period = period

        if best_period is None:
            start += test_days
            continue

        # Evaluate best period on test data
        test_res = run_backtest(test_slice, PairsStrategy, approach=approach, period=best_period)

        records.append(
            {
                "window_start": pair_df.index[start + train_days],
                "best_period": best_period,
                "test_sharpe": test_res["sharpe"],
                "test_return_pct": test_res["total_return_pct"],
                "test_trades": test_res["n_trades"],
            }
        )
        start += test_days

    return pd.DataFrame(records)
