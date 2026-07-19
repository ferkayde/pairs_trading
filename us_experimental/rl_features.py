"""
rl_features.py — state features for the RL pairs-trading environment (US data).

Every feature is backward-looking only: rolling statistics at bar t use bars
<= t, and the only cross-window quantity is the locked formation sigma, which is
frozen before trading starts (mirrors GGR's locked-σ convention).

Feature vector layout (N_FEATURES = 9):
    Static (precomputed per bar from prices)   — STATIC_FEATURES
    Dynamic (maintained by the environment)     — DYNAMIC_FEATURES
"""

from __future__ import annotations

import numpy as np
import pandas as pd

STATIC_FEATURES = ["z", "z_lag1", "z_lag5", "vol_ratio", "half_life"]
DYNAMIC_FEATURES = ["position", "days_in_pos", "unreal_pnl", "days_left"]
N_FEATURES = len(STATIC_FEATURES) + len(DYNAMIC_FEATURES)


def normalized_spread(
    p1: pd.Series, p2: pd.Series, p1_0: float, p2_0: float
) -> pd.Series:
    """GGR normalized-price spread: P1/P1_0 - P2/P2_0."""
    return p1 / p1_0 - p2 / p2_0


def rolling_half_life(spread: pd.Series, window: int = 60) -> pd.Series:
    """Rolling OU/AR(1) half-life of mean reversion, in bars.

    Fits Δs_t = a + b·s_{t-1} over a trailing window via the moment estimator
    b = cov(Δs_t, s_{t-1}) / var(s_{t-1}), so each bar uses only past data.
    With phi = 1 + b, half-life = -ln2 / ln(phi) when 0 < phi < 1; otherwise
    the spread shows no mean reversion in the window and +inf is returned.
    """
    s_lag = spread.shift(1)
    ds = spread - s_lag
    b = ds.rolling(window).cov(s_lag) / s_lag.rolling(window).var()
    phi = 1.0 + b
    with np.errstate(divide="ignore", invalid="ignore"):
        hl = np.where(
            (phi > 0) & (phi < 1), -np.log(2.0) / np.log(phi.to_numpy()), np.inf
        )
    return pd.Series(hl, index=spread.index)


def build_static_features(
    p1: pd.Series,
    p2: pd.Series,
    p1_0: float,
    p2_0: float,
    locked_sigma: float,
    trading_start,
    vol_window: int = 20,
    hl_window: int = 60,
    max_half_life: float = 126.0,
) -> pd.DataFrame:
    """Per-bar static state features for one pair's trading window.

    p1/p2 must include the formation history so rolling statistics are warm by
    the first trading bar; only rows >= trading_start are returned. All values
    are normalized with formation-window quantities only (locked_sigma) or are
    scale-free, and the result is NaN-free.

    Columns (STATIC_FEATURES order):
        z          spread / locked_sigma — the static rule's signal
        z_lag1     z at t-1
        z_lag5     z at t-5
        vol_ratio  20d rolling std of daily spread changes / locked_sigma
        half_life  rolling OU half-life, clipped to max_half_life and divided
                   by it (1.0 = no detectable mean reversion)
    """
    spread = normalized_spread(p1, p2, p1_0, p2_0)
    z = spread / locked_sigma
    vol_ratio = spread.diff().rolling(vol_window).std() / locked_sigma
    hl = rolling_half_life(spread, hl_window)
    hl_norm = hl.clip(upper=max_half_life).fillna(max_half_life) / max_half_life
    feats = pd.DataFrame(
        {
            "z": z,
            "z_lag1": z.shift(1),
            "z_lag5": z.shift(5),
            "vol_ratio": vol_ratio,
            "half_life": hl_norm,
        }
    )
    return feats.loc[trading_start:].fillna(0.0)
