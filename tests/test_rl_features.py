"""Tests for us_experimental.rl_features — RL state features.

The look-ahead audit is the critical test: every rolling feature must use
only past data (spec §6: "Look-ahead bias in features").
"""

import numpy as np
import pandas as pd
import pytest

from us_experimental.rl_features import (
    STATIC_FEATURES,
    DYNAMIC_FEATURES,
    N_FEATURES,
    normalized_spread,
    rolling_half_life,
    build_static_features,
)


def _dates(n, start="2015-01-01"):
    return pd.bdate_range(start, periods=n)


def _ou_spread(n, phi, sigma_eps=0.01, seed=0):
    """AR(1) spread s_t = phi * s_{t-1} + eps_t."""
    rng = np.random.default_rng(seed)
    s = np.zeros(n)
    for t in range(1, n):
        s[t] = phi * s[t - 1] + rng.normal(0, sigma_eps)
    return pd.Series(s, index=_dates(n))


def test_feature_name_constants():
    assert STATIC_FEATURES == ["z", "z_lag1", "z_lag5", "vol_ratio", "half_life"]
    assert DYNAMIC_FEATURES == ["position", "days_in_pos", "unreal_pnl", "days_left"]
    assert N_FEATURES == 9


def test_normalized_spread_definition():
    idx = _dates(3)
    p1 = pd.Series([10.0, 11.0, 12.0], index=idx)
    p2 = pd.Series([20.0, 20.0, 22.0], index=idx)
    spread = normalized_spread(p1, p2, 10.0, 20.0)
    expected = p1 / 10.0 - p2 / 20.0
    pd.testing.assert_series_equal(spread, expected)


def test_z_is_spread_over_locked_sigma():
    n = 400
    idx = _dates(n)
    p1 = pd.Series(np.linspace(10, 12, n), index=idx)
    p2 = pd.Series(np.linspace(20, 21, n), index=idx)
    sigma = 0.05
    trading_start = idx[300]
    feats = build_static_features(p1, p2, 10.0, 20.0, sigma, trading_start)
    spread = p1 / 10.0 - p2 / 20.0
    expected_z = (spread / sigma).loc[trading_start:]
    np.testing.assert_allclose(feats["z"].values, expected_z.values, atol=1e-12)
    # lags are shifted copies of z
    np.testing.assert_allclose(
        feats["z_lag1"].values[1:], feats["z"].values[:-1], atol=1e-12
    )


def test_half_life_recovers_ar1_coefficient():
    phi = 0.9  # true half-life = -ln2/ln(0.9) ~ 6.58 bars
    spread = _ou_spread(4000, phi, seed=1)
    hl = rolling_half_life(spread, window=1000)
    true_hl = -np.log(2) / np.log(phi)
    est = hl.iloc[-1]
    assert np.isfinite(est)
    assert abs(est - true_hl) / true_hl < 0.35  # noisy estimator, loose tolerance


def test_half_life_much_larger_for_random_walk():
    """A random walk (phi~1) must show a far larger half-life than a strongly
    mean-reverting AR(1). (The rolling estimator has Dickey-Fuller downward
    bias, so a random walk gives large-but-finite values, not +inf.)"""
    rng = np.random.default_rng(2)
    rw = pd.Series(np.cumsum(rng.normal(0, 1, 3000)), index=_dates(3000))
    hl_rw = rolling_half_life(rw, window=500).iloc[-500:]
    hl_mr = rolling_half_life(_ou_spread(3000, 0.9, seed=2), window=500).iloc[-500:]
    assert hl_rw.median() > 4 * hl_mr.median()


def test_no_lookahead_in_features():
    """Mutating prices AFTER bar t must leave features at bars <= t unchanged."""
    n = 500
    idx = _dates(n)
    rng = np.random.default_rng(3)
    p1 = pd.Series(10 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), index=idx)
    p2 = pd.Series(20 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), index=idx)
    trading_start = idx[300]
    cut = 400  # mutate bars after this position

    base = build_static_features(p1, p2, p1.iloc[0], p2.iloc[0], 0.05, trading_start)

    p1_mut = p1.copy()
    p2_mut = p2.copy()
    p1_mut.iloc[cut:] = p1_mut.iloc[cut:] * 3.0
    p2_mut.iloc[cut:] = p2_mut.iloc[cut:] * 0.5
    mut = build_static_features(p1_mut, p2_mut, p1.iloc[0], p2.iloc[0], 0.05, trading_start)

    # All bars strictly before the mutation point must be identical
    np.testing.assert_allclose(
        base.loc[: idx[cut - 1]].values, mut.loc[: idx[cut - 1]].values, atol=1e-12
    )


def test_features_nan_free_with_warm_history():
    n = 400
    idx = _dates(n)
    rng = np.random.default_rng(4)
    p1 = pd.Series(10 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), index=idx)
    p2 = pd.Series(20 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), index=idx)
    feats = build_static_features(p1, p2, p1.iloc[0], p2.iloc[0], 0.05, idx[252])
    assert not feats.isna().any().any()
    assert list(feats.columns) == STATIC_FEATURES
    assert len(feats) == n - 252
