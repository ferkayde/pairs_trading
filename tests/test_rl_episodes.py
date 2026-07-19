"""Tests for us_experimental.rl_episodes — walk-forward episode builder."""

import numpy as np
import pandas as pd
import pytest

from us_experimental.rl_episodes import (
    Episode,
    build_episodes,
    split_episodes,
    save_episodes,
    load_episodes,
)
from us_experimental.rl_features import STATIC_FEATURES


@pytest.fixture(scope="module")
def synth_prices():
    """6 tickers over ~3 years; three cointegrated pairs sharing common factors."""
    n = 756  # 3 years of business days
    idx = pd.bdate_range("2014-01-01", periods=n)
    rng = np.random.default_rng(42)
    data = {}
    for k in range(3):
        common = np.cumsum(rng.normal(0.0003, 0.01, n))
        eps1 = 0.002 * np.cumsum(rng.normal(0, 0.3, n))
        # mean-reverting idiosyncratic gap for the second leg
        gap = np.zeros(n)
        for t in range(1, n):
            gap[t] = 0.95 * gap[t - 1] + rng.normal(0, 0.004)
        base = 20.0 * (k + 1)
        data[f"A{k}"] = base * np.exp(common + eps1)
        data[f"B{k}"] = base * np.exp(common + eps1 + gap)
    return pd.DataFrame(data, index=idx)


@pytest.fixture(scope="module")
def episodes(synth_prices):
    return build_episodes(
        synth_prices,
        formation_days=252,
        trading_days=126,
        roll_days=63,
        top_n=3,
        verbose=False,
    )


def test_episodes_created(episodes):
    assert len(episodes) > 0
    ep = episodes[0]
    assert isinstance(ep, Episode)


def test_episode_shapes_consistent(episodes):
    for ep in episodes:
        assert len(ep.p1) == len(ep.p2) == len(ep.dates) == ep.length
        assert ep.features.shape == (ep.length, len(STATIC_FEATURES))
        assert ep.features.dtype == np.float64
        assert not np.isnan(ep.features).any()
        assert (ep.p1 > 0).all() and (ep.p2 > 0).all()
        assert ep.locked_sigma > 0


def test_episode_dates_inside_trading_window(episodes):
    for ep in episodes:
        assert ep.dates[0] >= ep.trading_start
        assert ep.dates[-1] <= ep.trading_end
        assert ep.formation_start < ep.trading_start


def test_z_feature_matches_locked_sigma_spread(episodes):
    ep = episodes[0]
    spread = ep.p1 / ep.p1_0 - ep.p2 / ep.p2_0
    np.testing.assert_allclose(
        ep.features[:, 0], spread / ep.locked_sigma, rtol=1e-5
    )


def test_split_boundaries():
    def mk(t_start, t_end):
        n = 5
        return Episode(
            ticker1="A", ticker2="B",
            formation_start=pd.Timestamp(t_start) - pd.Timedelta(days=365),
            trading_start=pd.Timestamp(t_start),
            trading_end=pd.Timestamp(t_end),
            dates=pd.bdate_range(t_start, periods=n),
            p1=np.ones(n), p2=np.ones(n), p1_0=1.0, p2_0=1.0,
            locked_sigma=0.1,
            features=np.zeros((n, 5), dtype=np.float32),
        )

    eps = [
        mk("2014-01-02", "2014-06-30"),   # train
        mk("2014-09-01", "2015-02-28"),   # straddles train/val boundary -> dropped
        mk("2016-01-04", "2016-06-30"),   # val
        mk("2019-09-02", "2020-02-28"),   # straddles val/test boundary -> dropped
        mk("2021-01-04", "2021-06-30"),   # test
    ]
    split = split_episodes(eps, train_end="2015-01-01", val_end="2020-01-01")
    assert [e.trading_start.year for e in split["train"]] == [2014]
    assert [e.trading_start.year for e in split["val"]] == [2016]
    assert [e.trading_start.year for e in split["test"]] == [2021]
    total = len(split["train"]) + len(split["val"]) + len(split["test"])
    assert total == 3  # two straddling windows dropped


def test_save_load_roundtrip(episodes, tmp_path):
    path = tmp_path / "eps.pkl"
    save_episodes(episodes, path)
    loaded = load_episodes(path)
    assert len(loaded) == len(episodes)
    np.testing.assert_array_equal(loaded[0].p1, episodes[0].p1)
    np.testing.assert_array_equal(loaded[0].features, episodes[0].features)
    assert loaded[0].ticker1 == episodes[0].ticker1
