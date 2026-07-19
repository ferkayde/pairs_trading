"""Tests for us_experimental.rolling_retrain — yearly walk-forward retraining."""

import numpy as np
import pandas as pd
import pytest

from us_experimental.rl_episodes import Episode
from us_experimental.rl_features import build_static_features
from us_experimental.rolling_retrain import rolling_year_splits


def make_ep(t_start, t_end, seed=0, n=30):
    idx = pd.bdate_range(t_start, periods=n)
    rng = np.random.default_rng(seed)
    p1 = pd.Series(10 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), index=idx)
    p2 = pd.Series(20 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), index=idx)
    feats = build_static_features(p1, p2, 10.0, 20.0, 0.05, idx[0])
    return Episode(
        ticker1="A", ticker2="B",
        formation_start=pd.Timestamp(t_start) - pd.Timedelta(days=365),
        trading_start=pd.Timestamp(t_start), trading_end=pd.Timestamp(t_end),
        dates=idx, p1=p1.to_numpy(), p2=p2.to_numpy(),
        p1_0=10.0, p2_0=20.0, locked_sigma=0.05,
        features=feats.to_numpy(np.float64),
    )


@pytest.fixture
def episodes():
    eps = []
    k = 0
    # two windows per year, 2014-2021; one straddler over each year boundary
    for year in range(2014, 2022):
        for month in (2, 7):
            k += 1
            eps.append(make_ep(f"{year}-{month:02d}-01", f"{year}-{month + 5:02d}-28", seed=k))
        k += 1
        eps.append(make_ep(f"{year}-10-01", f"{year + 1}-03-28", seed=k))  # straddles
    return eps


def test_rolling_splits_no_leakage(episodes):
    splits = rolling_year_splits(episodes, first_year=2019, last_year=2021, val_years=2)
    assert [s["year"] for s in splits] == [2019, 2020, 2021]
    for s in splits:
        y0 = pd.Timestamp(f"{s['year']}-01-01")
        v0 = pd.Timestamp(f"{s['year'] - 2}-01-01")
        assert s["train"], "train must not be empty"
        assert s["val"], "val must not be empty"
        assert s["test"], "test must not be empty"
        # train fully resolved before validation era
        assert max(e.trading_end for e in s["train"]) < v0
        # validation fully resolved before the test year
        assert min(e.trading_start for e in s["val"]) >= v0
        assert max(e.trading_end for e in s["val"]) < y0
        # test windows start inside the year
        for e in s["test"]:
            assert e.trading_start.year == s["year"]


def test_rolling_splits_test_slices_are_disjoint_and_cover(episodes):
    splits = rolling_year_splits(episodes, first_year=2019, last_year=2021)
    seen = set()
    for s in splits:
        for e in s["test"]:
            key = (e.trading_start, e.trading_end, id(e))
            assert key not in seen
            seen.add(key)
    expected = [e for e in episodes if 2019 <= e.trading_start.year <= 2021]
    assert len(seen) == len(expected)


def test_rolling_splits_skips_empty_years(episodes):
    # no episodes start in 2030
    splits = rolling_year_splits(episodes, first_year=2021, last_year=2030)
    assert [s["year"] for s in splits] == [2021]
