# tests/test_indicators.py
import pandas as pd
import numpy as np
import backtrader as bt
import pytest
from src.indicators import DistanceZScore, CointZScore
from src.backtest import PairFeed


def _run_indicator(indicator_cls, pair_df, **kwargs):
    """Helper: run a bt.Indicator on a PairFeed and collect zscore output."""

    class _Capture(bt.Strategy):
        def __init__(self):
            self.ind = indicator_cls(self.data, **kwargs)
            self.zscores = []

        def next(self):
            self.zscores.append(self.ind.zscore[0])

    cerebro = bt.Cerebro()
    cerebro.adddata(PairFeed(dataname=pair_df))
    cerebro.addstrategy(_Capture)
    result = cerebro.run()
    return result[0].zscores


def _make_pair_df(n=300, seed=42):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2018-01-01", periods=n, freq="B")
    common = np.cumsum(rng.normal(0, 1, n))
    p1 = pd.Series(100 + common + rng.normal(0, 0.5, n), index=idx)
    p2 = pd.Series(100 + common + rng.normal(0, 0.5, n), index=idx)
    df = pd.DataFrame({"Close": p1 / p2, "Price1": p1, "Price2": p2}, index=idx)
    df.attrs["name"] = "A_B"
    return df


class TestDistanceZScore:
    def test_output_has_values(self):
        pair_df = _make_pair_df()
        zs = _run_indicator(DistanceZScore, pair_df, period=60)
        assert len(zs) > 0

    def test_zscore_near_zero_mean(self):
        pair_df = _make_pair_df(n=500)
        zs = _run_indicator(DistanceZScore, pair_df, period=60)
        assert abs(np.mean(zs)) < 0.5

    def test_zscore_near_unit_std(self):
        pair_df = _make_pair_df(n=500)
        zs = _run_indicator(DistanceZScore, pair_df, period=60)
        assert 0.5 < np.std(zs) < 2.0


class TestCointZScore:
    def test_output_has_values(self):
        pair_df = _make_pair_df(n=300)
        zs = _run_indicator(CointZScore, pair_df, period=60)
        assert len(zs) > 0

    def test_zscore_finite(self):
        pair_df = _make_pair_df(n=300)
        zs = _run_indicator(CointZScore, pair_df, period=60)
        assert all(np.isfinite(v) for v in zs)
