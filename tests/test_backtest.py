# tests/test_backtest.py
import numpy as np
import pandas as pd
import pytest

from src.backtest import simulate_pair_returns


def _make_prices(vals1, vals2, start="2020-01-01"):
    idx = pd.date_range(start, periods=len(vals1), freq="B")
    return pd.Series(vals1, index=idx, dtype=float), pd.Series(vals2, index=idx, dtype=float)


class TestSimulatePairReturns:
    def test_no_signal_no_trades(self):
        # Spread never exceeds 2σ=0.2 (spread stays at 0)
        p1, p2 = _make_prices([100.0] * 20, [100.0] * 20)
        rets, n_trades = simulate_pair_returns(p1, p2, 100.0, 100.0,
                                               locked_sigma=0.1, entry_sigma=2.0)
        assert n_trades == 0
        assert (rets == 0).all()

    def test_one_trade_on_divergence_and_convergence(self):
        # p1 shoots up (spread > 2σ) then returns to base (zero crossing)
        vals1 = [100.0] * 5 + [130.0] * 10 + [100.0] * 5
        vals2 = [100.0] * 20
        p1, p2 = _make_prices(vals1, vals2)
        rets, n_trades = simulate_pair_returns(p1, p2, 100.0, 100.0,
                                               locked_sigma=0.05, entry_sigma=2.0,
                                               commission_bps=0.0)
        assert n_trades == 1

    def test_commission_reduces_returns(self):
        # Same trade with and without commission
        vals1 = [100.0] * 5 + [130.0] * 10 + [100.0] * 5
        vals2 = [100.0] * 20
        p1, p2 = _make_prices(vals1, vals2)
        rets_no_cost, _ = simulate_pair_returns(p1, p2, 100.0, 100.0, 0.05, 2.0, 0.0)
        rets_with_cost, _ = simulate_pair_returns(p1, p2, 100.0, 100.0, 0.05, 2.0, 10.0)
        assert rets_with_cost.sum() < rets_no_cost.sum()

    def test_short_spread_profits_on_convergence(self):
        # p1 outperforms → we short p1 long p2 → profit when p1 falls back
        vals1 = [100.0] * 3 + [130.0] * 5 + [100.0] * 5
        vals2 = [100.0] * 13
        p1, p2 = _make_prices(vals1, vals2)
        rets, n_trades = simulate_pair_returns(p1, p2, 100.0, 100.0,
                                               locked_sigma=0.05, entry_sigma=2.0,
                                               commission_bps=0.0)
        assert n_trades >= 1
        assert rets.sum() > 0  # zero-commission convergence trade should profit

    def test_time_stop_closes_position(self):
        # Spread exceeds threshold but never crosses zero → time-stop
        vals1 = [100.0] * 3 + [150.0] * 17  # stays high, never reverts
        vals2 = [100.0] * 20
        p1, p2 = _make_prices(vals1, vals2)
        rets, n_trades = simulate_pair_returns(p1, p2, 100.0, 100.0,
                                               locked_sigma=0.05, entry_sigma=2.0,
                                               commission_bps=0.0)
        # Time-stop counts as a trade
        assert n_trades >= 1

    def test_returns_series_index_matches_input(self):
        p1, p2 = _make_prices([100.0] * 10, [100.0] * 10)
        rets, _ = simulate_pair_returns(p1, p2, 100.0, 100.0, 0.05)
        assert list(rets.index) == list(p1.index)

    def test_normalization_uses_p0_not_first_trading_price(self):
        # Formation start price was 80; trading starts at 100 (already p1_star=1.25)
        # p2_star stays at 1.0 (p2_0=100, trading price also 100)
        # spread at day 0 = 1.25 - 1.0 = 0.25 > 2*0.05=0.10 → immediate entry
        vals1 = [100.0] * 10  # current prices
        vals2 = [100.0] * 10
        p1, p2 = _make_prices(vals1, vals2)
        rets, n_trades = simulate_pair_returns(
            p1, p2, p1_0=80.0, p2_0=100.0,   # formation-start prices
            locked_sigma=0.05, entry_sigma=2.0, commission_bps=0.0
        )
        # Entry fires immediately (spread = 0.25 > 0.10)
        assert n_trades >= 1
