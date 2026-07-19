"""Tests for us_experimental.rl_evaluate — rollouts, aggregation, significance."""

import numpy as np
import pandas as pd
import pytest

from us_experimental.rl_agents import FlatAgent, StaticRuleAgent
from us_experimental.rl_episodes import build_episodes
from us_experimental.rl_evaluate import (
    aggregate_daily,
    bootstrap_sharpe_diff,
    evaluate_policy,
    rollout_episode,
    subperiod_table,
    tune_static_entry,
)


@pytest.fixture(scope="module")
def synth_episodes():
    n = 756
    idx = pd.bdate_range("2014-01-01", periods=n)
    rng = np.random.default_rng(11)
    data = {}
    for k in range(4):
        common = np.cumsum(rng.normal(0.0003, 0.012, n))
        gap = np.zeros(n)
        for t in range(1, n):
            gap[t] = 0.93 * gap[t - 1] + rng.normal(0, 0.006)
        base = 15.0 * (k + 1)
        data[f"A{k}"] = base * np.exp(common)
        data[f"B{k}"] = base * np.exp(common + gap)
    prices = pd.DataFrame(data, index=idx)
    return build_episodes(prices, roll_days=126, top_n=4, verbose=False)


def test_flat_agent_zero_everything(synth_episodes):
    res = evaluate_policy(synth_episodes, FlatAgent())
    assert res["n_trades"] == 0
    assert res["sharpe"] == 0.0
    assert res["total_return_pct"] == pytest.approx(0.0)
    assert (res["daily"] == 0).all()


def test_static_agent_produces_trades(synth_episodes):
    res = evaluate_policy(synth_episodes, StaticRuleAgent(entry_z=2.0))
    assert res["n_trades"] > 0
    assert len(res["trade_log"]) == res["n_trades"]
    assert res["avg_holding_days"] > 0
    assert 0.0 <= res["win_rate"] <= 1.0
    assert len(res["daily"]) > 0
    assert res["equity"].iloc[0] > 0


def test_rollout_episode_matches_reference(synth_episodes):
    from src.backtest import simulate_pair_returns

    ep = synth_episodes[0]
    rets, trades = rollout_episode(ep, StaticRuleAgent(2.0), commission_bps=10.0)
    ref, _ = simulate_pair_returns(
        pd.Series(ep.p1, index=ep.dates), pd.Series(ep.p2, index=ep.dates),
        ep.p1_0, ep.p2_0, ep.locked_sigma, entry_sigma=2.0, commission_bps=10.0,
    )
    np.testing.assert_allclose(rets.values, ref.values, atol=1e-12)


def test_aggregate_daily_equal_weight_mean():
    idx1 = pd.bdate_range("2020-01-01", periods=3)
    idx2 = pd.bdate_range("2020-01-02", periods=3)  # overlaps on 2 days
    s1 = pd.Series([0.01, 0.02, 0.03], index=idx1)
    s2 = pd.Series([0.10, 0.20, 0.30], index=idx2)
    agg = aggregate_daily([s1, s2])
    # day1: only s1 -> 0.01 ; day2: mean(0.02, 0.10) ; day4: only s2 -> 0.30
    assert agg.loc[idx1[0]] == pytest.approx(0.01)
    assert agg.loc[idx1[1]] == pytest.approx(0.06)
    assert agg.loc[idx2[-1]] == pytest.approx(0.30)


def test_tune_static_entry_returns_grid_member(synth_episodes):
    grid = (1.5, 2.0, 2.5)
    best_z, table = tune_static_entry(synth_episodes, grid=grid)
    assert best_z in grid
    assert len(table) == len(grid)
    assert {"entry_z", "sharpe", "n_trades"} <= set(table.columns)
    # best_z must attain the max sharpe in the table
    assert table.loc[table["entry_z"] == best_z, "sharpe"].iloc[0] == table["sharpe"].max()


def test_bootstrap_identical_series_no_difference():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2020-01-01", periods=500)
    a = pd.Series(rng.normal(0.0005, 0.01, 500), index=idx)
    res = bootstrap_sharpe_diff(a, a.copy(), n_boot=200, seed=1)
    assert res["diff_observed"] == pytest.approx(0.0, abs=1e-12)
    assert res["ci_low"] == pytest.approx(0.0, abs=1e-9)
    assert res["ci_high"] == pytest.approx(0.0, abs=1e-9)


def test_bootstrap_detects_clear_difference():
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2020-01-01", periods=750)
    good = pd.Series(rng.normal(0.002, 0.01, 750), index=idx)
    bad = pd.Series(rng.normal(-0.001, 0.01, 750), index=idx)
    res = bootstrap_sharpe_diff(good, bad, n_boot=300, seed=2)
    assert res["diff_observed"] > 0
    assert res["p_value"] < 0.05
    assert res["ci_low"] > 0


def test_subperiod_table():
    idx = pd.bdate_range("2020-01-01", "2023-12-29")
    rng = np.random.default_rng(3)
    daily = pd.Series(rng.normal(0.0002, 0.005, len(idx)), index=idx)
    table = subperiod_table(daily, ["2022-01-01"])
    assert len(table) == 2
    assert {"start", "end", "sharpe", "total_return_pct", "max_drawdown"} <= set(table.columns)
