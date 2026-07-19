"""Tests for us_experimental.rl_agents — including the Milestone-1 parity gate.

The parity test is the project's foundation (spec §7): the static GGR rule
executed INSIDE the RL environment must reproduce
src.backtest.simulate_pair_returns to numerical precision. If this holds,
every downstream RL-vs-static comparison is trustworthy.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.backtest import simulate_pair_returns
from us_experimental.rl_agents import FlatAgent, StaticRuleAgent
from us_experimental.rl_env import ACTION_FLAT, ACTION_LONG, ACTION_SHORT, PairsTradingEnv
from us_experimental.rl_episodes import build_episodes

DATA = Path(__file__).resolve().parent.parent / "data" / "prices.csv"
COMMISSION_BPS = 10.0


def rollout_pnl(episode, agent, commission_bps=COMMISSION_BPS):
    """Run agent inside the env; return the per-bar net PnL series and trades."""
    env = PairsTradingEnv(
        [episode], commission_bps=commission_bps, reward_scale=1.0,
        sampling="sequential",
    )
    obs, _ = env.reset(options={"episode_index": 0})
    pnl = np.zeros(episode.length)
    trades = []
    for t in range(episode.length):
        action, _ = agent.predict(obs)
        obs, r, term, trunc, info = env.step(action)
        pnl[t] = info["pnl"]
        if term:
            trades = info["trades"]
            break
    return pd.Series(pnl, index=episode.dates), trades


def simulate_reference(episode, commission_bps=COMMISSION_BPS, entry_sigma=2.0):
    p1 = pd.Series(episode.p1, index=episode.dates)
    p2 = pd.Series(episode.p2, index=episode.dates)
    rets, n_trades = simulate_pair_returns(
        p1, p2, episode.p1_0, episode.p2_0, episode.locked_sigma,
        entry_sigma=entry_sigma, commission_bps=commission_bps,
        wait_one_day=False,
    )
    return rets, n_trades


# --------------------------------------------------------------------- agents
def test_static_agent_signals():
    agent = StaticRuleAgent(entry_z=2.0)

    def obs(z, pos):
        o = np.zeros(9, dtype=np.float32)
        o[0] = z
        o[5] = pos
        return o

    assert agent.predict(obs(2.5, 0))[0] == ACTION_SHORT
    assert agent.predict(obs(-2.5, 0))[0] == ACTION_LONG
    assert agent.predict(obs(1.0, 0))[0] == ACTION_FLAT
    # long position: hold below zero, close at crossing
    assert agent.predict(obs(-0.5, 1))[0] == ACTION_LONG
    assert agent.predict(obs(0.0, 1))[0] == ACTION_FLAT
    # short position: hold above zero, close at crossing
    assert agent.predict(obs(0.5, -1))[0] == ACTION_SHORT
    assert agent.predict(obs(-0.01, -1))[0] == ACTION_FLAT


def test_flat_agent_never_trades():
    agent = FlatAgent()
    assert agent.predict(np.ones(9, dtype=np.float32))[0] == ACTION_FLAT


# ------------------------------------------------------- parity on synthetic
@pytest.fixture(scope="module")
def synth_episodes():
    n = 756
    idx = pd.bdate_range("2014-01-01", periods=n)
    rng = np.random.default_rng(7)
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


def test_parity_on_synthetic_episodes(synth_episodes):
    assert len(synth_episodes) > 0
    agent = StaticRuleAgent(entry_z=2.0)
    for ep in synth_episodes:
        env_pnl, trades = rollout_pnl(ep, agent)
        ref_pnl, ref_trades = simulate_reference(ep)
        np.testing.assert_allclose(
            env_pnl.values, ref_pnl.values, atol=1e-12,
            err_msg=f"PnL mismatch for {ep.ticker1}/{ep.ticker2} "
                    f"window {ep.trading_start.date()}",
        )
        assert len(trades) == ref_trades


# ------------------------------------------------------- parity on real data
@pytest.mark.skipif(not DATA.exists(), reason="US price data not available")
def test_parity_on_real_data_window():
    """Milestone-1 gate on one real 2005 window of data/prices.csv."""
    prices = pd.read_csv(DATA, index_col=0, parse_dates=True)
    sub = prices.loc["2004-01-01":"2006-06-30"]
    episodes = build_episodes(
        sub, roll_days=10_000, top_n=20, verbose=False,  # single window
    )
    assert len(episodes) >= 10
    agent = StaticRuleAgent(entry_z=2.0)
    n_with_trades = 0
    for ep in episodes:
        env_pnl, trades = rollout_pnl(ep, agent)
        ref_pnl, ref_trades = simulate_reference(ep)
        np.testing.assert_allclose(
            env_pnl.values, ref_pnl.values, atol=1e-10,
            err_msg=f"PnL mismatch for {ep.ticker1}/{ep.ticker2}",
        )
        assert len(trades) == ref_trades
        n_with_trades += ref_trades > 0
    assert n_with_trades > 0  # the window must actually contain trades
