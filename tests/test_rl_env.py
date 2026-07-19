"""Tests for us_experimental.rl_env — PairsTradingEnv accounting and API."""

import numpy as np
import pandas as pd
import pytest

from us_experimental.rl_episodes import Episode
from us_experimental.rl_env import (
    ACTION_FLAT,
    ACTION_LONG,
    ACTION_SHORT,
    PairsTradingEnv,
)
from us_experimental.rl_features import build_static_features

COMMISSION_BPS = 10.0
C = COMMISSION_BPS / 10_000.0  # per-leg commission per trade side


def make_episode(p1_vals, p2_vals, p1_0=10.0, p2_0=20.0, sigma=0.05):
    n = len(p1_vals)
    idx = pd.bdate_range("2020-01-01", periods=n)
    p1 = pd.Series(p1_vals, index=idx, dtype=float)
    p2 = pd.Series(p2_vals, index=idx, dtype=float)
    feats = build_static_features(p1, p2, p1_0, p2_0, sigma, idx[0])
    return Episode(
        ticker1="X", ticker2="Y",
        formation_start=idx[0] - pd.Timedelta(days=365),
        trading_start=idx[0], trading_end=idx[-1],
        dates=idx,
        p1=p1.to_numpy(), p2=p2.to_numpy(),
        p1_0=p1_0, p2_0=p2_0, locked_sigma=sigma,
        features=feats.to_numpy(np.float32),
    )


@pytest.fixture
def flat_episode():
    """Constant prices: zero mark-to-market PnL, only costs matter."""
    n = 10
    return make_episode([10.0] * n, [20.0] * n)


def env_for(ep, **kw):
    kw.setdefault("commission_bps", COMMISSION_BPS)
    kw.setdefault("reward_scale", 1.0)
    kw.setdefault("sampling", "sequential")
    return PairsTradingEnv([ep], **kw)


def run_actions(env, actions):
    obs, _ = env.reset(options={"episode_index": 0})
    total_pnl, infos = 0.0, []
    for a in actions:
        obs, r, term, trunc, info = env.step(a)
        total_pnl += info["pnl"]
        infos.append(info)
        if term:
            break
    return total_pnl, infos


def test_gymnasium_api_compliance(flat_episode):
    from gymnasium.utils.env_checker import check_env

    check_env(env_for(flat_episode), skip_render_check=True)


def test_open_and_close_costs(flat_episode):
    n = flat_episode.length
    # open long at bar 0, close at bar 2, flat afterwards
    actions = [ACTION_LONG, ACTION_LONG, ACTION_FLAT] + [ACTION_FLAT] * (n - 3)
    total, infos = run_actions(env_for(flat_episode), actions)
    # open costs 2c, close costs 2c; zero mtm on constant prices
    assert total == pytest.approx(-4 * C)
    assert infos[0]["cost"] == pytest.approx(2 * C)
    assert infos[2]["cost"] == pytest.approx(2 * C)


def test_flip_costs_double(flat_episode):
    n = flat_episode.length
    actions = [ACTION_LONG, ACTION_SHORT] + [ACTION_FLAT] * (n - 2)
    total, infos = run_actions(env_for(flat_episode), actions)
    # open 2c + flip 4c + close 2c
    assert infos[1]["cost"] == pytest.approx(4 * C)
    assert total == pytest.approx(-8 * C)


def test_forced_close_at_terminal(flat_episode):
    n = flat_episode.length
    actions = [ACTION_LONG] * n  # never voluntarily closes
    total, infos = run_actions(env_for(flat_episode), actions)
    assert total == pytest.approx(-4 * C)  # open + forced time-stop close
    trades = infos[-1]["trades"]
    assert len(trades) == 1
    assert trades[0]["exit_reason"] == "time"


def test_mark_to_market_uses_entry_price_denominator():
    # leg1 rises 10 -> 11 -> 12, leg2 constant; long spread from bar 0
    ep = make_episode([10.0, 11.0, 12.0, 12.0], [20.0] * 4)
    actions = [ACTION_LONG, ACTION_LONG, ACTION_LONG, ACTION_FLAT]
    total, infos = run_actions(env_for(ep), actions)
    # GGR convention: daily pnl = (p1_t - p1_{t-1}) / p1_entry
    expected_gross = (11 - 10) / 10 + (12 - 11) / 10 + 0.0
    expected = expected_gross - 4 * C
    assert total == pytest.approx(expected)
    assert infos[1]["gross_pnl"] == pytest.approx(0.1)


def test_reward_equals_scaled_pnl(flat_episode):
    n = flat_episode.length
    env = env_for(flat_episode, reward_scale=100.0)
    obs, _ = env.reset(options={"episode_index": 0})
    for a in [ACTION_SHORT] + [ACTION_FLAT] * (n - 1):
        obs, r, term, trunc, info = env.step(a)
        assert r == pytest.approx(info["pnl"] * 100.0)
        if term:
            break


def test_episode_runs_exactly_length_steps(flat_episode):
    env = env_for(flat_episode)
    obs, _ = env.reset(options={"episode_index": 0})
    steps = 0
    term = False
    while not term:
        obs, r, term, trunc, info = env.step(ACTION_FLAT)
        steps += 1
    assert steps == flat_episode.length


def test_unreal_pnl_and_position_in_obs():
    ep = make_episode([10.0, 11.0, 11.0], [20.0, 20.0, 20.0])
    env = env_for(ep)
    obs, _ = env.reset(options={"episode_index": 0})
    assert obs[5] == 0.0  # position
    obs, r, term, trunc, info = env.step(ACTION_LONG)
    assert obs[5] == 1.0
    assert obs[7] == pytest.approx(0.0)  # unreal pnl right after entry
    obs, r, term, trunc, info = env.step(ACTION_LONG)
    assert obs[7] == pytest.approx(0.1)  # (11-10)/10 accrued


def test_deterministic_episode_selection(flat_episode):
    ep2 = make_episode([30.0] * 8, [40.0] * 8, p1_0=30.0, p2_0=40.0)
    env = PairsTradingEnv([flat_episode, ep2], commission_bps=COMMISSION_BPS,
                          reward_scale=1.0, sampling="sequential")
    _, info0 = env.reset(options={"episode_index": 1})
    assert info0["episode_index"] == 1
