"""Smoke tests for us_experimental.train_rl — DQN wiring, not learning quality."""

import numpy as np
import pandas as pd
import pytest

from us_experimental.rl_agents import SB3Agent
from us_experimental.rl_episodes import Episode
from us_experimental.rl_features import build_static_features
from us_experimental.train_rl import train_dqn_seed


def _make_episode(seed, n=40):
    idx = pd.bdate_range("2014-01-01", periods=n)
    rng = np.random.default_rng(seed)
    p1 = pd.Series(10 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), index=idx)
    p2 = pd.Series(20 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), index=idx)
    feats = build_static_features(p1, p2, 10.0, 20.0, 0.05, idx[0])
    return Episode(
        ticker1=f"S{seed}", ticker2=f"T{seed}",
        formation_start=idx[0] - pd.Timedelta(days=365),
        trading_start=idx[0], trading_end=idx[-1], dates=idx,
        p1=p1.to_numpy(), p2=p2.to_numpy(), p1_0=10.0, p2_0=20.0,
        locked_sigma=0.05, features=feats.to_numpy(np.float64),
    )


@pytest.fixture(scope="module")
def tiny_episodes():
    return [_make_episode(s) for s in range(12)]


def test_train_dqn_seed_smoke(tiny_episodes, tmp_path):
    result = train_dqn_seed(
        tiny_episodes[:8], tiny_episodes[8:],
        seed=0,
        total_timesteps=800,
        eval_every=400,
        model_dir=tmp_path,
        learning_starts=100,
        buffer_size=2_000,
        verbose=False,
    )
    assert result["seed"] == 0
    assert result["model_path"].exists()
    assert len(result["history"]) == 2  # 800 / 400 evaluation points
    assert np.isfinite(result["best_val_sharpe"])

    # reload the checkpoint and act with it
    from stable_baselines3 import DQN

    agent = SB3Agent(DQN.load(result["model_path"]))
    obs = np.zeros(9, dtype=np.float64)
    action, _ = agent.predict(obs)
    assert action in (0, 1, 2)
