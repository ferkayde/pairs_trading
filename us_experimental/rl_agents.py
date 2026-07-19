"""
rl_agents.py — policies over PairsTradingEnv observations.

All agents expose the SB3-style predict(obs, state, episode_start,
deterministic) -> (action, state) interface so the evaluation code can treat
the static benchmark, the do-nothing floor and trained SB3 models uniformly.
"""

from __future__ import annotations

import numpy as np

from us_experimental.rl_env import ACTION_FLAT, ACTION_LONG, ACTION_SHORT

_Z_IDX = 0        # z feature position in the observation vector
_POS_IDX = 5      # current-position feature


class BaseAgent:
    def predict(self, obs, state=None, episode_start=None, deterministic=True):
        raise NotImplementedError


class StaticRuleAgent(BaseAgent):
    """The GGR static rule expressed as an environment policy.

    Flat  : open short spread when z >  entry_z, long when z < -entry_z.
    Long  : close on zero-crossing (z >= 0), otherwise stay long.
    Short : close on zero-crossing (z <= 0), otherwise stay short.

    With entry_z=2.0 this reproduces src.backtest.simulate_pair_returns
    exactly (tests/test_rl_agents.py parity gate). entry_z is the tunable
    threshold for the "fairer fight" baseline (grid-searched on validation).
    """

    def __init__(self, entry_z: float = 2.0):
        self.entry_z = entry_z

    def predict(self, obs, state=None, episode_start=None, deterministic=True):
        z = float(obs[_Z_IDX])
        pos = int(round(float(obs[_POS_IDX])))
        if pos == 0:
            if z > self.entry_z:
                return ACTION_SHORT, None
            if z < -self.entry_z:
                return ACTION_LONG, None
            return ACTION_FLAT, None
        if pos == 1:
            return (ACTION_FLAT if z >= 0.0 else ACTION_LONG), None
        return (ACTION_FLAT if z <= 0.0 else ACTION_SHORT), None


class FlatAgent(BaseAgent):
    """Do-nothing sanity floor: always flat, zero trades, zero PnL."""

    def predict(self, obs, state=None, episode_start=None, deterministic=True):
        return ACTION_FLAT, None


class EnsembleAgent(BaseAgent):
    """Majority vote across several trained agents (variance reduction).

    Tie-break: keep the current position if that action is among the tied
    winners; otherwise go flat — when models disagree diametrically, the
    conservative call is not to trade.
    """

    def __init__(self, agents: list[BaseAgent]):
        if not agents:
            raise ValueError("agents list is empty")
        self.agents = agents

    def predict(self, obs, state=None, episode_start=None, deterministic=True):
        votes: dict[int, int] = {}
        for a in self.agents:
            act, _ = a.predict(obs, deterministic=deterministic)
            votes[int(act)] = votes.get(int(act), 0) + 1
        top = max(votes.values())
        tied = {a for a, c in votes.items() if c == top}
        if len(tied) == 1:
            return tied.pop(), None
        pos = int(round(float(obs[_POS_IDX])))
        hold_action = {0: ACTION_FLAT, 1: ACTION_LONG, -1: ACTION_SHORT}[pos]
        if hold_action in tied:
            return hold_action, None
        return ACTION_FLAT, None


class SB3Agent(BaseAgent):
    """Wraps a trained Stable-Baselines3 model behind the common interface."""

    def __init__(self, model):
        self.model = model

    def predict(self, obs, state=None, episode_start=None, deterministic=True):
        action, st = self.model.predict(
            np.asarray(obs, dtype=np.float32), deterministic=deterministic
        )
        return int(action), st
