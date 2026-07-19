"""
rl_env.py — Gymnasium environment for dynamic pairs trading.

One episode = one (pair, trading window) tuple (see rl_episodes.Episode).
The accounting contract intentionally matches src.backtest.simulate_pair_returns
bar for bar, so that the static GGR rule executed inside this environment
reproduces the replication engine's PnL exactly (validated in
tests/test_rl_agents.py). Per step at bar i:

    1. Mark-to-market the position carried into bar i (GGR eq. 3 convention:
       daily pnl = position * (Δp1 / p1_entry - Δp2 / p2_entry)).
    2. Apply the action: the new position takes effect at bar i's close.
       Cost = 2c per unit of position change (open/close = 2c, flip = 4c),
       c = commission_bps / 10^4 per leg per trade side.
    3. On the final bar any remaining position is force-closed for another
       2c (GGR time-stop; exit_reason "time").

Reward = (pnl - cost - risk_lambda * |position| * vol_ratio) * reward_scale.
reward_scale is a monotonic training-stability rescale; evaluation must use
info["pnl"], which is always in raw GGR per-$1-per-leg units.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from us_experimental.rl_episodes import Episode
from us_experimental.rl_features import N_FEATURES

ACTION_FLAT = 0
ACTION_LONG = 1
ACTION_SHORT = 2
_ACTION_TO_POS = {ACTION_FLAT: 0, ACTION_LONG: 1, ACTION_SHORT: -1}


class PairsTradingEnv(gym.Env):
    """Discrete-action pairs-trading environment over precomputed episodes.

    Observation (float32, clipped to ±obs_clip):
        [z, z_lag1, z_lag5, vol_ratio, half_life,          <- Episode.features
         position, days_in_pos, unreal_pnl, days_left]     <- dynamic

    Actions: 0 = flat/close, 1 = long spread, 2 = short spread.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        episodes: list[Episode],
        commission_bps: float = 10.0,
        reward_scale: float = 100.0,
        risk_lambda: float = 0.0,
        sampling: str = "random",
        obs_clip: float = 10.0,
        seed: int | None = None,
    ):
        if not episodes:
            raise ValueError("episodes list is empty")
        if sampling not in ("random", "sequential"):
            raise ValueError(f"unknown sampling mode {sampling!r}")
        self.episodes = list(episodes)
        self.commission = commission_bps / 10_000.0
        self.reward_scale = reward_scale
        self.risk_lambda = risk_lambda
        self.sampling = sampling
        self.obs_clip = obs_clip
        self._rng = np.random.default_rng(seed)
        self._next_ep = 0

        # float64 observations: the static-rule parity gate needs threshold
        # decisions (z vs entry_z) made in the same precision as the reference
        # pandas simulator. SB3 casts observations to float32 internally.
        self.observation_space = spaces.Box(
            low=-obs_clip, high=obs_clip, shape=(N_FEATURES,), dtype=np.float64
        )
        self.action_space = spaces.Discrete(3)

        self._ep: Episode | None = None

    # ------------------------------------------------------------------ reset
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        if options and "episode_index" in options:
            idx = int(options["episode_index"])
        elif self.sampling == "random":
            idx = int(self._rng.integers(len(self.episodes)))
        else:
            idx = self._next_ep % len(self.episodes)
            self._next_ep += 1

        self._ep = self.episodes[idx]
        self._i = 0
        self._position = 0
        self._entry_p1 = self._entry_p2 = None
        self._entry_bar = None
        self._unreal = 0.0
        self._trades: list[dict] = []
        self._open_info: dict | None = None
        return self._obs(), {"episode_index": idx}

    # ------------------------------------------------------------------- obs
    def _obs(self) -> np.ndarray:
        ep, i = self._ep, self._i
        T = ep.length
        if self._position != 0 and self._entry_bar is not None:
            days_in_pos = (i - self._entry_bar) / T
        else:
            days_in_pos = 0.0
        dyn = np.array(
            [self._position, days_in_pos, self._unreal, (T - 1 - i) / T],
            dtype=np.float64,
        )
        obs = np.concatenate([np.asarray(ep.features[i], dtype=np.float64), dyn])
        return np.clip(obs, -self.obs_clip, self.obs_clip)

    # ----------------------------------------------------------------- trades
    def _record_close(self, exit_bar: int, exit_reason: str) -> None:
        rec = dict(self._open_info or {})
        rec.update(
            exit_bar=exit_bar,
            holding_days=exit_bar - rec.get("entry_bar", exit_bar),
            z_close=float(self._ep.features[exit_bar, 0]),
            pnl=float(self._unreal),
            exit_reason=exit_reason,
        )
        self._trades.append(rec)
        self._open_info = None

    def _open_position(self, target: int, i: int) -> None:
        self._entry_p1 = self._ep.p1[i]
        self._entry_p2 = self._ep.p2[i]
        self._entry_bar = i
        self._unreal = 0.0
        self._open_info = {
            "direction": target,
            "entry_bar": i,
            "z_open": float(self._ep.features[i, 0]),
            "vol_open": float(self._ep.features[i, 3]),
        }

    # ------------------------------------------------------------------ step
    def step(self, action):
        ep, i = self._ep, self._i
        p1, p2 = ep.p1[i], ep.p2[i]

        # 1. Mark-to-market with the position carried into this bar.
        gross = 0.0
        if self._position != 0 and i > 0:
            gross = self._position * (
                (p1 - ep.p1[i - 1]) / self._entry_p1
                - (p2 - ep.p2[i - 1]) / self._entry_p2
            )
            self._unreal += gross

        # 2. Apply the action at this bar's close.
        cost = 0.0
        target = _ACTION_TO_POS[int(action)]
        if target != self._position:
            cost += 2.0 * self.commission * abs(target - self._position)
            if self._position != 0:
                self._record_close(i, "signal")
            if target != 0:
                self._open_position(target, i)
            else:
                self._entry_p1 = self._entry_p2 = self._entry_bar = None
                self._unreal = 0.0
            self._position = target

        # 3. Time-stop: force-close on the final bar (GGR window-end handling).
        terminated = i == ep.length - 1
        if terminated and self._position != 0:
            cost += 2.0 * self.commission
            self._record_close(i, "time")
            self._position = 0
            self._entry_p1 = self._entry_p2 = self._entry_bar = None
            self._unreal = 0.0

        pnl = gross - cost
        reward = pnl
        if self.risk_lambda > 0.0:
            reward -= self.risk_lambda * abs(self._position) * float(ep.features[i, 3])
        reward *= self.reward_scale

        if not terminated:
            self._i = i + 1
        obs = self._obs()

        info = {
            "pnl": pnl,
            "gross_pnl": gross,
            "cost": cost,
            "date": ep.dates[i],
            "position": self._position,
        }
        if terminated:
            info["trades"] = self._trades
        return obs, float(reward), terminated, False, info
