"""
train_rl.py — DQN training with validation-Sharpe model selection.

Training runs in chunks; after each chunk the current policy is evaluated
deterministically on (a fixed subsample of) the validation episodes and the
checkpoint with the best validation Sharpe is kept. This is the early-stopping
defense against overfitting (spec §6): RL on financial data will happily
memorize noise, so the final model is the best-on-validation snapshot, never
the last iterate.

Network deliberately small (MLP 2x64) and hyperparameters near SB3 defaults;
seeds are first-class because RL results in finance are seed-sensitive.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from us_experimental.rl_agents import SB3Agent  # noqa: E402
from us_experimental.rl_env import PairsTradingEnv  # noqa: E402
from us_experimental.rl_episodes import Episode  # noqa: E402
from us_experimental.rl_evaluate import evaluate_policy  # noqa: E402


def train_dqn_seed(
    train_episodes: list[Episode],
    val_episodes: list[Episode],
    seed: int,
    total_timesteps: int = 200_000,
    eval_every: int = 25_000,
    commission_bps: float = 10.0,
    model_dir="us_experimental/models",
    val_subsample: int = 300,
    learning_rate: float = 1e-4,
    reward_scale: float = 100.0,
    risk_lambda: float = 0.0,
    buffer_size: int = 200_000,
    learning_starts: int = 5_000,
    verbose: bool = True,
    train_commission_bps: float | None = None,
    min_holding_days: int = 0,
) -> dict:
    """Train one DQN seed; checkpoint the best-validation-Sharpe policy.

    Anti-churn options (spec §6 'reward hacking via cost model'):
    train_commission_bps lets the TRAINING reward charge a higher cost than
    the evaluation cost (commission_bps), regularizing against turnover;
    validation/model selection always uses the real commission_bps.
    min_holding_days applies the env's minimum-holding constraint in both
    training and validation (it is part of the policy definition).

    Returns {seed, model_path, best_val_sharpe, history} where history is a
    list of {timesteps, val_sharpe, val_return_pct, val_trades} dicts.
    """
    from stable_baselines3 import DQN
    from stable_baselines3.common.monitor import Monitor

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"dqn_seed{seed}.zip"

    # Fixed validation subsample so every evaluation is comparable across
    # chunks and seeds (subsampling only bounds evaluation cost).
    rng = np.random.default_rng(12345)
    if len(val_episodes) > val_subsample:
        pick = rng.choice(len(val_episodes), size=val_subsample, replace=False)
        val_eval = [val_episodes[i] for i in sorted(pick)]
    else:
        val_eval = list(val_episodes)

    env = Monitor(PairsTradingEnv(
        train_episodes,
        commission_bps=(train_commission_bps
                        if train_commission_bps is not None else commission_bps),
        reward_scale=reward_scale, risk_lambda=risk_lambda,
        sampling="random", seed=seed,
        min_holding_days=min_holding_days,
    ))

    model = DQN(
        "MlpPolicy",
        env,
        policy_kwargs={"net_arch": [64, 64]},
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        learning_starts=learning_starts,
        batch_size=64,
        gamma=0.99,
        train_freq=4,
        target_update_interval=2_000,
        exploration_fraction=0.3,
        exploration_final_eps=0.05,
        seed=seed,
        verbose=0,
    )

    best_sharpe = -np.inf
    history: list[dict] = []
    trained = 0
    while trained < total_timesteps:
        chunk = min(eval_every, total_timesteps - trained)
        model.learn(total_timesteps=chunk, reset_num_timesteps=False,
                    progress_bar=False)
        trained += chunk

        val = evaluate_policy(val_eval, SB3Agent(model), commission_bps,
                              min_holding_days=min_holding_days)
        history.append({
            "timesteps": trained,
            "val_sharpe": val["sharpe"],
            "val_return_pct": val["total_return_pct"],
            "val_trades": val["n_trades"],
        })
        if verbose:
            print(f"  seed {seed}  {trained:>7}/{total_timesteps} steps  "
                  f"val Sharpe {val['sharpe']:+.3f}  "
                  f"trades {val['n_trades']}", flush=True)
        if val["sharpe"] > best_sharpe:
            best_sharpe = val["sharpe"]
            model.save(model_path)

    if not model_path.exists():  # all evaluations were -inf/NaN — keep last
        model.save(model_path)

    return {
        "seed": seed,
        "model_path": model_path,
        "best_val_sharpe": float(best_sharpe) if np.isfinite(best_sharpe) else 0.0,
        "history": history,
    }


def train_seeds(
    train_episodes: list[Episode],
    val_episodes: list[Episode],
    seeds: list[int],
    **kwargs,
) -> list[dict]:
    """Train one model per seed (spec §4: report mean ± std across seeds)."""
    results = []
    for seed in seeds:
        print(f"--- training DQN seed {seed} ---", flush=True)
        results.append(train_dqn_seed(train_episodes, val_episodes, seed, **kwargs))
    return results
