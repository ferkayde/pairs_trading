# us_experimental — Dynamic Pairs Trading with Reinforcement Learning

Implements `rl-dynamic-pairs-trading.md` on the US universe (`data/prices.csv`,
500 tickers, 1990–2026): replaces the static GGR open-at-2σ / close-at-mean
rule with a DQN-learned policy and asks whether it beats the static rule out of
sample, net of transaction costs.

## Modules

| File | Responsibility |
|---|---|
| `rl_features.py` | Backward-looking state features: z-score (+lags), rolling spread vol, rolling OU half-life |
| `rl_episodes.py` | Walk-forward episode builder (reuses `src.pairs` formation: filters, SSD, locked σ); train/val/test split |
| `rl_env.py` | Gymnasium env; accounting matches `src.backtest.simulate_pair_returns` bar-for-bar |
| `rl_agents.py` | `StaticRuleAgent` (GGR rule as a policy), `FlatAgent`, `SB3Agent` wrapper |
| `rl_evaluate.py` | Deterministic rollouts, GGR-convention daily aggregation, metrics, threshold tuning, block bootstrap |
| `train_rl.py` | SB3 DQN (MLP 2×64), chunked training with best-validation-Sharpe checkpointing, multi-seed |
| `run_experiment.py` | End-to-end orchestrator |

## Design anchors

- **Parity gate (Milestone 1):** the static rule executed *inside* the env
  reproduces `simulate_pair_returns` to 1e-10 on every episode; the experiment
  aborts otherwise. This makes every RL-vs-static comparison trustworthy.
- **Episode = (pair, 126-day trading window)** after a 252-day formation
  window, rolled monthly — exactly the GGR walk-forward.
- **Splits:** train < 2015, validation 2015–2019 (model selection + static
  threshold tuning), test 2020+ (touched once). Windows straddling a boundary
  are dropped.
- **Costs:** 10 bps per leg per side (repo default); open/close = 2c, flip = 4c.
- **Observation (9-dim):** z, z(t−1), z(t−5), 20d spread vol / locked σ,
  rolling OU half-life, position, days-in-position, unrealized PnL,
  days-remaining. All backward-looking; normalized only with formation-locked σ.
- **Actions:** 0 = flat/close, 1 = long spread, 2 = short spread. Forced close
  at window end (GGR time-stop).
- **Reward:** mark-to-market ΔPnL − cost·|Δposition| (optional λ·|pos|·σ risk
  penalty, default 0), ×100 for training stability only.

## Run

```bash
python -m pytest tests/test_rl_*.py         # from repo root
python us_experimental/run_experiment.py --quick    # smoke run
python us_experimental/run_experiment.py            # full: 5 seeds x 200k steps
```

Outputs land in `us_experimental/results/`:
`rl_parity_check.csv`, `rl_comparison.csv` (headline table),
`rl_bootstrap.csv` (block-bootstrap Sharpe differences),
`rl_subperiods.csv`, `rl_tuning.csv`, `rl_training_history.csv`,
`rl_equity_test.png`, `rl_behavior_thresholds.png` (learned thresholds vs
volatility regime).

Episode caches go to `us_experimental/cache/`, model checkpoints to
`us_experimental/models/` (one `dqn_seed{n}.zip` per seed).
