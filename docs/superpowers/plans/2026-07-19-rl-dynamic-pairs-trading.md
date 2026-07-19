# RL Dynamic Pairs Trading (US) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the static GGR open-at-2σ / close-at-mean rule with a DQN-learned trading policy on US data (`data/prices.csv`, 500 tickers, 1990–2026) and answer: does the learned policy beat the static rule out of sample, net of costs?

**Architecture:** New `us_experimental/` package mirrors `bist_experimental/`. Pair formation reuses `src.pairs` unchanged. A Gymnasium env (`PairsTradingEnv`) replays precomputed (pair, trading-window) episodes with accounting that exactly matches `src.backtest.simulate_pair_returns` — validated by running the static rule *inside* the env (Milestone 1 parity gate). SB3 DQN trains on pre-2015 episodes, model selection on 2015–2019 validation Sharpe, final evaluation on 2020+ test episodes vs three baselines.

**Tech Stack:** python 3.14, gymnasium 1.3.0, stable-baselines3 2.9.0, torch 2.13.0 (CPU), pandas 3.0, numpy 2.4, statsmodels, matplotlib.

## Global Constraints

- Walk-forward splits only: train = episodes with `trading_end < 2015-01-01`; validation = `trading_start >= 2015-01-01 and trading_end < 2020-01-01`; test = `trading_start >= 2020-01-01`. Windows straddling boundaries are dropped.
- Every rolling feature is backward-looking only; normalization uses formation-window quantities (locked σ) only.
- Cost model: `commission_bps=10` per leg per side (matches repo default) → 2·c per position open, 2·c per close, 4·c per flip, c = bps/10⁴.
- Episode structure mirrors GGR: 252-day formation, 126-day trading, 21-day roll, top-20 SSD pairs.
- Network: MLP 2×64. Algorithm: DQN. Seeds: ≥5 for the headline run (configurable; reduced for smoke tests).
- Reward = ΔPnL(mark-to-market, GGR per-$1-per-leg units) − cost·|Δposition| − λ·|position|·σ (λ default 0); reward_scale=100 for training stability (monotonic, does not change optimal policy); evaluation always uses raw `info["pnl"]`.
- Observation (9-dim): `[z, z_lag1, z_lag5, vol_ratio, half_life, position, days_in_pos, unreal_pnl, days_left]`, clipped to ±10.
- Action space Discrete(3): 0=flat/close, 1=long spread, 2=short spread.
- Tests live in root `tests/` (pytest testpaths). `us_experimental` gets `__init__.py` so `from us_experimental.rl_env import ...` works from repo root.

## File Structure

- Create: `us_experimental/__init__.py` — empty package marker
- Create: `us_experimental/rl_features.py` — spread, z, rolling vol, OU half-life feature builder
- Create: `us_experimental/rl_episodes.py` — `Episode` dataclass, `build_episodes`, `split_episodes`, pickle cache
- Create: `us_experimental/rl_env.py` — `PairsTradingEnv(gym.Env)`
- Create: `us_experimental/rl_agents.py` — `StaticRuleAgent`, `FlatAgent`, `SB3Agent`
- Create: `us_experimental/rl_evaluate.py` — rollouts, aggregation, metrics, threshold tuning, block bootstrap
- Create: `us_experimental/train_rl.py` — DQN per-seed training with val-Sharpe checkpointing
- Create: `us_experimental/run_experiment.py` — end-to-end orchestrator (CLI)
- Create: `us_experimental/README.md`
- Modify: `requirements.txt` — add gymnasium, stable-baselines3, torch
- Test: `tests/test_rl_features.py`, `tests/test_rl_episodes.py`, `tests/test_rl_env.py`, `tests/test_rl_agents.py`, `tests/test_rl_evaluate.py`, `tests/test_rl_train.py`

---

### Task 1: Features (`rl_features.py`)

**Interfaces — Produces:**
- `STATIC_FEATURES = ["z", "z_lag1", "z_lag5", "vol_ratio", "half_life"]`, `DYNAMIC_FEATURES = ["position", "days_in_pos", "unreal_pnl", "days_left"]`, `N_FEATURES = 9`
- `normalized_spread(p1, p2, p1_0, p2_0) -> pd.Series`
- `rolling_half_life(spread: pd.Series, window: int = 60) -> pd.Series` (bars; +inf where no mean reversion)
- `build_static_features(p1, p2, p1_0, p2_0, locked_sigma, trading_start, vol_window=20, hl_window=60, max_half_life=126.0) -> pd.DataFrame` — rows are bars ≥ trading_start, columns STATIC_FEATURES; NaN-free (leading NaNs → 0.0, half-life NaN/inf → 1.0 after /max normalization).

- [ ] Write failing tests (`tests/test_rl_features.py`): z equals spread/σ; half-life recovers a synthetic AR(1)'s true half-life within tolerance; **look-ahead audit** — mutating prices after bar t leaves features at ≤ t unchanged; output NaN-free with formation warm-up.
- [ ] Run tests → FAIL (module missing)
- [ ] Implement:

```python
def rolling_half_life(spread, window=60):
    s_lag = spread.shift(1)
    ds = spread - s_lag
    b = ds.rolling(window).cov(s_lag) / s_lag.rolling(window).var()
    phi = 1.0 + b
    hl = np.where((phi > 0) & (phi < 1), -np.log(2.0) / np.log(phi), np.inf)
    return pd.Series(hl, index=spread.index)

def build_static_features(p1, p2, p1_0, p2_0, locked_sigma, trading_start,
                          vol_window=20, hl_window=60, max_half_life=126.0):
    spread = p1 / p1_0 - p2 / p2_0
    z = spread / locked_sigma
    vol_ratio = spread.diff().rolling(vol_window).std() / locked_sigma
    hl = rolling_half_life(spread, hl_window)
    hl_norm = hl.clip(upper=max_half_life).fillna(max_half_life) / max_half_life
    feats = pd.DataFrame({"z": z, "z_lag1": z.shift(1), "z_lag5": z.shift(5),
                          "vol_ratio": vol_ratio, "half_life": hl_norm})
    return feats.loc[trading_start:].fillna(0.0)
```

- [ ] Run tests → PASS; commit `feat: RL feature construction for us_experimental`

### Task 2: Episodes (`rl_episodes.py`)

**Interfaces — Consumes:** `src.pairs` (normalize_prices, compute_ssd, select_top_pairs, compute_locked_sigma, liquidity_filter, activity_filter), Task 1 `build_static_features`.
**Produces:**
- `@dataclass Episode`: `ticker1, ticker2, formation_start, trading_start, trading_end, dates (DatetimeIndex of trading bars), p1, p2 (np.ndarray float), p1_0, p2_0, locked_sigma (float), features (np.ndarray float32 (T,5))`; property `length`.
- `build_episodes(prices, formation_days=252, trading_days=126, roll_days=21, top_n=20, min_trading_bars=5, vol_window=20, hl_window=60, verbose=True) -> list[Episode]` — same window loop / filters / SSD ranking / σ-locking as `run_ggr_portfolio`; trading bars = dropna intersection of both legs within the trading window, all prices required > 0; features built on the formation+trading common series so rolling stats are warm.
- `split_episodes(episodes, train_end="2015-01-01", val_end="2020-01-01") -> dict[str, list[Episode]]` with keys train/val/test per Global Constraints.
- `save_episodes(episodes, path)` / `load_episodes(path)` (pickle).

- [ ] Write failing tests: synthetic 6-ticker/3-year cointegrated universe → episodes have `len(p1)==len(features)==length`, dates inside trading window, σ>0; split boundary logic incl. straddling windows dropped; save/load round-trip.
- [ ] Run → FAIL; implement; run → PASS; commit `feat: RL walk-forward episode builder`

### Task 3: Environment (`rl_env.py`)

**Interfaces — Consumes:** `Episode`, `N_FEATURES`. **Produces:**
- `ACTION_FLAT=0, ACTION_LONG=1, ACTION_SHORT=2`
- `PairsTradingEnv(episodes, commission_bps=10.0, reward_scale=100.0, risk_lambda=0.0, sampling="random"|"sequential", obs_clip=10.0, seed=None)`; `reset(options={"episode_index": k})` supported; `info` per step has `pnl` (raw net), `gross_pnl`, `cost`, `date`, `position`; terminal info has `trades` list of dicts (`direction, entry_bar, exit_bar, holding_days, z_open, vol_open, z_close, pnl, exit_reason∈{"signal","time"}`).

**Accounting contract (must equal `simulate_pair_returns` bar-for-bar):** step at bar i = (1) mark-to-market with the position carried into bar i, denominators = entry prices; (2) apply action → position change at bar-i close, cost `2c·|Δpos|`, entry prices recorded at bar i; (3) if last bar, force-close any remaining position for another 2c (`exit_reason="time"`). Episode = exactly `length` steps.

- [ ] Write failing tests: `gymnasium.utils.env_checker.check_env` passes; open/close/flip costs; forced terminal close; sum(reward)/reward_scale == sum(info.pnl); deterministic reset via `episode_index`; unreal_pnl feature resets on open.
- [ ] Run → FAIL; implement; run → PASS; commit `feat: Gymnasium pairs-trading environment`

### Task 4: Agents + Milestone-1 parity (`rl_agents.py`)

**Produces:**
- `StaticRuleAgent(entry_z=2.0)` — `.predict(obs, ...) -> (action, None)`: flat→short if z>entry_z, long if z<−entry_z; long→close at z≥0; short→close at z≤0 (zero-crossing exit hardcoded, matching GGR).
- `FlatAgent`, `SB3Agent(model)` (same predict signature; wraps `model.predict(deterministic=True)`).

- [ ] Write failing tests incl. **the parity gate**: for one real-data window (`data/prices.csv`, e.g. formation starting 2005), every episode's static-agent PnL series inside the env equals `src.backtest.simulate_pair_returns(...)` (same commission, entry_sigma=2, wait_one_day=False) to 1e-10, and n_trades match. Plus synthetic-path parity.
- [ ] Run → FAIL; implement; run → PASS; commit `feat: static-rule/flat/SB3 agents + env parity validation`

### Task 5: Evaluation (`rl_evaluate.py`)

**Produces:**
- `rollout_episode(episode, agent, commission_bps=10.0) -> (pd.Series daily returns, list trades)`
- `aggregate_daily(series_list) -> pd.Series` — equal-weight mean across active episodes per date (same convention as `run_ggr_portfolio._aggregate`)
- `evaluate_policy(episodes, agent, commission_bps=10.0) -> dict`: `daily, equity, sharpe, total_return_pct, max_drawdown, n_trades, win_rate, avg_holding_days, trades_per_episode, trade_log`
- `tune_static_entry(val_episodes, grid=(1.5,1.75,2.0,2.25,2.5,2.75,3.0), commission_bps=10.0) -> (best_z, pd.DataFrame)` — maximize validation Sharpe
- `bootstrap_sharpe_diff(daily_a, daily_b, n_boot=2000, block_len=21, seed=42) -> dict`: moving-block bootstrap on the aligned pair of daily series; `diff_observed, ci_low, ci_high, p_value` (p = fraction of resamples with diff ≤ 0)
- `subperiod_table(daily, breakpoints) -> pd.DataFrame`

- [ ] Failing tests: FlatAgent → zero returns/0 trades; aggregate matches hand-computed mean; tuning returns grid member; bootstrap of identical series → diff 0, p≈ambient; run → FAIL; implement; PASS; commit `feat: RL policy evaluation and significance testing`

### Task 6: Training (`train_rl.py`)

**Produces:**
- `train_dqn_seed(train_eps, val_eps, seed, total_timesteps=200_000, eval_every=25_000, commission_bps=10.0, model_dir="us_experimental/models", val_subsample=300, learning_rate=1e-4, reward_scale=100.0, verbose=True) -> dict`: chunked `model.learn(..., reset_num_timesteps=False)`, after each chunk evaluate `SB3Agent(model)` on (subsampled, fixed-seed) validation episodes, keep checkpoint with best val Sharpe → `dqn_seed{seed}.zip`; returns `{seed, model_path, best_val_sharpe, history}`.
- DQN config: `MlpPolicy`, `net_arch=[64,64]`, `buffer_size=200_000`, `learning_starts=5_000`, `batch_size=64`, `train_freq=4`, `target_update_interval=2_000`, `exploration_fraction=0.3`, `exploration_final_eps=0.05`, `gamma=0.99`, seeded.
- `train_seeds(train_eps, val_eps, seeds, **kw) -> list[dict]`

- [ ] Smoke test: 12 tiny synthetic episodes, `total_timesteps=800, eval_every=400` → returns dict, model file exists, loaded SB3Agent emits valid actions; run → FAIL; implement; PASS; commit `feat: DQN training loop with validation-based model selection`

### Task 7: Orchestrator (`run_experiment.py`) + README + requirements

- CLI: `--quick` (ticker subset 120, roll 63, 2 seeds, 30k steps), `--seeds N` (default 5), `--timesteps` (default 200k), `--commission` (default 10), `--rebuild-cache`.
- Steps: load prices → build/cache episodes (`us_experimental/cache/episodes.pkl`) → split → **parity check across ALL episodes** (max |diff| written to `results/rl_parity_check.csv`; abort if > 1e-8) → baselines (static 2σ, tuned static via validation grid, flat) → train seeds → per-seed test evaluation → outputs:
  - `results/rl_comparison.csv` (all policies × metrics, DQN mean±std across seeds)
  - `results/rl_equity_test.png`, `results/rl_behavior_thresholds.png` (|z_open| vs vol_open, RL vs static), `results/rl_subperiods.csv`, `results/rl_bootstrap.csv`
- `README.md` documenting the pipeline and how to reproduce; `requirements.txt` += `gymnasium>=1.0.0`, `stable-baselines3>=2.3.0`, `torch>=2.2.0`.
- [ ] Implement; smoke `--quick` run completes end-to-end; commit `feat: RL experiment orchestrator`

### Task 8: Full experiment run

- [ ] `python us_experimental/run_experiment.py` (full data, ≥5 seeds if runtime permits — otherwise document reduced setup), verify parity gate passes, results written; commit results + any fixes.

## Self-Review Notes

- Spec coverage: state features (all 8) ✓, discrete 3-action ✓, reward with cost + optional λ ✓, GGR episode structure ✓, DQN+MLP64 ✓, ≥5 seeds (headline, configurable) ✓, walk-forward splits ✓, 3 baselines ✓, metrics ✓, bootstrap significance ✓, behavioral thresholds-vs-vol figure ✓, sub-period reporting ✓, Milestone-1 parity gate ✓. PPO robustness check and short-ban handling (BIST-specific) deferred as spec extensions.
- Type consistency: `Episode` field names and agent `.predict` signature used identically across Tasks 3–7.
