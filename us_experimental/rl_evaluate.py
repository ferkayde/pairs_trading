"""
rl_evaluate.py — policy evaluation, baselines tuning and significance tests.

Rollouts run any agent (static rule, flat, trained SB3 model) deterministically
over a list of episodes; daily returns are aggregated with the same
equal-weight-across-active-episodes convention as run_ggr_portfolio, so all
policies and the existing replication numbers share one return basis.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.metrics import max_drawdown, sharpe_ratio  # noqa: E402
from us_experimental.rl_agents import BaseAgent, StaticRuleAgent  # noqa: E402
from us_experimental.rl_env import PairsTradingEnv  # noqa: E402
from us_experimental.rl_episodes import Episode  # noqa: E402


def rollout_episode(
    episode: Episode, agent: BaseAgent, commission_bps: float = 10.0
) -> tuple[pd.Series, list[dict]]:
    """Run one episode with the agent acting greedily; return (net daily
    returns indexed by trading dates, trade log)."""
    env = PairsTradingEnv(
        [episode], commission_bps=commission_bps, reward_scale=1.0,
        sampling="sequential",
    )
    obs, _ = env.reset(options={"episode_index": 0})
    pnl = np.zeros(episode.length)
    trades: list[dict] = []
    for t in range(episode.length):
        action, _ = agent.predict(obs, deterministic=True)
        obs, _r, term, _tr, info = env.step(action)
        pnl[t] = info["pnl"]
        if term:
            trades = info["trades"]
            break
    return pd.Series(pnl, index=episode.dates), trades


def aggregate_daily(series_list: list[pd.Series]) -> pd.Series:
    """Equal-weight mean across episodes active on each date (same convention
    as run_ggr_portfolio's window aggregation)."""
    if not series_list:
        return pd.Series(dtype=float)
    all_idx = pd.DatetimeIndex(sorted({d for s in series_list for d in s.index}))
    total = pd.Series(0.0, index=all_idx)
    cnt = pd.Series(0, index=all_idx)
    for s in series_list:
        c = all_idx.intersection(s.index)
        total[c] += s[c]
        cnt[c] += 1
    return total.div(cnt.replace(0, np.nan)).fillna(0.0)


def evaluate_policy(
    episodes: list[Episode],
    agent: BaseAgent,
    commission_bps: float = 10.0,
    verbose: bool = False,
) -> dict:
    """Deterministic rollout of `agent` over all episodes + summary metrics.

    Returns dict with: daily, equity, sharpe, total_return_pct, max_drawdown,
    n_trades, win_rate, avg_holding_days, trades_per_episode, trade_log.
    """
    per_episode: list[pd.Series] = []
    trade_log: list[dict] = []
    for k, ep in enumerate(episodes):
        if verbose and k % 200 == 0:
            print(f"    rollout {k}/{len(episodes)}", end="\r", flush=True)
        rets, trades = rollout_episode(ep, agent, commission_bps)
        per_episode.append(rets)
        for t in trades:
            t = dict(t)
            t["ticker1"], t["ticker2"] = ep.ticker1, ep.ticker2
            t["trading_start"] = ep.trading_start
            trade_log.append(t)

    daily = aggregate_daily(per_episode)
    equity = (1 + daily).cumprod() if len(daily) else pd.Series(dtype=float)
    pnls = [t["pnl"] for t in trade_log]
    return {
        "daily": daily,
        "equity": equity,
        "sharpe": sharpe_ratio(daily) if len(daily) else 0.0,
        "total_return_pct": float((equity.iloc[-1] - 1) * 100) if len(equity) else 0.0,
        "max_drawdown": max_drawdown(equity) if len(equity) else 0.0,
        "n_trades": len(trade_log),
        "win_rate": float(np.mean([p > 0 for p in pnls])) if pnls else 0.0,
        "avg_holding_days": float(np.mean([t["holding_days"] for t in trade_log]))
        if trade_log else 0.0,
        "trades_per_episode": len(trade_log) / len(episodes) if episodes else 0.0,
        "trade_log": trade_log,
    }


def tune_static_entry(
    val_episodes: list[Episode],
    grid: tuple = (1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0),
    commission_bps: float = 10.0,
) -> tuple[float, pd.DataFrame]:
    """Grid-search the static rule's entry threshold on validation Sharpe.

    The 'fairer fight' baseline (spec §5.2): the static rule gets the same
    validation data the RL agent's model selection uses.
    """
    rows = []
    for z in grid:
        res = evaluate_policy(val_episodes, StaticRuleAgent(entry_z=z), commission_bps)
        rows.append({
            "entry_z": z,
            "sharpe": res["sharpe"],
            "total_return_pct": res["total_return_pct"],
            "max_drawdown": res["max_drawdown"],
            "n_trades": res["n_trades"],
        })
    table = pd.DataFrame(rows)
    best_z = float(table.loc[table["sharpe"].idxmax(), "entry_z"])
    return best_z, table


def bootstrap_sharpe_diff(
    daily_a: pd.Series,
    daily_b: pd.Series,
    n_boot: int = 2000,
    block_len: int = 21,
    seed: int = 42,
) -> dict:
    """Moving-block bootstrap of the Sharpe difference (A minus B).

    Daily series are aligned on the union of dates (missing days = 0, the
    committed-capital convention). Blocks of consecutive days are resampled
    JOINTLY for both series, preserving autocorrelation and the cross-series
    dependence. p_value = fraction of resamples with diff <= 0 (one-sided:
    is A genuinely better than B?).
    """
    idx = daily_a.index.union(daily_b.index)
    a = daily_a.reindex(idx, fill_value=0.0).to_numpy()
    b = daily_b.reindex(idx, fill_value=0.0).to_numpy()
    n = len(idx)
    observed = sharpe_ratio(pd.Series(a)) - sharpe_ratio(pd.Series(b))

    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block_len))
    diffs = np.empty(n_boot)
    for k in range(n_boot):
        starts = rng.integers(0, max(1, n - block_len + 1), size=n_blocks)
        take = np.concatenate([np.arange(s, s + block_len) for s in starts])[:n]
        diffs[k] = sharpe_ratio(pd.Series(a[take])) - sharpe_ratio(pd.Series(b[take]))

    return {
        "diff_observed": float(observed),
        "ci_low": float(np.percentile(diffs, 2.5)),
        "ci_high": float(np.percentile(diffs, 97.5)),
        "p_value": float(np.mean(diffs <= 0.0)),
        "n_boot": n_boot,
        "block_len": block_len,
    }


def subperiod_table(daily: pd.Series, breakpoints: list[str]) -> pd.DataFrame:
    """Metrics per sub-period (spec §6 non-stationarity: report by era)."""
    edges = (
        [daily.index[0]]
        + [pd.Timestamp(b) for b in breakpoints]
        + [daily.index[-1] + pd.Timedelta(days=1)]
    )
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        seg = daily[(daily.index >= lo) & (daily.index < hi)]
        if len(seg) == 0:
            continue
        eq = (1 + seg).cumprod()
        rows.append({
            "start": seg.index[0].date(),
            "end": seg.index[-1].date(),
            "sharpe": sharpe_ratio(seg),
            "total_return_pct": float((eq.iloc[-1] - 1) * 100),
            "max_drawdown": max_drawdown(eq),
        })
    return pd.DataFrame(rows)
