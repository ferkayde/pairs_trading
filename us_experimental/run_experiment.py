"""
run_experiment.py — end-to-end RL dynamic pairs trading experiment (US data).

Pipeline (spec: rl-dynamic-pairs-trading.md):
    1. Build (or load cached) walk-forward episodes from data/prices.csv.
    2. Split train (<2015) / validation (2015-2019) / test (2020+).
    3. Parity gate: static rule inside the env must reproduce
       src.backtest.simulate_pair_returns on EVERY episode (Milestone 1).
    4. Baselines on test: static 2σ, validation-tuned static, do-nothing.
    5. Train DQN with N seeds (validation-Sharpe checkpointing).
    6. Evaluate every seed on test; report mean ± std, bootstrap significance,
       sub-period breakdown and the learned-thresholds-vs-volatility figure.

Usage:
    python us_experimental/run_experiment.py            # full run, 5 seeds
    python us_experimental/run_experiment.py --quick    # small smoke run
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.backtest import simulate_pair_returns  # noqa: E402
from src.metrics import sharpe_ratio  # noqa: E402
from us_experimental.rl_agents import (  # noqa: E402
    EnsembleAgent, FlatAgent, SB3Agent, StaticRuleAgent,
)
from us_experimental.rl_episodes import (  # noqa: E402
    build_episodes, load_episodes, save_episodes, split_episodes,
)
from us_experimental.rl_evaluate import (  # noqa: E402
    aggregate_daily, bootstrap_sharpe_diff, evaluate_policy, rollout_episode,
    subperiod_table, tune_static_entry,
)
from us_experimental.train_rl import train_seeds  # noqa: E402

DATA = _ROOT / "data" / "prices.csv"
RESULTS = _HERE / "results"
CACHE = _HERE / "cache"
MODELS = _HERE / "models"


# ----------------------------------------------------------------- pipeline
def get_episodes(args) -> list:
    tag = "quick" if args.quick else "full"
    cache_path = CACHE / f"episodes_{tag}_roll{args.roll_days}.pkl"
    if cache_path.exists() and not args.rebuild_cache:
        print(f"Loading cached episodes from {cache_path.name} ...")
        return load_episodes(cache_path)

    print("Building episodes from price data ...")
    prices = pd.read_csv(DATA, index_col=0, parse_dates=True)
    if args.quick:
        # widest-history tickers keep the quick universe meaningful
        counts = prices.notna().sum().sort_values(ascending=False)
        prices = prices[counts.index[:120]]
    t0 = time.time()
    episodes = build_episodes(prices, roll_days=args.roll_days, top_n=20)
    print(f"  built {len(episodes)} episodes in {time.time() - t0:.0f}s")
    save_episodes(episodes, cache_path)
    return episodes


def parity_check(episodes, commission_bps: float) -> pd.DataFrame:
    """Milestone-1 gate over every episode; abort on any mismatch."""
    print(f"Parity check: static rule in env vs simulate_pair_returns "
          f"({len(episodes)} episodes) ...")
    agent = StaticRuleAgent(entry_z=2.0)
    max_diff = 0.0
    worst = None
    n_trades_env = n_trades_ref = 0
    for k, ep in enumerate(episodes):
        if k % 250 == 0:
            print(f"    {k}/{len(episodes)}", end="\r", flush=True)
        env_pnl, trades = rollout_episode(ep, agent, commission_bps)
        ref_pnl, ref_n = simulate_pair_returns(
            pd.Series(ep.p1, index=ep.dates), pd.Series(ep.p2, index=ep.dates),
            ep.p1_0, ep.p2_0, ep.locked_sigma,
            entry_sigma=2.0, commission_bps=commission_bps,
        )
        diff = float(np.max(np.abs(env_pnl.values - ref_pnl.values)))
        if diff > max_diff:
            max_diff = diff
            worst = f"{ep.ticker1}/{ep.ticker2} {ep.trading_start.date()}"
        n_trades_env += len(trades)
        n_trades_ref += ref_n

    report = pd.DataFrame([{
        "n_episodes": len(episodes),
        "max_abs_pnl_diff": max_diff,
        "worst_episode": worst,
        "n_trades_env": n_trades_env,
        "n_trades_reference": n_trades_ref,
        "passed": max_diff < 1e-8 and n_trades_env == n_trades_ref,
    }])
    report.to_csv(RESULTS / "rl_parity_check.csv", index=False)
    print(f"\n  max |PnL diff| = {max_diff:.2e}   "
          f"trades env/ref = {n_trades_env}/{n_trades_ref}")
    if not bool(report["passed"].iloc[0]):
        raise SystemExit("PARITY CHECK FAILED — environment accounting does "
                         "not match the replication engine; aborting.")
    return report


def metrics_row(name: str, res: dict) -> dict:
    return {
        "policy": name,
        "sharpe": res["sharpe"],
        "total_return_pct": res["total_return_pct"],
        "max_drawdown": res["max_drawdown"],
        "n_trades": res["n_trades"],
        "win_rate": res["win_rate"],
        "avg_holding_days": res["avg_holding_days"],
        "trades_per_episode": res["trades_per_episode"],
    }


def behavior_figure(rl_trade_logs: list[list[dict]], static_trades: list[dict]):
    """|z at open| vs volatility state at open — has the agent learned
    vol-adaptive thresholds? (spec §5.5: 'the most interesting figure')."""
    fig, ax = plt.subplots(figsize=(9, 6))
    st_v = [t["vol_open"] for t in static_trades]
    st_z = [abs(t["z_open"]) for t in static_trades]
    ax.scatter(st_v, st_z, s=12, alpha=0.35, color="#888888",
               label=f"static 2σ rule ({len(st_z)} trades)")
    rl_v = [t["vol_open"] for log in rl_trade_logs for t in log]
    rl_z = [abs(t["z_open"]) for log in rl_trade_logs for t in log]
    if rl_z:
        ax.scatter(rl_v, rl_z, s=12, alpha=0.35, color="#d62728",
                   label=f"DQN policy ({len(rl_z)} trades, all seeds)")
        # binned medians to expose the effective threshold curve
        df = pd.DataFrame({"v": rl_v, "z": rl_z})
        df["bin"] = pd.qcut(df["v"], q=min(8, max(2, len(df) // 30)),
                            duplicates="drop")
        med = df.groupby("bin", observed=True).median()
        ax.plot(med["v"], med["z"], "-o", color="#7a1518", lw=2,
                label="DQN median opening |z| by vol bin")
    ax.axhline(2.0, color="#444444", ls="--", lw=1, label="static threshold (2σ)")
    ax.set_xlabel("spread volatility at open (20d σ of Δspread / locked σ)")
    ax.set_ylabel("|z-score| at position open")
    ax.set_title("Learned opening thresholds vs volatility regime (test set)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "rl_behavior_thresholds.png", dpi=150)
    plt.close(fig)


def equity_figure(test_results: dict[str, dict], rl_daily_mean: pd.Series):
    fig, ax = plt.subplots(figsize=(11, 6))
    for name, res, color in [
        ("static 2σ", test_results["static_2sigma"], "#1f77b4"),
        (test_results["tuned_label"], test_results["static_tuned"], "#2ca02c"),
    ]:
        eq = res["equity"]
        ax.plot(eq.index, eq.values, label=name, color=color, lw=1.6)
    for seed_name, res in test_results["dqn_seeds"].items():
        eq = res["equity"]
        ax.plot(eq.index, eq.values, color="#d62728", lw=0.7, alpha=0.35)
    eq_rl = (1 + rl_daily_mean).cumprod()
    ax.plot(eq_rl.index, eq_rl.values, color="#d62728", lw=2.2,
            label=f"DQN (mean of {len(test_results['dqn_seeds'])} seeds)")
    if test_results.get("ensemble") is not None:
        eq_e = test_results["ensemble"]["equity"]
        ax.plot(eq_e.index, eq_e.values, color="#9467bd", lw=2.0,
                label="DQN ensemble (majority vote)")
    ax.axhline(1.0, color="#999999", lw=0.8, ls=":")
    ax.set_title("Out-of-sample test equity (2020+), net of costs")
    ax.set_ylabel("cumulative value of $1 (per-pair-leg return basis)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "rl_equity_test.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true",
                    help="small smoke run: 120 tickers, quarterly roll, "
                         "2 seeds, 30k timesteps")
    ap.add_argument("--seeds", type=int, default=None,
                    help="number of DQN seeds (default 5, quick: 2)")
    ap.add_argument("--timesteps", type=int, default=None,
                    help="training timesteps per seed (default 200k, quick: 30k)")
    ap.add_argument("--commission", type=float, default=10.0,
                    help="commission bps per leg per side (default 10)")
    ap.add_argument("--roll-days", type=int, default=None,
                    help="walk-forward roll step (default 21, quick: 63)")
    ap.add_argument("--rebuild-cache", action="store_true")
    ap.add_argument("--train-end", default="2015-01-01")
    ap.add_argument("--val-end", default="2020-01-01")
    ap.add_argument("--train-commission-bps", type=float, default=None,
                    help="anti-churn: cost charged in the TRAINING reward "
                         "(evaluation always uses --commission)")
    ap.add_argument("--min-holding-days", type=int, default=0,
                    help="anti-churn: bars an RL position must be held before "
                         "it can be closed/flipped (applies to the RL policy "
                         "in training and evaluation)")
    args = ap.parse_args()

    args.roll_days = args.roll_days or (63 if args.quick else 21)
    n_seeds = args.seeds or (2 if args.quick else 5)
    timesteps = args.timesteps or (30_000 if args.quick else 200_000)
    cbps = args.commission

    RESULTS.mkdir(exist_ok=True)
    MODELS.mkdir(exist_ok=True)

    episodes = get_episodes(args)
    split = split_episodes(episodes, args.train_end, args.val_end)
    print(f"Episodes: train {len(split['train'])}, val {len(split['val'])}, "
          f"test {len(split['test'])}")
    if not (split["train"] and split["val"] and split["test"]):
        raise SystemExit("empty split — check data range / split dates")

    # ------------------------------------------------ Milestone-1 parity gate
    parity_check(episodes, cbps)

    # ------------------------------------------------------------- baselines
    print("Baselines on test ...")
    res_static = evaluate_policy(split["test"], StaticRuleAgent(2.0), cbps)
    res_flat = evaluate_policy(split["test"], FlatAgent(), cbps)

    print("Tuning static threshold on validation ...")
    best_z, tune_table = tune_static_entry(split["val"], commission_bps=cbps)
    tune_table.to_csv(RESULTS / "rl_tuning.csv", index=False)
    print(f"  best validation entry_z = {best_z}")
    res_tuned = evaluate_policy(split["test"], StaticRuleAgent(best_z), cbps)

    # -------------------------------------------------------------- training
    print(f"Training DQN: {n_seeds} seeds x {timesteps} timesteps ...")
    if args.train_commission_bps is not None:
        print(f"  anti-churn: training reward cost {args.train_commission_bps} bps"
              f" (evaluation {cbps} bps)")
    if args.min_holding_days:
        print(f"  anti-churn: minimum holding {args.min_holding_days} bars")
    seed_infos = train_seeds(
        split["train"], split["val"], seeds=list(range(n_seeds)),
        total_timesteps=timesteps, commission_bps=cbps, model_dir=MODELS,
        train_commission_bps=args.train_commission_bps,
        min_holding_days=args.min_holding_days,
    )
    pd.DataFrame(
        [dict(h, seed=si["seed"]) for si in seed_infos for h in si["history"]]
    ).to_csv(RESULTS / "rl_training_history.csv", index=False)

    # ------------------------------------------------------------ evaluation
    from stable_baselines3 import DQN

    print("Evaluating DQN seeds on test ...")
    dqn_results: dict[str, dict] = {}
    seed_agents: list[SB3Agent] = []
    for si in seed_infos:
        agent = SB3Agent(DQN.load(si["model_path"]))
        seed_agents.append(agent)
        dqn_results[f"dqn_seed{si['seed']}"] = evaluate_policy(
            split["test"], agent, cbps, min_holding_days=args.min_holding_days)

    res_ensemble = None
    if len(seed_agents) > 1:
        print("Evaluating seed-ensemble (majority vote) on test ...")
        res_ensemble = evaluate_policy(
            split["test"], EnsembleAgent(seed_agents), cbps,
            min_holding_days=args.min_holding_days)

    rows = [
        metrics_row("static_2sigma", res_static),
        metrics_row(f"static_tuned_{best_z}sigma", res_tuned),
        metrics_row("flat_do_nothing", res_flat),
    ]
    for name, res in dqn_results.items():
        rows.append(metrics_row(name, res))

    seed_sharpes = [r["sharpe"] for r in dqn_results.values()]
    seed_rets = [r["total_return_pct"] for r in dqn_results.values()]
    rl_daily_mean = aggregate_daily([r["daily"] for r in dqn_results.values()])
    rows.append({
        "policy": "dqn_mean_across_seeds",
        "sharpe": float(np.mean(seed_sharpes)),
        "total_return_pct": float(np.mean(seed_rets)),
        "max_drawdown": float(np.mean([r["max_drawdown"] for r in dqn_results.values()])),
        "n_trades": float(np.mean([r["n_trades"] for r in dqn_results.values()])),
        "win_rate": float(np.mean([r["win_rate"] for r in dqn_results.values()])),
        "avg_holding_days": float(np.mean([r["avg_holding_days"] for r in dqn_results.values()])),
        "trades_per_episode": float(np.mean([r["trades_per_episode"] for r in dqn_results.values()])),
    })
    if res_ensemble is not None:
        rows.append(metrics_row("dqn_ensemble_vote", res_ensemble))
    rows.append({
        "policy": "dqn_std_across_seeds",
        "sharpe": float(np.std(seed_sharpes)),
        "total_return_pct": float(np.std(seed_rets)),
        "max_drawdown": float(np.std([r["max_drawdown"] for r in dqn_results.values()])),
        "n_trades": float(np.std([r["n_trades"] for r in dqn_results.values()])),
        "win_rate": float(np.std([r["win_rate"] for r in dqn_results.values()])),
        "avg_holding_days": float(np.std([r["avg_holding_days"] for r in dqn_results.values()])),
        "trades_per_episode": float(np.std([r["trades_per_episode"] for r in dqn_results.values()])),
    })
    comparison = pd.DataFrame(rows)
    comparison.to_csv(RESULTS / "rl_comparison.csv", index=False)
    print(comparison.to_string(index=False))

    # --------------------------------------------------------- significance
    boot_rows = []
    for base_name, base_res in [("static_2sigma", res_static),
                                (f"static_tuned_{best_z}sigma", res_tuned)]:
        b = bootstrap_sharpe_diff(rl_daily_mean, base_res["daily"])
        b["comparison"] = f"dqn_mean_vs_{base_name}"
        boot_rows.append(b)
        if res_ensemble is not None:
            b = bootstrap_sharpe_diff(res_ensemble["daily"], base_res["daily"])
            b["comparison"] = f"dqn_ensemble_vs_{base_name}"
            boot_rows.append(b)
    boot = pd.DataFrame(boot_rows)
    boot.to_csv(RESULTS / "rl_bootstrap.csv", index=False)
    print(boot.to_string(index=False))

    # ----------------------------------------------------------- subperiods
    sub_rows = []
    sub_specs = [("static_2sigma", res_static["daily"]),
                 (f"static_tuned_{best_z}sigma", res_tuned["daily"]),
                 ("dqn_mean", rl_daily_mean)]
    if res_ensemble is not None:
        sub_specs.append(("dqn_ensemble", res_ensemble["daily"]))
    for name, daily in sub_specs:
        t = subperiod_table(daily, ["2022-01-01", "2024-01-01"])
        t.insert(0, "policy", name)
        sub_rows.append(t)
    pd.concat(sub_rows).to_csv(RESULTS / "rl_subperiods.csv", index=False)

    # ---------------------------------------------------------------- plots
    equity_figure(
        {"static_2sigma": res_static, "static_tuned": res_tuned,
         "tuned_label": f"static tuned ({best_z}σ)", "dqn_seeds": dqn_results,
         "ensemble": res_ensemble},
        rl_daily_mean,
    )
    behavior_figure(
        [r["trade_log"] for r in dqn_results.values()], res_static["trade_log"]
    )

    print(f"\nDone. Results written to {RESULTS}")


if __name__ == "__main__":
    main()
