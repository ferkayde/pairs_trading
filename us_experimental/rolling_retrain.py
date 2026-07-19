"""
rolling_retrain.py — improvement 3: yearly walk-forward retraining.

The frozen-policy experiments showed a textbook non-stationarity pattern: the
DQN was the best policy in 2020-21, then decayed while trading with a model
last trained on pre-2015 data. This driver retrains every test year Y:

    train : episodes fully resolved before Y-2   (expanding window)
    val   : episodes inside [Y-2, Y)             (checkpoint selection)
    test  : episodes whose trading window STARTS in year Y

Windows straddling any boundary are dropped from train/val, so no test-year
information ever reaches training or model selection. Year Y's windows are
traded by year Y's seed models (and their majority-vote ensemble), exactly as
a practitioner rerunning the pipeline each January would do.

Anti-churn settings from the second experiment are kept (higher training
cost, minimum holding period). Baselines (static 2-sigma, validation-tuned
static, frozen-DQN ensemble from us_experimental/models/dqn_seed*.zip) are
evaluated on the identical test episodes.

Usage:
    python us_experimental/rolling_retrain.py            # 3 seeds x 200k/year
    python us_experimental/rolling_retrain.py --quick    # tiny smoke run
"""

from __future__ import annotations

import argparse
import sys
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

from us_experimental.rl_agents import EnsembleAgent, SB3Agent, StaticRuleAgent  # noqa: E402
from us_experimental.rl_episodes import Episode, load_episodes  # noqa: E402
from us_experimental.rl_evaluate import (  # noqa: E402
    bootstrap_sharpe_diff, evaluate_policy, rollout_episode, subperiod_table,
    summarize_rollouts, tune_static_entry,
)
from us_experimental.train_rl import train_dqn_seed  # noqa: E402

RESULTS = _HERE / "results"
CACHE = _HERE / "cache"
MODELS = _HERE / "models"


def rolling_year_splits(
    episodes: list[Episode],
    first_year: int,
    last_year: int,
    val_years: int = 2,
) -> list[dict]:
    """Per-test-year expanding-window splits; empty test years are skipped.

    For year Y: train = trading_end < (Y-val_years)-01-01;
    val = trading_start >= (Y-val_years)-01-01 and trading_end < Y-01-01;
    test = trading_start in year Y. Straddling windows are dropped from
    train/val by construction.
    """
    splits = []
    for year in range(first_year, last_year + 1):
        y0 = pd.Timestamp(f"{year}-01-01")
        y1 = pd.Timestamp(f"{year + 1}-01-01")
        v0 = pd.Timestamp(f"{year - val_years}-01-01")
        test = [e for e in episodes if y0 <= e.trading_start < y1]
        if not test:
            continue
        splits.append({
            "year": year,
            "train": [e for e in episodes if e.trading_end < v0],
            "val": [e for e in episodes
                    if e.trading_start >= v0 and e.trading_end < y0],
            "test": test,
        })
    return splits


def _collect_rollouts(episodes, agent, commission_bps, min_holding_days,
                      series_out: list, trades_out: list):
    """Roll `agent` over episodes, appending per-episode series/trades."""
    for ep in episodes:
        rets, trades = rollout_episode(ep, agent, commission_bps, min_holding_days)
        series_out.append(rets)
        for t in trades:
            t = dict(t)
            t["ticker1"], t["ticker2"] = ep.ticker1, ep.ticker2
            t["trading_start"] = ep.trading_start
            trades_out.append(t)


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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=3,
                    help="seeds retrained per year (default 3)")
    ap.add_argument("--timesteps", type=int, default=200_000)
    ap.add_argument("--eval-every", type=int, default=25_000)
    ap.add_argument("--commission", type=float, default=10.0)
    ap.add_argument("--train-commission-bps", type=float, default=25.0)
    ap.add_argument("--min-holding-days", type=int, default=5)
    ap.add_argument("--first-year", type=int, default=2020)
    ap.add_argument("--last-year", type=int, default=None,
                    help="default: last year with any test window")
    ap.add_argument("--val-years", type=int, default=2)
    ap.add_argument("--cache-file", default=None,
                    help="episode cache (default cache/episodes_full_roll21.pkl)")
    ap.add_argument("--quick", action="store_true",
                    help="smoke run: quick cache, 2 years, 1 seed, 10k steps")
    ap.add_argument("--resume", action="store_true",
                    help="reuse existing per-year seed checkpoints in "
                         "models/rolling/<year>/ and train only missing ones "
                         "(delete a checkpoint to force its retrain)")
    args = ap.parse_args()

    if args.quick:
        cache = CACHE / "episodes_quick_roll63.pkl"
        args.seeds, args.timesteps, args.eval_every = 1, 10_000, 5_000
        args.last_year = args.last_year or args.first_year + 1
    else:
        cache = Path(args.cache_file) if args.cache_file else (
            CACHE / "episodes_full_roll21.pkl")
    if not cache.exists():
        raise SystemExit(f"episode cache {cache} not found — run "
                         "run_experiment.py first (it builds the cache)")

    RESULTS.mkdir(exist_ok=True)
    episodes = load_episodes(cache)
    last_year = args.last_year or max(e.trading_start.year for e in episodes)
    splits = rolling_year_splits(episodes, args.first_year, last_year,
                                 args.val_years)
    print(f"Rolling retraining over {len(splits)} test years "
          f"({splits[0]['year']}-{splits[-1]['year']}), "
          f"{args.seeds} seeds x {args.timesteps} steps per year")

    cbps = args.commission
    seeds = list(range(args.seeds))
    seed_series: dict[int, list] = {s: [] for s in seeds}
    seed_trades: dict[int, list] = {s: [] for s in seeds}
    ens_series: list = []
    ens_trades: list = []
    history_rows: list[dict] = []
    test_all: list[Episode] = []

    from stable_baselines3 import DQN

    for sp in splits:
        year = sp["year"]
        print(f"=== year {year}: train {len(sp['train'])}, "
              f"val {len(sp['val'])}, test {len(sp['test'])} episodes ===",
              flush=True)
        year_dir = MODELS / "rolling" / str(year)
        infos = []
        for s in seeds:
            ckpt = year_dir / f"dqn_seed{s}.zip"
            if args.resume and ckpt.exists():
                print(f"  seed {s}: reusing existing checkpoint {ckpt.name}",
                      flush=True)
                infos.append({"seed": s, "model_path": ckpt, "history": []})
                continue
            print(f"--- training DQN seed {s} ---", flush=True)
            infos.append(train_dqn_seed(
                sp["train"], sp["val"], s,
                total_timesteps=args.timesteps, eval_every=args.eval_every,
                commission_bps=cbps, model_dir=year_dir,
                train_commission_bps=args.train_commission_bps,
                min_holding_days=args.min_holding_days,
            ))
        for si in infos:
            for h in si["history"]:
                history_rows.append(dict(h, seed=si["seed"], year=year))

        agents = {si["seed"]: SB3Agent(DQN.load(si["model_path"]))
                  for si in infos}
        for s in seeds:
            _collect_rollouts(sp["test"], agents[s], cbps,
                              args.min_holding_days,
                              seed_series[s], seed_trades[s])
        if len(agents) > 1:
            _collect_rollouts(sp["test"], EnsembleAgent(list(agents.values())),
                              cbps, args.min_holding_days,
                              ens_series, ens_trades)
        test_all.extend(sp["test"])

    pd.DataFrame(history_rows).to_csv(RESULTS / "rl_rolling_history.csv",
                                      index=False)

    # ------------------------------------------------------------- summaries
    n_test = len(test_all)
    seed_res = {s: summarize_rollouts(seed_series[s], seed_trades[s], n_test)
                for s in seeds}
    rows = []

    print("Baselines on the same test episodes ...")
    res_static = evaluate_policy(test_all, StaticRuleAgent(2.0), cbps)
    rows.append(metrics_row("static_2sigma", res_static))

    val_static = [e for e in episodes
                  if e.trading_start >= pd.Timestamp("2015-01-01")
                  and e.trading_end < pd.Timestamp(f"{args.first_year}-01-01")]
    best_z, _ = tune_static_entry(val_static, commission_bps=cbps)
    res_tuned = evaluate_policy(test_all, StaticRuleAgent(best_z), cbps)
    rows.append(metrics_row(f"static_tuned_{best_z}sigma", res_tuned))

    res_frozen = None
    frozen_paths = sorted(MODELS.glob("dqn_seed*.zip"))
    if frozen_paths:
        frozen = EnsembleAgent([SB3Agent(DQN.load(p)) for p in frozen_paths])
        res_frozen = evaluate_policy(test_all, frozen, cbps,
                                     min_holding_days=args.min_holding_days)
        rows.append(metrics_row("dqn_frozen_ensemble", res_frozen))

    for s in seeds:
        rows.append(metrics_row(f"rolling_dqn_seed{s}", seed_res[s]))
    sharpes = [seed_res[s]["sharpe"] for s in seeds]
    rows.append({
        "policy": "rolling_dqn_mean_across_seeds",
        "sharpe": float(np.mean(sharpes)),
        "total_return_pct": float(np.mean([seed_res[s]["total_return_pct"] for s in seeds])),
        "max_drawdown": float(np.mean([seed_res[s]["max_drawdown"] for s in seeds])),
        "n_trades": float(np.mean([seed_res[s]["n_trades"] for s in seeds])),
        "win_rate": float(np.mean([seed_res[s]["win_rate"] for s in seeds])),
        "avg_holding_days": float(np.mean([seed_res[s]["avg_holding_days"] for s in seeds])),
        "trades_per_episode": float(np.mean([seed_res[s]["trades_per_episode"] for s in seeds])),
    })
    rows.append({
        "policy": "rolling_dqn_std_across_seeds",
        "sharpe": float(np.std(sharpes)),
        "total_return_pct": float(np.std([seed_res[s]["total_return_pct"] for s in seeds])),
        "max_drawdown": float(np.std([seed_res[s]["max_drawdown"] for s in seeds])),
        "n_trades": float(np.std([seed_res[s]["n_trades"] for s in seeds])),
        "win_rate": float(np.std([seed_res[s]["win_rate"] for s in seeds])),
        "avg_holding_days": float(np.std([seed_res[s]["avg_holding_days"] for s in seeds])),
        "trades_per_episode": float(np.std([seed_res[s]["trades_per_episode"] for s in seeds])),
    })

    res_ens = None
    if ens_series:
        res_ens = summarize_rollouts(ens_series, ens_trades, n_test)
        rows.append(metrics_row("rolling_dqn_ensemble_vote", res_ens))

    comparison = pd.DataFrame(rows)
    comparison.to_csv(RESULTS / "rl_rolling_comparison.csv", index=False)
    print(comparison.to_string(index=False))

    # --------------------------------------------------------- significance
    headline = res_ens if res_ens is not None else seed_res[seeds[0]]
    boot_rows = []
    pairs = [("rolling_ens_vs_static_2sigma", headline, res_static),
             (f"rolling_ens_vs_static_tuned_{best_z}sigma", headline, res_tuned)]
    if res_frozen is not None:
        pairs.append(("rolling_ens_vs_frozen_ensemble", headline, res_frozen))
    for name, a, b in pairs:
        r = bootstrap_sharpe_diff(a["daily"], b["daily"])
        r["comparison"] = name
        boot_rows.append(r)
    boot = pd.DataFrame(boot_rows)
    boot.to_csv(RESULTS / "rl_rolling_bootstrap.csv", index=False)
    print(boot.to_string(index=False))

    # ----------------------------------------------------------- subperiods
    sub_specs = [("static_2sigma", res_static["daily"]),
                 (f"static_tuned_{best_z}sigma", res_tuned["daily"])]
    if res_frozen is not None:
        sub_specs.append(("dqn_frozen_ensemble", res_frozen["daily"]))
    if res_ens is not None:
        sub_specs.append(("rolling_dqn_ensemble", res_ens["daily"]))
    sub_rows = []
    for name, daily in sub_specs:
        t = subperiod_table(daily, ["2022-01-01", "2024-01-01"])
        t.insert(0, "policy", name)
        sub_rows.append(t)
    pd.concat(sub_rows).to_csv(RESULTS / "rl_rolling_subperiods.csv", index=False)

    # ---------------------------------------------------------------- plot
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(res_static["equity"], label="static 2σ", color="#1f77b4", lw=1.6)
    ax.plot(res_tuned["equity"], label=f"static tuned ({best_z}σ)",
            color="#2ca02c", lw=1.6)
    if res_frozen is not None:
        ax.plot(res_frozen["equity"], label="DQN frozen ensemble (pre-2015 model)",
                color="#ff7f0e", lw=1.8)
    if res_ens is not None:
        ax.plot(res_ens["equity"], label="DQN rolling ensemble (retrained yearly)",
                color="#d62728", lw=2.2)
    ax.axhline(1.0, color="#999999", lw=0.8, ls=":")
    ax.set_title("Rolling yearly retraining vs frozen policy and static rules "
                 "(test 2020+, net of costs)")
    ax.set_ylabel("cumulative value of $1 (per-pair-leg return basis)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "rl_rolling_equity.png", dpi=150)
    plt.close(fig)

    print(f"\nDone. Results written to {RESULTS}")


if __name__ == "__main__":
    main()
