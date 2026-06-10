"""
experimental_engine.py — GGR walk-forward portfolio with two experimental hooks.

This is a thin extension of src.backtest.run_ggr_portfolio that adds the two
modifications required by the experimental ideas, while keeping everything else
(σ-scaling, ±2σ entry, zero-crossing exit, committed/fully-invested aggregation)
identical to the baseline replication.

  Idea 1 — TCMB shock filter:
      `shock_dates` blocks NEW pair entries on shock days.  Open positions run
      to their normal exit.  Implemented by passing a per-window blocked-date
      set into the pandas pair simulator.

  Idea 2 — Real (CPI-deflated) pair formation:
      `real_prices` is used *only* for SSD ranking and σ-locking in the
      formation window.  Trading still uses nominal prices and nominal σ —
      exactly as ideas.md specifies (real prices are a selection lens, not a
      change to how the spread is traded).

Both hooks reuse the same fast pandas pair simulator so the shock filter can
be injected at the bar level.  The simulator here is a copy of
src.backtest.simulate_pair_returns with one extra argument: `blocked_dates`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Make `src` importable when the notebook runs from bist_experimental/.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.pairs import (  # noqa: E402
    normalize_prices, compute_ssd, select_top_pairs,
    compute_locked_sigma, liquidity_filter, activity_filter,
)
from src.metrics import sharpe_ratio, max_drawdown  # noqa: E402


def simulate_pair_returns_filtered(
    prices1: pd.Series,
    prices2: pd.Series,
    p1_0: float,
    p2_0: float,
    locked_sigma: float,
    entry_sigma: float = 2.0,
    commission_bps: float = 10.0,
    blocked_dates: set | None = None,
    stop_sigma: float | None = None,
    return_trade_log: bool = False,
) -> tuple:
    """Pandas GGR pair simulation with optional entry-block + 3σ stop-loss.

    Identical to src.backtest.simulate_pair_returns (no one-day delay branch),
    plus two optional hooks:
      • `blocked_dates` (Idea 1): a fresh entry signal is skipped when the bar's
        date is in the set.  Open positions are unaffected.
      • `stop_sigma` (Idea 5): if set, an OPEN position is closed for a loss the
        moment |spread| >= stop_sigma × locked_sigma (e.g. 3σ).  This is checked
        BEFORE the zero-crossing test, and only the exit rule changes — entry,
        sizing and σ are unchanged from GGR.

    trade_log entries gain an "exit_reason" ∈ {"converge","stop","time"}.

    Returns (daily_returns, n_trades[, trade_log]).
    """
    commission = commission_bps / 10_000.0
    threshold = entry_sigma * locked_sigma
    stop_level = (stop_sigma * locked_sigma) if stop_sigma is not None else None
    blocked = blocked_dates or set()

    daily_returns = pd.Series(0.0, index=prices1.index)
    position = 0
    n_trades = 0
    entry_p1 = entry_p2 = entry_bar = entry_spread = None
    trade_log: list = []
    broken = False   # set once a 3σ stop fires → no re-entry this window (Idea 5)

    dates = prices1.index

    for i in range(len(prices1)):
        p1 = prices1.iloc[i]
        p2 = prices2.iloc[i]
        if pd.isna(p1) or pd.isna(p2) or p1 <= 0 or p2 <= 0:
            continue

        spread = p1 / p1_0 - p2 / p2_0

        # Mark-to-market P&L for an open position (buy-and-hold from entry).
        if position != 0 and i > 0 and entry_p1 is not None:
            prev_p1, prev_p2 = prices1.iloc[i - 1], prices2.iloc[i - 1]
            if not pd.isna(prev_p1) and not pd.isna(prev_p2) and prev_p1 > 0 and prev_p2 > 0:
                r1 = (p1 - prev_p1) / entry_p1
                r2 = (p2 - prev_p2) / entry_p2
                daily_returns.iloc[i] += position * (r1 - r2)

        if position == 0:
            # No re-entry once the pair has been stopped out as broken (Idea 5).
            if broken:
                continue
            # New entry — blocked on shock days (Idea 1).
            if dates[i] in blocked:
                continue
            if spread > threshold:
                position = -1
                entry_p1, entry_p2, entry_bar, entry_spread = p1, p2, i, abs(spread)
                daily_returns.iloc[i] -= 2 * commission
            elif spread < -threshold:
                position = 1
                entry_p1, entry_p2, entry_bar, entry_spread = p1, p2, i, abs(spread)
                daily_returns.iloc[i] -= 2 * commission
        else:
            converged = (position == 1 and spread >= 0) or (position == -1 and spread <= 0)
            stopped = stop_level is not None and abs(spread) >= stop_level
            # 3σ stop takes priority over convergence (Idea 5): a position this
            # far from the mean is treated as structurally broken, closed at loss.
            if stopped or converged:
                daily_returns.iloc[i] -= 2 * commission
                if return_trade_log and entry_bar is not None:
                    trade_log.append({
                        "entry_spread_pct": float(entry_spread * 100) if entry_spread else 0.0,
                        "holding_days": i - entry_bar,
                        "exit_reason": "stop" if stopped else "converge",
                    })
                position = 0
                entry_p1 = entry_p2 = entry_bar = entry_spread = None
                n_trades += 1
                if stopped:
                    broken = True   # pair is structurally broken; stop trading it

    if position != 0:
        daily_returns.iloc[-1] -= 2 * commission
        if return_trade_log and entry_bar is not None:
            trade_log.append({
                "entry_spread_pct": float(entry_spread * 100) if entry_spread else 0.0,
                "holding_days": len(prices1) - 1 - entry_bar,
                "exit_reason": "time",
            })
        n_trades += 1

    if return_trade_log:
        return daily_returns, n_trades, trade_log
    return daily_returns, n_trades


def run_ggr_experimental(
    prices: pd.DataFrame,
    real_prices: pd.DataFrame | None = None,
    shock_dates: pd.DatetimeIndex | None = None,
    formation_days: int = 252,
    trading_days: int = 126,
    roll_days: int = 21,
    top_n: int = 20,
    entry_sigma: float = 2.0,
    commission_bps: float = 10.0,
    pair_offset: int = 0,
    sigma_mode: str = "rescale",
    stop_sigma: float | None = None,
    verbose: bool = True,
) -> dict:
    """GGR walk-forward portfolio with optional real-price formation + shock filter.

    Parameters
    ----------
    prices       : nominal adjusted-close prices (columns = tickers).
    real_prices  : CPI-deflated prices, SAME shape/index as `prices`.  When
                   provided, SSD ranking and σ-locking use these (Idea 2);
                   trading still uses nominal `prices`.  When None → baseline.
    shock_dates  : DatetimeIndex of dates on which new entries are blocked
                   (Idea 1).  When None → no filter.

    All other parameters match src.backtest.run_ggr_portfolio.

    Returns the same dict shape as run_ggr_portfolio, plus 'all_trade_logs'
    (a flat list of {entry_spread_pct, holding_days} across every window) so
    win-rate / holding-period analysis can be done downstream.
    """
    blocked_set = set(pd.DatetimeIndex(shock_dates)) if shock_dates is not None else set()

    all_dates = prices.index
    n_total = len(all_dates)
    window_series: list = []
    window_stats: list[dict] = []
    all_trade_logs: list[dict] = []

    n_windows_total = max(1, (n_total - formation_days - trading_days) // roll_days + 1)
    window_num = 0

    start_idx = 0
    while start_idx + formation_days + trading_days <= n_total:
        f_start = all_dates[start_idx]
        f_end = all_dates[start_idx + formation_days - 1]
        t_start = all_dates[start_idx + formation_days]
        t_end = all_dates[min(start_idx + formation_days + trading_days - 1, n_total - 1)]

        window_num += 1
        if verbose:
            print(f"  Window {window_num}/{n_windows_total}  "
                  f"formation {f_start.date()} -> {f_end.date()}  "
                  f"trading {t_start.date()} -> {t_end.date()}", end="\r", flush=True)

        # 1. Liquidity + activity filter on NOMINAL prices (universe definition
        #    must not depend on the deflator).
        liquid = liquidity_filter(prices, f_start, f_end)
        liquid = activity_filter(prices.loc[f_start:f_end], liquid)
        if len(liquid) < 4:
            start_idx += roll_days
            continue

        # 2. Formation normalization — on REAL prices if given (Idea 2), else nominal.
        form_src = real_prices if real_prices is not None else prices
        norm_form = normalize_prices(form_src.loc[f_start:f_end, liquid])

        # 3. SSD ranking + top pairs (on the chosen formation basis).
        ssd_df = compute_ssd(norm_form)
        available = max(0, len(ssd_df) - pair_offset)
        top_pairs = select_top_pairs(ssd_df, n=min(top_n, available), offset=pair_offset)

        # 4. Lock σ on the SAME basis used for ranking, so entry threshold is
        #    consistent with the formation lens.  (For Idea 2 this is real-σ; the
        #    spread traded is still nominal, but the entry band scales with the
        #    formation-window dispersion, matching GGR's design.)
        sigmas = compute_locked_sigma(norm_form, top_pairs)
        p0_row = form_src.loc[f_start]
        nominal_p0 = prices.loc[f_start]

        # 5. Trade each pair on NOMINAL prices over the trading window.
        pair_series: list[pd.Series] = []
        n_active = 0
        n_trades_total = 0
        window_tlogs: list[dict] = []

        for t1, t2, _ in top_pairs:
            sigma = sigmas.get((t1, t2))
            if sigma is None or sigma < 1e-10:
                continue
            # Normalization base for the TRADED (nominal) spread.
            p1_0 = nominal_p0.get(t1, np.nan)
            p2_0 = nominal_p0.get(t2, np.nan)
            if pd.isna(p1_0) or pd.isna(p2_0) or p1_0 <= 0 or p2_0 <= 0:
                continue
            tp1 = prices.loc[t_start:t_end, t1] if t1 in prices.columns else None
            tp2 = prices.loc[t_start:t_end, t2] if t2 in prices.columns else None
            if tp1 is None or tp2 is None:
                continue
            common = tp1.dropna().index.intersection(tp2.dropna().index)
            if len(common) < 5:
                continue

            # Idea 2: pairs were RANKED on real prices, but the spread is TRADED
            # on nominal prices.  `sigma_mode` controls how the ±2σ entry band is
            # set when real_prices is given (ablation knob to test robustness):
            #   "rescale" — real-σ scaled into nominal-spread units (default).
            #   "real"    — real-σ used as-is (band in real-spread units).
            #   "nominal" — σ re-locked on NOMINAL spread; real prices only
            #               affect WHICH pairs are picked, not the band. This is
            #               the purest "real prices = selection lens only" reading.
            if real_prices is None:
                sigma_trade = sigma
            elif sigma_mode == "nominal":
                nom_form = normalize_prices(prices.loc[f_start:f_end, [t1, t2]])
                sigma_trade = float((nom_form[t1] - nom_form[t2]).std())
            elif sigma_mode == "real":
                sigma_trade = sigma
            else:  # "rescale"
                sigma_trade = _rescale_sigma_to_nominal(
                    sigma, prices, real_prices, t1, t2, f_start, f_end,
                )
            if sigma_trade is None or sigma_trade < 1e-10:
                continue

            rets, n_t, tlog = simulate_pair_returns_filtered(
                tp1.loc[common], tp2.loc[common],
                p1_0, p2_0, sigma_trade, entry_sigma, commission_bps,
                blocked_dates=blocked_set, stop_sigma=stop_sigma,
                return_trade_log=True,
            )
            if len(rets) == 0:
                continue
            pair_series.append(rets)
            n_active += 1
            n_trades_total += n_t
            window_tlogs.extend(tlog)

        if not pair_series:
            start_idx += roll_days
            continue

        trading_idx = prices.loc[t_start:t_end].index
        pair_matrix = pd.DataFrame(pair_series).T.reindex(trading_idx, fill_value=0.0)
        window_fully_invested = pair_matrix.mean(axis=1)
        while len(pair_matrix.columns) < top_n:
            pair_matrix[f"_pad_{len(pair_matrix.columns)}"] = 0.0
        window_committed = pair_matrix.mean(axis=1)
        window_series.append((window_committed, window_fully_invested))

        all_trade_logs.extend(window_tlogs)

        avg_spread = float(np.mean([t["entry_spread_pct"] for t in window_tlogs])) if window_tlogs else 0.0
        avg_holding = float(np.mean([t["holding_days"] for t in window_tlogs])) if window_tlogs else 0.0

        window_stats.append({
            "formation_start": str(f_start.date()),
            "trading_start": str(t_start.date()),
            "trading_end": str(t_end.date()),
            "n_liquid": len(liquid),
            "n_active_pairs": n_active,
            "n_trades": n_trades_total,
            "avg_spread_at_entry_pct": round(avg_spread, 4),
            "avg_holding_days": round(avg_holding, 2),
            "window_return_pct": round(float(window_committed.sum()) * 100, 3),
            "top_pairs": [(t1, t2) for t1, t2, _ in top_pairs],
        })

        start_idx += roll_days

    if verbose:
        print(f"\n  Done: {len(window_series)} windows completed.")

    if not window_series:
        empty = pd.Series(dtype=float)
        return {
            "portfolio_returns": empty, "equity_curve": empty,
            "fully_invested_returns": empty, "fully_invested_equity": empty,
            "n_windows": 0, "sharpe": 0.0, "fully_invested_sharpe": 0.0,
            "max_drawdown": 0.0, "total_return_pct": 0.0,
            "fully_invested_return_pct": 0.0, "window_stats": [],
            "all_trade_logs": [], "total_trades": 0,
        }

    committed_series = [ws[0] for ws in window_series]
    fully_inv_series = [ws[1] for ws in window_series]

    def _aggregate(series_list):
        all_idx = pd.DatetimeIndex(sorted({d for s in series_list for d in s.index}))
        total = pd.Series(0.0, index=all_idx)
        cnt = pd.Series(0, index=all_idx)
        for ws in series_list:
            c = all_idx.intersection(ws.index)
            total[c] += ws[c]
            cnt[c] += 1
        rets = total.div(cnt.replace(0, np.nan)).fillna(0.0)
        eq = (1 + rets).cumprod()
        return rets, eq

    port_rets, equity_curve = _aggregate(committed_series)
    fi_rets, fi_equity = _aggregate(fully_inv_series)

    return {
        "portfolio_returns": port_rets,
        "equity_curve": equity_curve,
        "fully_invested_returns": fi_rets,
        "fully_invested_equity": fi_equity,
        "n_windows": len(window_series),
        "sharpe": sharpe_ratio(port_rets.replace(0, np.nan).dropna()),
        "fully_invested_sharpe": sharpe_ratio(fi_rets.replace(0, np.nan).dropna()),
        "max_drawdown": max_drawdown(equity_curve),
        "total_return_pct": float((equity_curve.iloc[-1] - 1) * 100),
        "fully_invested_return_pct": float((fi_equity.iloc[-1] - 1) * 100),
        "window_stats": window_stats,
        "all_trade_logs": all_trade_logs,
        "total_trades": int(sum(w["n_trades"] for w in window_stats)),
    }


def _rescale_sigma_to_nominal(
    real_sigma: float,
    prices: pd.DataFrame,
    real_prices: pd.DataFrame,
    t1: str, t2: str,
    f_start, f_end,
) -> float:
    """Convert a real-price-locked σ into nominal-spread units.

    The spread traded in the trading window is the NOMINAL normalized spread
    P_t/P_0 (nominal).  But σ was locked on the REAL normalized spread.  To keep
    the ±2σ entry band economically equivalent, scale real-σ by the ratio
    (nominal-spread std / real-spread std) measured over the formation window:

        σ_nominal = real_sigma × std(nominal_spread) / std(real_spread)

    Both stds use the same P_t/P_0 normalization on their respective price
    series, so the ratio is dimensionless and ≈1 in low-inflation periods.
    """
    nom = normalize_prices(prices.loc[f_start:f_end, [t1, t2]])
    rl = normalize_prices(real_prices.loc[f_start:f_end, [t1, t2]])
    nom_std = float((nom[t1] - nom[t2]).std())
    rl_std = float((rl[t1] - rl[t2]).std())
    if rl_std < 1e-12:
        return real_sigma
    return real_sigma * nom_std / rl_std


def pair_overlap(stats_a: list[dict], stats_b: list[dict]) -> pd.DataFrame:
    """Per-window overlap between two strategies' selected top pairs.

    Returns a DataFrame with one row per matched trading window:
    [trading_start, n_pairs_a, n_pairs_b, n_overlap, overlap_frac].
    """
    by_start_b = {w["trading_start"]: w for w in stats_b}
    rows = []
    for wa in stats_a:
        wb = by_start_b.get(wa["trading_start"])
        if wb is None:
            continue
        set_a = {frozenset(p) for p in wa.get("top_pairs", [])}
        set_b = {frozenset(p) for p in wb.get("top_pairs", [])}
        inter = set_a & set_b
        denom = max(len(set_a), 1)
        rows.append({
            "trading_start": wa["trading_start"],
            "n_pairs_a": len(set_a),
            "n_pairs_b": len(set_b),
            "n_overlap": len(inter),
            "overlap_frac": len(inter) / denom,
        })
    return pd.DataFrame(rows)
