"""
cross_listing.py — Idea 3: BIST vs. cross-listed ADR/GDR parity arbitrage.

ideas.md §Idea 3 proposes trading the price gap between a Turkish stock on BIST
and its depositary receipt abroad.  The LSE GDR tickers in the original idea are
not available on Yahoo Finance, but **Turkcell** has a liquid NYSE ADR (`TKC`,
full history 2012→), so this module implements the idea as a clean single-pair
case study: TCELL.IS (BIST, TRY) vs. TKC (NYSE, USD).

Strategy (faithful to ideas.md §Idea 3):
  • theoretical_adr_usd = TCELL_try × ratio / USDTRY
        where `ratio` = ordinary shares per ADR (Turkcell ≈ 2.5; we also infer
        it empirically and validate against the documented value).
  • parity spread (%) = (TKC_actual − theoretical) / theoretical × 100
  • Enter when |spread| > threshold:
        spread > 0  → ADR rich  → short TKC, long the BIST leg
        spread < 0  → ADR cheap → long TKC, short the BIST leg
  • Exit at the next parity crossing (spread sign flip), or at a time-stop.
  • P&L is tracked in USD; the BIST leg carries TRY→USD FX risk by construction.

Unavoidable caveats (reported in the notebook, not hidden):
  • Single pair → a case study, not a portfolio result.
  • BIST closes ~5 h before NYSE, so same-date closes are not synchronous; the
    measured spread contains a stale-quote component.
  • High FX beta: a sharp TRY move during a holding period can swamp the gap.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _close_series(ticker: str, start: str) -> pd.Series:
    """Download one adjusted-close Series from Yahoo, flattening MultiIndex."""
    import yfinance as yf

    d = yf.download(ticker, start=start, auto_adjust=True, progress=False)["Close"]
    if isinstance(d, pd.DataFrame):
        d = d.iloc[:, 0]
    return pd.Series(d.values, index=pd.DatetimeIndex(d.index), name=ticker)


def load_cross_listing_data(
    bist_ticker: str = "TCELL.IS",
    adr_ticker: str = "TKC",
    fx_ticker: str = "USDTRY=X",
    start: str = "2012-01-01",
    cache_path: str | None = "data/cross_listing.csv",
) -> pd.DataFrame:
    """Load (and cache) the three aligned series for the cross-listing pair.

    Returns a DataFrame indexed by date with columns
    [bist_try, adr_usd, usdtry], inner-joined on common trading days.
    If `cache_path` exists it is read; otherwise data is downloaded and saved.
    """
    if cache_path is not None and Path(cache_path).exists():
        return pd.read_csv(cache_path, index_col=0, parse_dates=True)

    bist = _close_series(bist_ticker, start)
    adr = _close_series(adr_ticker, start)
    fx = _close_series(fx_ticker, start)
    df = pd.concat([bist, adr, fx], axis=1)
    df.columns = ["bist_try", "adr_usd", "usdtry"]
    df = df.dropna()

    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path)
    return df


def infer_adr_ratio(df: pd.DataFrame) -> float:
    """Empirical ordinary-shares-per-ADR ratio = median(adr_usd × USDTRY / bist_try).

    For a correctly-priced ADR this equals the contractual conversion ratio.
    Returning the median makes it robust to the transient mispricings we trade.
    """
    implied = df["adr_usd"] * df["usdtry"] / df["bist_try"]
    return float(implied.median())


def compute_parity_spread(df: pd.DataFrame, ratio: float) -> pd.DataFrame:
    """Add theoretical ADR price and the parity spread (%) to the frame.

    theoretical_adr_usd = bist_try × ratio / USDTRY
    spread_pct          = (adr_usd − theoretical) / theoretical × 100
    bist_usd            = theoretical (the BIST leg expressed in USD per ADR)
    Positive spread → the ADR trades rich vs. its BIST underlying.
    """
    out = df.copy()
    out["bist_usd"] = out["bist_try"] * ratio / out["usdtry"]
    out["adr_theo_usd"] = out["bist_usd"]
    out["spread_pct"] = (out["adr_usd"] - out["adr_theo_usd"]) / out["adr_theo_usd"] * 100.0
    return out


def backtest_parity(
    data: pd.DataFrame,
    entry_threshold_pct: float = 1.5,
    commission_bps: float = 10.0,
    fx_cost_bps: float = 5.0,
    max_holding_days: int = 60,
    signal_lag: int = 1,
    return_trade_log: bool = False,
) -> dict:
    """Backtest the parity-convergence trade in USD terms.

    Both legs are one USD of notional (dollar-neutral long-short):
      • adr_usd      — the NYSE ADR price (USD).
      • bist_usd     — the BIST underlying expressed in USD per ADR
                        (= bist_try × ratio / USDTRY); carries TRY FX risk.

    Rules (ideas.md §Idea 3):
      enter when |spread_pct| > entry_threshold_pct
        spread > 0  → position = -1  (short ADR, long BIST)  profit if spread ↓
        spread < 0  → position = +1  (long ADR, short BIST)  profit if spread ↑
      exit at the next parity crossing (spread sign flip), or after
        `max_holding_days` (time-stop).

    Daily P&L uses GGR buy-and-hold-from-entry weighting: the denominator is the
    entry price of each leg, so the position is a fixed-notional long-short held
    until exit.  `commission_bps` is charged per leg per side; `fx_cost_bps` adds
    a one-off conversion cost on the BIST (TRY) leg at entry and exit.

    Returns dict: daily_returns, equity_curve, n_trades, trade_log (optional),
    plus summary stats (sharpe, total_return_pct, max_drawdown, win_rate,
    avg_holding_days, avg_entry_spread_pct).
    """
    from src.metrics import sharpe_ratio, max_drawdown

    adr = data["adr_usd"].values
    bist = data["bist_usd"].values
    spread = data["spread_pct"].values
    idx = data.index

    comm = commission_bps / 10_000.0
    fxc = fx_cost_bps / 10_000.0
    # round-trip cost charged on open and on close: 2 legs × commission + FX leg cost
    entry_cost = 2 * comm + fxc
    exit_cost = 2 * comm + fxc

    daily = np.zeros(len(data))
    position = 0
    entry_adr = entry_bist = None
    entry_bar = None
    entry_spread = None
    n_trades = 0
    trade_log: list = []

    def _close(i, reason):
        nonlocal position, entry_adr, entry_bist, entry_bar, entry_spread, n_trades
        daily[i] -= exit_cost
        if return_trade_log and entry_bar is not None:
            # realised trade return ≈ sum of daily P&L over the holding span
            ret = float(daily[entry_bar:i + 1].sum())
            trade_log.append({
                "entry_date": str(idx[entry_bar].date()),
                "exit_date": str(idx[i].date()),
                "holding_days": i - entry_bar,
                "entry_spread_pct": float(entry_spread),
                "trade_return_pct": ret * 100.0,
                "reason": reason,
            })
        position = 0
        entry_adr = entry_bist = entry_bar = entry_spread = None
        n_trades += 1

    for i in range(len(data)):
        # 1) mark-to-market the open position (buy-and-hold from entry)
        if position != 0 and i > 0 and entry_adr is not None:
            r_adr = (adr[i] - adr[i - 1]) / entry_adr
            r_bist = (bist[i] - bist[i - 1]) / entry_bist
            # position=-1: short ADR/long BIST → +(r_bist - r_adr)
            # position=+1: long ADR/short BIST → +(r_adr - r_bist)
            daily[i] += position * (r_adr - r_bist)

        # Decisions use the spread observed `signal_lag` days ago but EXECUTE at
        # today's close.  signal_lag=0 trades on the same close that defined the
        # spread (look-ahead-prone — captures spurious reversion from the BIST/NYSE
        # non-synchronous close).  signal_lag=1 (default) is the realistic version:
        # you act on yesterday's observable gap, removing the stale-price artifact.
        if i < signal_lag:
            continue
        sig = spread[i - signal_lag]

        if position == 0:
            if sig > entry_threshold_pct:
                position = -1
                entry_adr, entry_bist, entry_bar, entry_spread = adr[i], bist[i], i, sig
                daily[i] -= entry_cost
            elif sig < -entry_threshold_pct:
                position = +1
                entry_adr, entry_bist, entry_bar, entry_spread = adr[i], bist[i], i, sig
                daily[i] -= entry_cost
        else:
            crossed = (position == -1 and sig <= 0) or (position == +1 and sig >= 0)
            timed_out = entry_bar is not None and (i - entry_bar) >= max_holding_days
            if crossed:
                _close(i, "parity")
            elif timed_out:
                _close(i, "time")

    if position != 0:
        _close(len(data) - 1, "end")

    rets = pd.Series(daily, index=idx)
    equity = (1 + rets).cumprod()

    wins = [t for t in trade_log if t["trade_return_pct"] > 0] if return_trade_log else []
    out = {
        "daily_returns": rets,
        "equity_curve": equity,
        "n_trades": n_trades,
        "sharpe": sharpe_ratio(rets.replace(0, np.nan).dropna()),
        "total_return_pct": float((equity.iloc[-1] - 1) * 100) if len(equity) else 0.0,
        "max_drawdown": max_drawdown(equity) if len(equity) else 0.0,
    }
    if return_trade_log:
        out["trade_log"] = trade_log
        out["win_rate"] = (len(wins) / len(trade_log) * 100) if trade_log else 0.0
        out["avg_holding_days"] = float(np.mean([t["holding_days"] for t in trade_log])) if trade_log else 0.0
        out["avg_entry_spread_pct"] = float(np.mean([abs(t["entry_spread_pct"]) for t in trade_log])) if trade_log else 0.0
    return out
