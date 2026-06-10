# src/backtest.py
"""
GGR (2006) backtesting: Backtrader single-pair engine and pandas portfolio simulator.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import backtrader as bt
import backtrader.feeds as btfeeds

from src.strategies import GGRStrategy


def run_ggr_pair_backtest(
    trading_prices1: pd.Series,
    trading_prices2: pd.Series,
    p1_0: float,
    p2_0: float,
    locked_sigma: float,
    pair_name: str = "pair",
    entry_sigma: float = 2.0,
    commission_bps: float = 10.0,
    cash_per_leg: float = 1_000_000,
    wait_one_day: bool = False,
) -> dict:
    """Backtrader backtest of GGRStrategy for one pair over one trading window.

    Trades two actual stock legs — true dollar-neutral long-short — using the
    locked formation σ and zero-crossing exit rule of GGR (2006).

    Parameters
    ----------
    trading_prices1/2 : Adjusted close prices during the TRADING window only.
    p1_0, p2_0        : Prices at FORMATION START (normalization base P_i_0).
    locked_sigma      : Formation-period σ of (P*_A - P*_B).
    pair_name         : Label used in Cerebro data names.
    entry_sigma       : Entry threshold multiplier (GGR: 2.0).
    commission_bps    : Bps per leg per trade side. Default 10 bps (BIST realistic).
    cash_per_leg      : TL per leg; also sets position sizing via GGRStrategy.

    Returns
    -------
    dict: sharpe, max_drawdown_pct, total_return_pct, n_trades, final_value, equity_curve
    """
    def _ohlcv(prices: pd.Series) -> pd.DataFrame:
        df = pd.DataFrame({
            "Open": prices, "High": prices, "Low": prices,
            "Close": prices, "Volume": 1.0,
        })
        return df.dropna()

    df1 = _ohlcv(trading_prices1)
    df2 = _ohlcv(trading_prices2)
    common = df1.index.intersection(df2.index)
    df1, df2 = df1.loc[common], df2.loc[common]

    if len(common) < 5:
        return {
            "sharpe": 0.0, "max_drawdown_pct": 0.0, "total_return_pct": 0.0,
            "n_trades": 0, "final_value": cash_per_leg * 4,
            "equity_curve": pd.Series(dtype=float),
        }

    initial_cash = cash_per_leg * 4

    cerebro = bt.Cerebro()
    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(commission=commission_bps / 10_000.0)
    cerebro.broker.set_shortcash(False)
    cerebro.broker.set_coc(True)

    cerebro.adddata(btfeeds.PandasData(dataname=df1), name=f"{pair_name}_leg1")
    cerebro.adddata(btfeeds.PandasData(dataname=df2), name=f"{pair_name}_leg2")

    cerebro.addstrategy(
        GGRStrategy,
        locked_sigma=locked_sigma,
        p1_0=p1_0,
        p2_0=p2_0,
        entry_sigma=entry_sigma,
        cash_per_leg=cash_per_leg,
        wait_one_day=wait_one_day,
    )

    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe",
                        riskfreerate=0.0, annualize=True, timeframe=bt.TimeFrame.Days)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="dd")
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name="timereturn")

    result = cerebro.run()
    strat = result[0]

    sharpe_raw = strat.analyzers.sharpe.get_analysis().get("sharperatio", None)
    dd_raw = strat.analyzers.dd.get_analysis()
    time_return_raw = strat.analyzers.timereturn.get_analysis()
    # Pair round trips from the strategy's own log — TradeAnalyzer counts the
    # two legs of one pair position as two separate trades.
    n_round_trips = len(getattr(strat, "trade_log", []))

    try:
        fv = cerebro.broker.getvalue()
        final_value = fv if not pd.isna(fv) else initial_cash
    except Exception:
        final_value = initial_cash

    return {
        "sharpe": sharpe_raw if sharpe_raw is not None else 0.0,
        "max_drawdown_pct": dd_raw.get("max", {}).get("drawdown", 0.0),
        "total_return_pct": (final_value / initial_cash - 1) * 100,
        "n_trades": n_round_trips,
        "final_value": final_value,
        "equity_curve": pd.Series(time_return_raw),
    }


def simulate_pair_returns(
    prices1: pd.Series,
    prices2: pd.Series,
    p1_0: float,
    p2_0: float,
    locked_sigma: float,
    entry_sigma: float = 2.0,
    commission_bps: float = 10.0,
    wait_one_day: bool = False,
    return_trade_log: bool = False,
) -> tuple:
    """Pandas simulation of one pair's returns over its trading window.

    Implements GGR rules: locked-σ entry, zero-crossing exit, time-stop.
    Faster than Backtrader; used for portfolio-level sensitivity sweeps.

    Return computation — GGR §2.3 equations (2)–(3):
        Positions are buy-and-hold from the entry bar: daily P&L uses the
        ENTRY price as denominator, not the previous day's price.  This
        matches the paper's value-weighted mark-to-market convention where
        w_{i,t} = P_{i,t} / P_{i,entry}.

    wait_one_day (GGR Panel B):
        Signal on day t → execute on day t+1. The return difference between
        Panel A (wait_one_day=False) and Panel B estimates the bid-ask spread.

    Returns
    -------
    (daily_returns, n_trades)  — or (daily_returns, n_trades, trade_log) when
    return_trade_log=True.  trade_log is a list of dicts with keys
    'entry_spread_pct' (|spread|×100 at open) and 'holding_days'.
    """
    commission = commission_bps / 10_000.0
    threshold = entry_sigma * locked_sigma
    daily_returns = pd.Series(0.0, index=prices1.index)
    position = 0        # 0=flat, 1=long leg1/short leg2, -1=short leg1/long leg2
    n_trades = 0
    entry_p1 = None     # price of leg1 at position open (buy-and-hold denominator)
    entry_p2 = None
    entry_bar = None    # bar index when position opened (for holding-day calc)
    entry_spread = None # |spread| at entry (for Table 2 avg-trigger stat)
    trade_log: list = []

    # One-day delay queues (Panel B)
    pending_entry = 0   # +1 or -1 direction queued for next bar
    pending_exit = False

    for i in range(len(prices1)):
        p1 = prices1.iloc[i]
        p2 = prices2.iloc[i]
        if pd.isna(p1) or pd.isna(p2) or p1 <= 0 or p2 <= 0:
            continue

        spread = p1 / p1_0 - p2 / p2_0

        # Step 1 — P&L for current bar while in position (buy-and-hold from entry)
        if position != 0 and i > 0 and entry_p1 is not None:
            prev_p1, prev_p2 = prices1.iloc[i - 1], prices2.iloc[i - 1]
            if not pd.isna(prev_p1) and not pd.isna(prev_p2) and prev_p1 > 0 and prev_p2 > 0:
                # GGR eq.(3): weight = P_{t-1}/P_entry; daily return = ΔP/P_entry
                r1 = (p1 - prev_p1) / entry_p1
                r2 = (p2 - prev_p2) / entry_p2
                daily_returns.iloc[i] += position * (r1 - r2)

        # Step 2 — Execute pending exit (day after convergence signal, Panel B)
        if pending_exit:
            daily_returns.iloc[i] -= 2 * commission
            if return_trade_log and entry_bar is not None:
                trade_log.append({
                    "entry_spread_pct": float(entry_spread * 100) if entry_spread else 0.0,
                    "holding_days": i - entry_bar,
                })
            position = 0
            entry_p1 = entry_p2 = entry_bar = entry_spread = None
            n_trades += 1
            pending_exit = False
            continue

        # Step 3 — Execute pending entry (day after divergence signal, Panel B)
        if pending_entry != 0:
            position = pending_entry
            entry_p1, entry_p2 = p1, p2
            entry_bar = i
            entry_spread = abs(spread)
            daily_returns.iloc[i] -= 2 * commission
            pending_entry = 0
            continue

        # Step 4 — Signal generation
        if position == 0:
            if spread > threshold:
                if wait_one_day:
                    pending_entry = -1
                else:
                    position = -1
                    entry_p1, entry_p2 = p1, p2
                    entry_bar = i
                    entry_spread = abs(spread)
                    daily_returns.iloc[i] -= 2 * commission
            elif spread < -threshold:
                if wait_one_day:
                    pending_entry = 1
                else:
                    position = 1
                    entry_p1, entry_p2 = p1, p2
                    entry_bar = i
                    entry_spread = abs(spread)
                    daily_returns.iloc[i] -= 2 * commission
        else:
            if (position == 1 and spread >= 0) or (position == -1 and spread <= 0):
                if wait_one_day:
                    pending_exit = True
                else:
                    daily_returns.iloc[i] -= 2 * commission
                    if return_trade_log and entry_bar is not None:
                        trade_log.append({
                            "entry_spread_pct": float(entry_spread * 100) if entry_spread else 0.0,
                            "holding_days": i - entry_bar,
                        })
                    position = 0
                    entry_p1 = entry_p2 = entry_bar = entry_spread = None
                    n_trades += 1

    # Time-stop: force close at end of window
    if position != 0 or pending_exit:
        daily_returns.iloc[-1] -= 2 * commission
        if return_trade_log and entry_bar is not None:
            trade_log.append({
                "entry_spread_pct": float(entry_spread * 100) if entry_spread else 0.0,
                "holding_days": len(prices1) - 1 - entry_bar,
            })
        n_trades += 1

    if return_trade_log:
        return daily_returns, n_trades, trade_log
    return daily_returns, n_trades


def _bt_pair_returns(
    trading_prices1: pd.Series,
    trading_prices2: pd.Series,
    p1_0: float,
    p2_0: float,
    locked_sigma: float,
    entry_sigma: float = 2.0,
    commission_bps: float = 10.0,
    cash_per_leg: float = 1_000_000,
    wait_one_day: bool = False,
    strategy_cls=None,
    extra_params: dict | None = None,
) -> tuple:
    """Lightweight Backtrader run for the portfolio loop.

    Runs GGRStrategy (or `strategy_cls`, e.g. CointStrategy with
    extra_params={'beta': ..., 'alpha': ...}) on one pair over one trading
    window.  Returns (daily_returns: pd.Series, n_trades: int, trade_log: list).

    Skips the heavy SharpeRatio and DrawDown analyzers used in
    run_ggr_pair_backtest — only TimeReturn is attached, keeping per-call
    overhead low across hundreds of windows × 20 pairs.

    n_trades counts PAIR round trips (from the strategy's trade_log), not
    Backtrader's per-leg trade count — closing one pair position closes two
    legs, which TradeAnalyzer would count as two trades.

    Return basis: TimeReturn measures P&L relative to total broker value
    (initial cash = 4 × cash_per_leg), while GGR — and simulate_pair_returns —
    express the pair's P&L per $1 of one leg.  The TimeReturn series is
    therefore rescaled by (4 × cash_per_leg) / cash_per_leg = 4 so both
    engines share the GGR convention and are directly comparable to the
    paper's Table 1 levels.
    """
    def _ohlcv(prices: pd.Series) -> pd.DataFrame:
        df = pd.DataFrame({
            "Open": prices, "High": prices, "Low": prices,
            "Close": prices, "Volume": 1.0,
        })
        return df.dropna()

    df1 = _ohlcv(trading_prices1)
    df2 = _ohlcv(trading_prices2)
    common = df1.index.intersection(df2.index)
    df1, df2 = df1.loc[common], df2.loc[common]

    if len(common) < 5:
        return pd.Series(dtype=float), 0, []

    cerebro = bt.Cerebro()
    cerebro.broker.setcash(cash_per_leg * 4)
    cerebro.broker.setcommission(commission=commission_bps / 10_000.0)
    cerebro.broker.set_shortcash(False)
    cerebro.broker.set_coc(True)

    cerebro.adddata(btfeeds.PandasData(dataname=df1), name="leg1")
    cerebro.adddata(btfeeds.PandasData(dataname=df2), name="leg2")

    cerebro.addstrategy(
        strategy_cls or GGRStrategy,
        locked_sigma=locked_sigma,
        p1_0=p1_0,
        p2_0=p2_0,
        entry_sigma=entry_sigma,
        cash_per_leg=cash_per_leg,
        wait_one_day=wait_one_day,
        **(extra_params or {}),
    )

    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name="timereturn")

    result = cerebro.run()
    strat = result[0]

    time_return_raw = strat.analyzers.timereturn.get_analysis()
    trade_log = list(getattr(strat, "trade_log", []))
    n_trades = len(trade_log)

    ec = pd.Series(time_return_raw)
    if len(ec) > 0:
        ec.index = pd.DatetimeIndex([pd.Timestamp(d) for d in ec.index])
        # Rescale from P&L/(4×cash_per_leg) to P&L/cash_per_leg — GGR's
        # per-$1-of-one-leg convention (see docstring).
        ec = ec * 4.0

    return ec, n_trades, trade_log


def run_ggr_portfolio(  # noqa: PLR0912, PLR0915
    prices: pd.DataFrame,
    formation_days: int = 252,
    trading_days: int = 126,
    roll_days: int = 21,
    top_n: int = 20,
    entry_sigma: float = 2.0,
    commission_bps: float = 10.0,
    cash_per_leg: float = 1_000_000,
    pair_offset: int = 0,
    use_backtrader: bool = True,
    wait_one_day: bool = False,
    method: str = "distance",
    coint_pval: float = 0.05,
    coint_candidates: int = 100,
    verbose: bool = True,
) -> dict:
    """Full GGR (2006) walk-forward overlapping portfolio simulation.

    method="distance" (default) is the GGR SSD rule.  method="cointegration"
    replaces pair selection and the trading spread with the Engle-Granger
    procedure: pairs are pre-ranked by SSD (top `coint_candidates`), screened
    by the EG test (p < coint_pval), and traded on the EG residual spread
    ln(P_A) − β·ln(P_B) − α via CointStrategy.  The cointegration method
    always runs through Backtrader.

    Steps through history monthly. For each step:
        1. Liquidity filter — GGR criterion: complete price history in formation window.
        2. Normalize prices to P*_{i,t} = P_{i,t} / P_{i,0}.
        3. Exhaustive SSD over all N(N−1)/2 pairs; select Top-N.
        4. Lock σ = std(P*_A − P*_B) for each top pair.
        5. Simulate trading via Backtrader: enter at |spread| > 2σ,
           exit at zero-crossing/time-stop. One Cerebro instance per pair.

    Up to (trading_days / roll_days) portfolios run concurrently. The aggregate
    daily return is the equal-weighted average across all active windows
    (committed-capital convention: inactive pair slots contribute 0 return).

    Parameters
    ----------
    prices         : Adjusted close prices (columns = tickers).
    formation_days : 252 (12 months, per GGR).
    trading_days   : 126 (6 months, per GGR).
    roll_days      : 21 (1 month step, per GGR).
    top_n          : 20 (top pairs per portfolio, per GGR).
    entry_sigma    : 2.0 (per GGR).
    commission_bps : 10 bps per leg per trade side.
    cash_per_leg      : TL per leg for position sizing (default 1 000 000).
    pair_offset       : Start rank for pair selection (0 = Top-N, 100 = pairs 101-120).
                        GGR use offset=100 as a control group ("no longer profitable").
    use_backtrader    : If True (default) each pair is simulated via a Backtrader
                        Cerebro instance (GGRStrategy + TimeReturn analyzer).
                        Set False to use the fast pandas fallback
                        (simulate_pair_returns) — useful for sensitivity sweeps
                        where many runs are needed and the Backtrader overhead
                        would dominate.
    verbose           : Print window-level progress if True.

    Returns
    -------
    dict: portfolio_returns, equity_curve, n_windows, sharpe, max_drawdown,
          total_return_pct, window_stats
    """
    from src.pairs import (
        normalize_prices, compute_ssd, select_top_pairs,
        compute_locked_sigma, liquidity_filter, activity_filter,
        select_cointegrated_pairs,
    )
    from src.metrics import sharpe_ratio, max_drawdown
    from src.strategies import CointStrategy

    if method not in ("distance", "cointegration"):
        raise ValueError(f"unknown method {method!r}")
    if method == "cointegration":
        use_backtrader = True   # course requirement: trading runs in Backtrader

    all_dates = prices.index
    n_total = len(all_dates)
    window_series: list[pd.Series] = []
    window_stats: list[dict] = []

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

        # 1. Liquidity filter: complete price history in formation window (GGR §2)
        #    + activity filter: remove dormant stocks (>30% zero-return days).
        #    Yahoo Finance repeats stale quotes for non-trading stocks; CRSP
        #    recorded NaN, so GGR's universe implicitly excluded them.
        liquid = liquidity_filter(prices, f_start, f_end)
        liquid = activity_filter(prices.loc[f_start:f_end], liquid)

        if len(liquid) < 4:
            start_idx += roll_days
            continue

        # 2. Normalize formation window
        norm_form = normalize_prices(prices.loc[f_start:f_end, liquid])

        # 3. Exhaustive SSD + top pairs (with optional rank offset for control groups)
        ssd_df = compute_ssd(norm_form)
        eg_params: dict = {}
        if method == "cointegration":
            # SSD pre-rank → Engle-Granger screen → best p-values
            candidates = select_top_pairs(
                ssd_df, n=min(coint_candidates, len(ssd_df)), offset=pair_offset)
            top_pairs, eg_params = select_cointegrated_pairs(
                prices.loc[f_start:f_end, liquid], candidates,
                n=top_n, pval_threshold=coint_pval)
        else:
            available = max(0, len(ssd_df) - pair_offset)
            top_pairs = select_top_pairs(ssd_df, n=min(top_n, available), offset=pair_offset)

        # 4. Locked σ — formation SSD-spread std (distance) or EG residual std (coint)
        if method == "cointegration":
            sigmas = {k: v["sigma_eg"] for k, v in eg_params.items()}
        else:
            sigmas = compute_locked_sigma(norm_form, top_pairs)
        p0_row = prices.loc[f_start]

        # 5. Simulate each pair over trading window
        pair_series: list[pd.Series] = []
        n_active = 0
        n_trades_total = 0
        all_trade_logs: list[dict] = []

        for t1, t2, _ in top_pairs:
            sigma = sigmas.get((t1, t2))
            if sigma is None or sigma < 1e-10:
                continue
            p1_0 = p0_row.get(t1, np.nan)
            p2_0 = p0_row.get(t2, np.nan)
            if pd.isna(p1_0) or pd.isna(p2_0) or p1_0 <= 0 or p2_0 <= 0:
                continue
            tp1 = prices.loc[t_start:t_end, t1] if t1 in prices.columns else None
            tp2 = prices.loc[t_start:t_end, t2] if t2 in prices.columns else None
            if tp1 is None or tp2 is None:
                continue
            common = tp1.dropna().index.intersection(tp2.dropna().index)
            if len(common) < 5:
                continue

            if use_backtrader:
                if method == "cointegration":
                    eg = eg_params[(t1, t2)]
                    strategy_cls = CointStrategy
                    extra_params = {"beta": eg["beta"], "alpha": eg["alpha"]}
                else:
                    strategy_cls, extra_params = None, None
                rets, n_t, pair_tlog = _bt_pair_returns(
                    tp1.loc[common], tp2.loc[common],
                    p1_0, p2_0, sigma, entry_sigma, commission_bps, cash_per_leg,
                    wait_one_day=wait_one_day,
                    strategy_cls=strategy_cls, extra_params=extra_params,
                )
            else:
                rets, n_t, pair_tlog = simulate_pair_returns(
                    tp1.loc[common], tp2.loc[common],
                    p1_0, p2_0, sigma, entry_sigma, commission_bps,
                    wait_one_day=wait_one_day, return_trade_log=True,
                )

            if len(rets) == 0:
                continue
            # GGR §2.3: "fully invested" divides by pairs that actually OPEN a
            # position during the trading interval, "committed" by all top_n
            # slots.  A pair with price data but no trades belongs only to the
            # committed denominator.
            pair_series.append((rets, n_t > 0))
            if n_t > 0:
                n_active += 1
            n_trades_total += n_t
            all_trade_logs.extend(pair_tlog)

        if not pair_series:
            start_idx += roll_days
            continue

        # Committed capital: average over all top_n slots (zero-padded).
        # Fully invested: average only over pairs that opened a trade (GGR §2.3).
        trading_idx = prices.loc[t_start:t_end].index
        all_rets    = [r for r, _ in pair_series]
        traded_rets = [r for r, traded in pair_series if traded]

        pair_matrix = pd.DataFrame(all_rets).T.reindex(trading_idx, fill_value=0.0)
        while len(pair_matrix.columns) < top_n:
            pair_matrix[f"_pad_{len(pair_matrix.columns)}"] = 0.0
        window_committed = pair_matrix.mean(axis=1)

        if traded_rets:
            fi_matrix = pd.DataFrame(traded_rets).T.reindex(trading_idx, fill_value=0.0)
            window_fully_invested = fi_matrix.mean(axis=1)
        else:
            window_fully_invested = pd.Series(0.0, index=trading_idx)
        window_series.append((window_committed, window_fully_invested))

        # Aggregate trade-log statistics for Table 2
        avg_spread = (
            float(np.mean([t["entry_spread_pct"] for t in all_trade_logs]))
            if all_trade_logs else 0.0
        )
        avg_holding = (
            float(np.mean([t["holding_days"] for t in all_trade_logs]))
            if all_trade_logs else 0.0
        )
        round_trips_per_pair = (
            n_trades_total / n_active if n_active > 0 else 0.0
        )

        window_stats.append({
            "formation_start": str(f_start.date()),
            "trading_start": str(t_start.date()),
            "trading_end": str(t_end.date()),
            "n_liquid": len(liquid),
            "n_active_pairs": n_active,
            "n_trades": n_trades_total,
            "avg_spread_at_entry_pct": round(avg_spread, 4),
            "avg_holding_days": round(avg_holding, 2),
            "round_trips_per_pair": round(round_trips_per_pair, 3),
            "window_return_pct": round(float(window_committed.sum()) * 100, 3),
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
        }

    # Aggregate across all active windows (equal-weight) — both return bases
    committed_series   = [ws[0] for ws in window_series]
    fully_inv_series   = [ws[1] for ws in window_series]

    def _aggregate(series_list):
        all_idx = pd.DatetimeIndex(sorted({d for s in series_list for d in s.index}))
        total = pd.Series(0.0, index=all_idx)
        cnt   = pd.Series(0,   index=all_idx)
        for ws in series_list:
            c = all_idx.intersection(ws.index)
            total[c] += ws[c]
            cnt[c]   += 1
        rets = total.div(cnt.replace(0, np.nan)).fillna(0.0)
        eq   = (1 + rets).cumprod()
        return rets, eq

    port_rets,   equity_curve   = _aggregate(committed_series)
    fi_rets,     fi_equity      = _aggregate(fully_inv_series)

    # Sharpe is computed on the full daily series, zeros included: under the
    # committed-capital convention a flat day is a real 0% return, and the
    # equity curve / max drawdown already use the full series.  Dropping zeros
    # (former behaviour) inflated the daily mean and made the Sharpe
    # inconsistent with the equity curve it is reported next to.
    return {
        "portfolio_returns": port_rets,
        "equity_curve": equity_curve,
        "fully_invested_returns": fi_rets,
        "fully_invested_equity": fi_equity,
        "n_windows": len(window_series),
        "sharpe": sharpe_ratio(port_rets),
        "fully_invested_sharpe": sharpe_ratio(fi_rets),
        "max_drawdown": max_drawdown(equity_curve),
        "total_return_pct": float((equity_curve.iloc[-1] - 1) * 100),
        "fully_invested_return_pct": float((fi_equity.iloc[-1] - 1) * 100),
        "window_stats": window_stats,
    }
