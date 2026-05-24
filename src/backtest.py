# src/backtest.py
"""
PairFeed definition and run_backtest() helper.
"""

from __future__ import annotations

import backtrader as bt
import backtrader.feeds as btfeeds
import pandas as pd
from src.sizers import FixedCashSizer


class PairFeed(btfeeds.PandasData):
    """PandasData feed that carries ratio (close), Price1 and Price2 as lines.

    The DataFrame passed as `dataname` must have columns:
        Close   — Price1 / Price2 ratio
        Price1  — raw price of leg 1
        Price2  — raw price of leg 2

    Open/High/Low are mapped to the ratio value (= Close).  If the source
    DataFrame does not already carry these columns, :meth:`start` injects them
    automatically.  This is REQUIRED: Backtrader fills market orders at the next
    bar's open price, so if the open line were missing it would be NaN, every
    fill would execute at price=NaN, and the broker's portfolio value would
    become NaN (the original 0-trades / NaN-getvalue bug).
    """

    lines = ("price1", "price2")
    params = (
        ("datetime", None),   # index is the datetime
        ("open", "Open"),     # = Close so order fills have a valid price
        ("high", "High"),
        ("low", "Low"),
        ("close", "Close"),
        ("volume", -1),
        ("openinterest", -1),
        ("price1", "Price1"),
        ("price2", "Price2"),
    )

    def start(self):
        # Auto-populate OHLC from Close when absent so the feed is valid no
        # matter how it is constructed (tests, run_backtest, notebook, etc.).
        df = self.p.dataname
        if "Close" in df.columns:
            missing = [c for c in ("Open", "High", "Low") if c not in df.columns]
            if missing:
                df = df.copy()
                for col in missing:
                    df[col] = df["Close"]
                self.p.dataname = df
        super().start()


def _prepare_feed_df(pair_df: pd.DataFrame) -> pd.DataFrame:
    """Return a clean copy of ``pair_df`` safe to feed to :class:`PairFeed`.

    Drops any rows with NaN in the price lines used for execution.  (OHLC
    columns are injected automatically by :meth:`PairFeed.start`, so they are
    not required here.)  A NaN in Close/Price1/Price2 would otherwise produce a
    NaN execution price and poison the broker's portfolio value.
    """
    return pair_df.dropna(subset=["Close", "Price1", "Price2"]).copy()


def run_backtest(
    pair_df: pd.DataFrame,
    strategy_cls,
    initial_cash: float = 10_000_000,
    cash_per_trade: float = 100_000,
    **strategy_params,
) -> dict:
    """Run a single Backtrader backtest for one pair.

    Parameters
    ----------
    pair_df       : pd.DataFrame  Output of construct_pair().
    strategy_cls  : bt.Strategy subclass
    initial_cash  : float  Starting portfolio cash in TL.  Defaults to
                   10,000,000 — a larger notional than the course's nominal
                   1,000,000 TL is needed so the long+short legs of the ratio
                   instrument never trip a margin rejection.  Returns are still
                   reported as a percentage, so the result is unaffected.
    cash_per_trade: float  Per-leg trade size in TL.
    **strategy_params : forwarded to strategy.

    Returns
    -------
    dict with keys: sharpe, max_drawdown_pct, total_return_pct,
                    n_trades, final_value, equity_curve
    """
    feed_df = _prepare_feed_df(pair_df)

    cerebro = bt.Cerebro()
    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(commission=0)
    cerebro.broker.set_shortcash(False)
    # Cheat-on-close: fill orders at the current bar's (valid) close price
    # rather than waiting for the next bar's open.  Combined with the populated
    # OHLC columns this guarantees every fill has a finite execution price.
    cerebro.broker.set_coc(True)
    cerebro.addsizer(FixedCashSizer, cash_per_trade=cash_per_trade)

    feed = PairFeed(dataname=feed_df)
    cerebro.adddata(feed, name=pair_df.attrs.get("name", "pair"))

    cerebro.addstrategy(strategy_cls, **strategy_params)

    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe",
                        riskfreerate=0.0, annualize=True,
                        timeframe=bt.TimeFrame.Days)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="dd")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
    cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name="timereturn")

    result = cerebro.run()
    strat = result[0]

    sharpe_raw = strat.analyzers.sharpe.get_analysis().get("sharperatio", None)
    dd_raw = strat.analyzers.dd.get_analysis()
    trade_raw = strat.analyzers.trades.get_analysis()
    time_return_raw = strat.analyzers.timereturn.get_analysis()

    # Defensive: if the broker still reports NaN for any reason, fall back to
    # cash so the returned metrics stay finite.
    try:
        fv = cerebro.broker.getvalue()
        final_value = fv if not pd.isna(fv) else cerebro.broker.getcash()
    except Exception:
        final_value = cerebro.broker.getcash()

    return {
        "sharpe": sharpe_raw if sharpe_raw is not None else 0.0,
        "max_drawdown_pct": dd_raw.get("max", {}).get("drawdown", 0.0),
        "total_return_pct": (final_value / initial_cash - 1) * 100,
        "n_trades": trade_raw.get("total", {}).get("closed", 0),
        "final_value": final_value,
        "equity_curve": pd.Series(time_return_raw),
    }
