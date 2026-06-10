# src/strategies.py
"""
GGRStrategy — Backtrader implementation of Gatev, Goetzmann & Rouwenhorst (2006).
"""

import backtrader as bt


class GGRStrategy(bt.Strategy):
    """Pairs trading strategy strictly following GGR (2006).

    Requires exactly TWO data feeds added to the Cerebro instance:
        data0 (self.data)  — leg 1 (stock A)
        data1              — leg 2 (stock B)

    The spread is defined in normalized price space:
        spread_t = P*_A_t - P*_B_t,   P*_i_t = P_i_t / P_i_0

    where P_i_0 is the price at the START of the formation period.

    Entry (GGR §2):
        spread_t >  entry_sigma × locked_sigma  →  short A, long  B
        spread_t < -entry_sigma × locked_sigma  →  long  A, short B

    Exit (GGR §2):
        Convergence : close immediately when spread crosses zero.
        Time-stop   : remaining position is closed at end of data.

    Commission is applied at the broker level — set it in the calling code via
    cerebro.broker.setcommission(commission=0.001) for 10 bps.

    params
    ------
    locked_sigma  float  Formation-period σ of (P*_A - P*_B). Required.
    p1_0          float  Price of leg1 at formation start.
    p2_0          float  Price of leg2 at formation start.
    entry_sigma   float  Entry multiplier (paper: 2.0).
    cash_per_leg  float  TL allocated per leg (for position sizing).
    """

    params = (
        ("locked_sigma", 1.0),
        ("p1_0", 1.0),
        ("p2_0", 1.0),
        ("entry_sigma", 2.0),
        ("cash_per_leg", 1_000_000),
        # GGR Panel B: delay entry/exit by one trading day to avoid bid-ask bounce.
        # Signal fires on day t; order executes at day t+1 close.
        ("wait_one_day", False),
    )

    def __init__(self):
        self._in_long = False
        self._in_short = False
        self._pending_entry = 0    # +1 or -1: direction queued for next bar
        self._pending_exit = False # exit queued for next bar
        # Pair-level round-trip log for GGR Table 2 statistics.
        # One dict per round trip: entry_spread_pct (|spread|×100 at open)
        # and holding_days (bars between open and close).
        self.trade_log: list = []
        self._entry_bar = None
        self._entry_spread = None

    def _spread(self) -> float:
        p1_star = self.data0.close[0] / self.p.p1_0
        p2_star = self.data1.close[0] / self.p.p2_0
        return p1_star - p2_star

    def _execute_entry(self, direction: int) -> None:
        p1 = self.data0.close[0]
        p2 = self.data1.close[0]
        if p1 <= 0 or p2 <= 0:
            return
        s1 = int(self.p.cash_per_leg / p1)
        s2 = int(self.p.cash_per_leg / p2)
        if s1 == 0 or s2 == 0:
            return
        if direction == -1:
            self.sell(data=self.data0, size=s1)
            self.buy(data=self.data1, size=s2)
            self._in_short = True
        else:
            self.buy(data=self.data0, size=s1)
            self.sell(data=self.data1, size=s2)
            self._in_long = True
        self._entry_bar = len(self)
        self._entry_spread = abs(self._spread())

    def _log_close(self) -> None:
        if self._entry_bar is not None:
            self.trade_log.append({
                "entry_spread_pct": float(self._entry_spread * 100)
                                    if self._entry_spread else 0.0,
                "holding_days": len(self) - self._entry_bar,
            })
        self._entry_bar = None
        self._entry_spread = None

    def next(self):
        spread = self._spread()
        threshold = self.p.entry_sigma * self.p.locked_sigma

        # Execute orders that were queued on the previous bar (Panel B delay)
        if self._pending_exit:
            self.close(data=self.data0)
            self.close(data=self.data1)
            self._log_close()
            self._in_long = False
            self._in_short = False
            self._pending_exit = False
            return  # don't generate new signals on the same bar as exit fill

        if self._pending_entry != 0:
            self._execute_entry(self._pending_entry)
            self._pending_entry = 0
            return  # don't check exit on the same bar as entry fill

        # Exit logic
        if self._in_long:
            if spread >= 0.0:
                if self.p.wait_one_day:
                    self._pending_exit = True
                else:
                    self.close(data=self.data0)
                    self.close(data=self.data1)
                    self._log_close()
                    self._in_long = False
            return

        if self._in_short:
            if spread <= 0.0:
                if self.p.wait_one_day:
                    self._pending_exit = True
                else:
                    self.close(data=self.data0)
                    self.close(data=self.data1)
                    self._log_close()
                    self._in_short = False
            return

        # Entry logic (flat and no pending order)
        if spread > threshold:
            if self.p.wait_one_day:
                self._pending_entry = -1
            else:
                self._execute_entry(-1)
        elif spread < -threshold:
            if self.p.wait_one_day:
                self._pending_entry = 1
            else:
                self._execute_entry(1)

    def stop(self):
        if self._in_long or self._in_short or self._pending_exit:
            self.close(data=self.data0)
            self.close(data=self.data1)
            self._log_close()
            self._in_long = False
            self._in_short = False
            self._pending_exit = False


class CointStrategy(GGRStrategy):
    """Cointegration-method pairs trading — Engle-Granger spread, Backtrader.

    Identical trade mechanics to GGRStrategy (entry/exit/time-stop, Panel B
    delay, trade log); only the SPREAD DEFINITION differs:

        spread_t = ln(P_A_t) − β·ln(P_B_t) − α

    where α and β come from the formation-period Engle-Granger first-stage
    OLS  ln(P_A) = α + β·ln(P_B) + ε.  The residual spread is stationary for
    a cointegrated pair, so:

        Entry : |spread_t| > entry_sigma × locked_sigma
                (locked_sigma = formation-period std of the EG residual)
        Exit  : spread crosses zero (its formation mean, since α absorbs it)

    Position sizing stays dollar-neutral $1:$1 like GGRStrategy so that the
    distance and cointegration methods are compared on the same capital
    basis; β enters only through the signal, not the position weights.

    Extra params (on top of GGRStrategy's)
    --------------------------------------
    beta   float  EG hedge ratio from formation OLS.
    alpha  float  EG intercept from formation OLS.
    """

    params = (
        ("beta", 1.0),
        ("alpha", 0.0),
    )

    def _spread(self) -> float:
        import math
        p1 = self.data0.close[0]
        p2 = self.data1.close[0]
        if p1 <= 0 or p2 <= 0:
            return 0.0
        return math.log(p1) - self.p.beta * math.log(p2) - self.p.alpha
