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
    )

    def __init__(self):
        self._in_long = False    # long leg1, short leg2
        self._in_short = False   # short leg1, long leg2

    def _spread(self) -> float:
        p1_star = self.data0.close[0] / self.p.p1_0
        p2_star = self.data1.close[0] / self.p.p2_0
        return p1_star - p2_star

    def next(self):
        spread = self._spread()
        threshold = self.p.entry_sigma * self.p.locked_sigma

        if not self._in_long and not self._in_short:
            p1 = self.data0.close[0]
            p2 = self.data1.close[0]
            if p1 <= 0 or p2 <= 0:
                return
            s1 = int(self.p.cash_per_leg / p1)
            s2 = int(self.p.cash_per_leg / p2)
            if s1 == 0 or s2 == 0:
                return

            if spread > threshold:
                # Leg1 is the outperformer → short leg1, long leg2
                self.sell(data=self.data0, size=s1)
                self.buy(data=self.data1, size=s2)
                self._in_short = True
            elif spread < -threshold:
                # Leg2 is the outperformer → long leg1, short leg2
                self.buy(data=self.data0, size=s1)
                self.sell(data=self.data1, size=s2)
                self._in_long = True

        elif self._in_long:
            # Entered when spread < -threshold; exit when spread crosses zero
            if spread >= 0.0:
                self.close(data=self.data0)
                self.close(data=self.data1)
                self._in_long = False

        elif self._in_short:
            # Entered when spread > +threshold; exit when spread crosses zero
            if spread <= 0.0:
                self.close(data=self.data0)
                self.close(data=self.data1)
                self._in_short = False

    def stop(self):
        # Time-stop: mark any remaining position at end of the trading window.
        # The order won't fill (no more bars) but broker.getvalue() still
        # reflects the mark-to-market so total return is computed correctly.
        if self._in_long or self._in_short:
            self.close(data=self.data0)
            self.close(data=self.data1)
            self._in_long = False
            self._in_short = False
