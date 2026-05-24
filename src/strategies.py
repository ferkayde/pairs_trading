# src/strategies.py
"""
PairsStrategy — Backtrader strategy supporting Distance and Cointegration approaches.
"""

import backtrader as bt
from src.indicators import DistanceZScore, CointZScore


class PairsStrategy(bt.Strategy):
    """Long-short pairs trading strategy.

    params:
        period       (int)   : Formation window for z-score. Default 60.
        entry_z      (float) : |z-score| threshold to open position. Default 2.0.
        exit_z       (float) : |z-score| threshold to close position. Default 1.0.
        approach     (str)   : 'distance' or 'coint'. Default 'distance'.
        wait_reversal(bool)  : Wait for z-score to start reverting before entry.
                               Default False.
        cash_per_leg (float) : Cash per trade leg. Default 100_000.

    Trade logic:
        LONG  pair: z < -entry_z  -> buy spread (buy leg1 proportion)
        SHORT pair: z >  entry_z  -> sell spread
        Close long : z > -exit_z
        Close short: z <  exit_z
    """

    params = (
        ("period", 60),
        ("entry_z", 2.0),
        ("exit_z", 1.0),
        ("approach", "distance"),
        ("wait_reversal", False),
        ("cash_per_leg", 100_000),
    )

    def __init__(self):
        if self.p.approach == "distance":
            self.zscore_ind = DistanceZScore(self.data, period=self.p.period)
        else:
            self.zscore_ind = CointZScore(self.data, period=self.p.period)

        self._in_long = False
        self._in_short = False

    def _get_sizes(self):
        p1 = self.data.price1[0]
        p2 = self.data.price2[0]
        size1 = int(self.p.cash_per_leg / p1) if p1 > 0 else 0
        size2 = int(self.p.cash_per_leg / p2) if p2 > 0 else 0
        return size1, size2

    def _should_enter(self, z: float, direction: str) -> bool:
        if direction == "long":
            crossed = z < -self.p.entry_z
        else:
            crossed = z > self.p.entry_z

        if not crossed:
            return False

        if self.p.wait_reversal:
            prev_z = self.zscore_ind.zscore[-1]
            if direction == "long":
                return z > prev_z
            else:
                return z < prev_z

        return True

    def next(self):
        z = self.zscore_ind.zscore[0]
        size1, size2 = self._get_sizes()

        if size1 == 0 or size2 == 0:
            return

        if not self._in_long and not self._in_short:
            if self._should_enter(z, "long"):
                self.buy(size=size1)
                self._in_long = True
            elif self._should_enter(z, "short"):
                self.sell(size=size1)
                self._in_short = True

        elif self._in_long:
            if z > -self.p.exit_z:
                self.close()
                self._in_long = False

        elif self._in_short:
            if z < self.p.exit_z:
                self.close()
                self._in_short = False
