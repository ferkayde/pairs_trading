# src/sizers.py
"""Fixed-cash position sizer as specified by the course (100,000 TL per trade)."""

import backtrader as bt


class FixedCashSizer(bt.Sizer):
    """Allocate a fixed cash amount per trade leg.

    The course requires each trade to be sized at 100,000 TL.
    For pair trades, this sizer is applied to EACH leg individually
    by passing the leg price directly (see PairsStrategy.next()).
    """

    params = (("cash_per_trade", 100_000),)

    def _getsizing(self, comminfo, cash, data, isbuy):
        close_price = data.close[0]
        if close_price <= 0:
            return 0
        return int(self.params.cash_per_trade / close_price)
