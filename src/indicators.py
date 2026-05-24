# src/indicators.py
"""
Backtrader indicators for pairs trading z-scores.
"""

import backtrader as bt


class DistanceZScore(bt.Indicator):
    """Rolling z-score of the Price1/Price2 ratio (Distance Approach).

    Lines:
        zscore — standardized deviation of the spread from its rolling mean.

    params:
        period (int): Formation window in bars. Default 60.
    """

    lines = ("zscore",)
    params = (("period", 60),)

    def __init__(self):
        ratio = self.data.price1 / self.data.price2
        sma = bt.indicators.SMA(ratio, period=self.p.period)
        std = bt.indicators.StdDev(ratio, period=self.p.period)
        self.lines.zscore = (ratio - sma) / std


class CointZScore(bt.Indicator):
    """Rolling OLS residual z-score (Cointegration Approach).

    Uses Backtrader's built-in OLS_TransformationN which provides
    the 'zscore' line directly (residual / rolling std of residuals).

    Lines:
        zscore — z-score of the current OLS residual.

    params:
        period (int): Formation window in bars. Default 60.
    """

    lines = ("zscore",)
    params = (("period", 60),)

    def __init__(self):
        ols = bt.indicators.OLS_TransformationN(
            self.data.price1,
            self.data.price2,
            period=self.p.period,
        )
        self.lines.zscore = ols.zscore
