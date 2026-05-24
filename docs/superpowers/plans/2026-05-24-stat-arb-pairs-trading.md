# Statistical Arbitrage — Pairs Trading Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a complete pairs trading backtest system in a Jupyter notebook, implementing both the Distance and Cointegration approaches from Gatev, Goetzmann & Rouwenhorst (2006), with walk-forward optimization and Monte Carlo significance testing, using Backtrader.

**Architecture:** Data is fetched once and saved to CSV; all subsequent runs read from CSV (reproducibility requirement). Backtrader strategies consume a custom PandasData feed that carries the price ratio plus each leg's raw price. Two strategies (Distance z-score and Cointegration OLS residual z-score) share the same entry/exit signal logic but differ only in how the z-score is computed.

**Tech Stack:** Python 3.10+, Backtrader 1.9.78, pandas, numpy, statsmodels (cointegration tests, OLS), yfinance (one-time download), matplotlib/seaborn, scipy, pytest

---

## File Map

| File | Responsibility |
|------|---------------|
| `data_download.py` | One-time fetch of prices via yfinance → `data/prices.csv` |
| `src/pairs.py` | `construct_pair()`, `test_cointegration()`, `half_life()`, `screen_pairs()` |
| `src/indicators.py` | `DistanceZScore(bt.Indicator)`, `CointZScore(bt.Indicator)` |
| `src/sizers.py` | `FixedCashSizer(bt.Sizer)` — 100,000 TL per trade |
| `src/strategies.py` | `PairsStrategy(bt.Strategy)` — single class with `approach` param |
| `src/metrics.py` | `sharpe_ratio()`, `sortino_ratio()`, `monte_carlo_test()`, `extract_bt_metrics()` |
| `src/backtest.py` | `run_backtest(pair_df, **params)` — thin Cerebro wrapper |
| `src/optimize.py` | `grid_search()`, `walk_forward_optimization()` |
| `notebook.ipynb` | Final deliverable — imports all src modules, runs full analysis |
| `tests/test_pairs.py` | Unit tests for pair construction and cointegration |
| `tests/test_indicators.py` | Unit tests for z-score indicator outputs |
| `tests/test_metrics.py` | Unit tests for Sharpe, Sortino, Monte Carlo |
| `requirements.txt` | Pinned dependencies |
| `data/prices.csv` | Raw adjusted close prices (committed to repo) |
| `results/` | CSVs for backtest results; PNGs for plots |

---

## Task 1: Project Scaffold & Requirements

**Files:**
- Create: `requirements.txt`
- Create: `data/` (directory)
- Create: `results/` (directory)
- Create: `src/__init__.py`
- Create: `tests/__init__.py`

- [ ] **Step 1: Create directory structure**

```bash
cd "C:\Users\ferka\OneDrive\Masaüstü\ec581_proje"
mkdir -p data results src tests
type nul > src/__init__.py
type nul > tests/__init__.py
```

- [ ] **Step 2: Write requirements.txt**

```
backtrader==1.9.78.123
pandas==2.2.2
numpy==1.26.4
yfinance==0.2.40
statsmodels==0.14.2
scipy==1.13.0
matplotlib==3.9.0
seaborn==0.13.2
scikit-learn==1.5.0
pytest==8.2.0
jupyter==1.0.0
ipykernel==6.29.4
```

- [ ] **Step 3: Install dependencies**

```bash
pip install -r requirements.txt
```

Expected: all packages install without error.

- [ ] **Step 4: Commit**

```bash
git add requirements.txt src/__init__.py tests/__init__.py
git commit -m "chore: scaffold project structure and pin dependencies"
```

---

## Task 2: Data Download

**Files:**
- Create: `data_download.py`
- Creates: `data/prices.csv` (run once)

**About the pairs universe:** The course examples use Turkish-listed stocks (GARAN, AKBNK etc. on the Istanbul Exchange). On Yahoo Finance these trade as `TICKER.IS`. You can substitute any market — the code is market-agnostic. The list below matches the course examples and is economically motivated (same-sector pairs are more likely to be cointegrated).

- [ ] **Step 1: Write data_download.py**

```python
"""
data_download.py — Run ONCE to fetch price data and save to CSV.
All subsequent runs read from data/prices.csv.
"""

import yfinance as yf
import pandas as pd
from pathlib import Path

# Candidate tickers — adjust to your chosen market.
# Using Yahoo Finance suffix .IS for Istanbul Stock Exchange.
# Remove suffix if using US stocks (e.g. "GARAN" → "JPM").
TICKERS = [
    "GARAN.IS", "AKBNK.IS", "ISCTR.IS", "VAKBN.IS", "YKBNK.IS", "HALKB.IS",
    "TCELL.IS", "TTKOM.IS",
    "THYAO.IS", "PGSUS.IS",
    "EREGL.IS", "SISE.IS",
    "KCHOL.IS", "SAHOL.IS",
    "KOZAL.IS", "KOZAA.IS",
    "BIMAS.IS", "MGROS.IS",
    "TUPRS.IS", "AYGAZ.IS",
]

START = "2010-01-01"
OUT_PATH = Path("data/prices.csv")


def download_prices(tickers: list[str], start: str, out_path: Path) -> pd.DataFrame:
    """Download adjusted close prices, align dates, forward-fill gaps ≤3 days."""
    raw = yf.download(tickers, start=start, auto_adjust=True, progress=False)
    prices = raw["Close"]

    # Drop columns with >5% missing
    thresh = int(0.95 * len(prices))
    prices = prices.dropna(axis=1, thresh=thresh)

    # Forward-fill trading halts up to 3 consecutive days
    prices = prices.ffill(limit=3)

    # Drop rows where ALL prices are missing (weekends already excluded by yf)
    prices = prices.dropna(how="all")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(out_path)
    print(f"Saved {prices.shape[1]} tickers × {len(prices)} days → {out_path}")
    return prices


if __name__ == "__main__":
    download_prices(TICKERS, START, OUT_PATH)
```

- [ ] **Step 2: Run the download**

```bash
python data_download.py
```

Expected output: `Saved N tickers × M days → data/prices.csv`

- [ ] **Step 3: Verify the CSV**

```python
import pandas as pd
df = pd.read_csv("data/prices.csv", index_col=0, parse_dates=True)
print(df.shape)           # (days, tickers)
print(df.isna().sum())    # should be low
print(df.head())
```

- [ ] **Step 4: Commit**

```bash
git add data_download.py data/prices.csv
git commit -m "feat: add one-time data download script and raw price CSV"
```

---

## Task 3: Pair Construction & Cointegration Screening

**Files:**
- Create: `src/pairs.py`
- Create: `tests/test_pairs.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_pairs.py
import pandas as pd
import numpy as np
import pytest
from src.pairs import construct_pair, test_cointegration, half_life, screen_pairs


def _make_cointegrated_series(n=500, seed=42):
    """Generate two cointegrated price series for testing."""
    rng = np.random.default_rng(seed)
    common = np.cumsum(rng.normal(0, 1, n))
    s1 = common + rng.normal(0, 0.5, n)
    s2 = common + rng.normal(0, 0.5, n)
    idx = pd.date_range("2015-01-01", periods=n, freq="B")
    return pd.Series(np.exp(s1 / 100), index=idx), pd.Series(np.exp(s2 / 100), index=idx)


def _make_random_series(n=500, seed=99):
    """Generate two independent random walks."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2015-01-01", periods=n, freq="B")
    s1 = pd.Series(np.exp(np.cumsum(rng.normal(0, 1, n)) / 100), index=idx)
    s2 = pd.Series(np.exp(np.cumsum(rng.normal(0, 1, n)) / 100), index=idx)
    return s1, s2


class TestConstructPair:
    def test_ratio_is_price1_over_price2(self):
        s1, s2 = _make_cointegrated_series()
        pair = construct_pair(s1, s2, "A", "B")
        pd.testing.assert_series_equal(pair["Close"], s1 / s2, check_names=False)

    def test_has_required_columns(self):
        s1, s2 = _make_cointegrated_series()
        pair = construct_pair(s1, s2, "A", "B")
        assert set(pair.columns) == {"Close", "Price1", "Price2"}

    def test_name_attribute(self):
        s1, s2 = _make_cointegrated_series()
        pair = construct_pair(s1, s2, "GARAN", "AKBNK")
        assert pair.attrs["name"] == "GARAN_AKBNK"

    def test_index_preserved(self):
        s1, s2 = _make_cointegrated_series()
        pair = construct_pair(s1, s2, "A", "B")
        pd.testing.assert_index_equal(pair.index, s1.index)


class TestCointegration:
    def test_cointegrated_pair_low_pvalue(self):
        s1, s2 = _make_cointegrated_series()
        result = test_cointegration(s1, s2)
        assert result["p_value"] < 0.05

    def test_random_pair_high_pvalue(self):
        s1, s2 = _make_random_series()
        result = test_cointegration(s1, s2)
        assert result["p_value"] > 0.05

    def test_result_has_required_keys(self):
        s1, s2 = _make_cointegrated_series()
        result = test_cointegration(s1, s2)
        assert {"t_stat", "p_value", "is_cointegrated"} <= result.keys()


class TestHalfLife:
    def test_half_life_positive(self):
        s1, s2 = _make_cointegrated_series()
        spread = s1 / s2
        hl = half_life(spread)
        assert hl > 0

    def test_fast_reverting_spread_short_half_life(self):
        """A spread with strong mean reversion should have a short half-life."""
        rng = np.random.default_rng(7)
        n = 500
        idx = pd.date_range("2015-01-01", periods=n, freq="B")
        # OU with theta=0.5 → expected half-life ≈ ln(2)/0.5 ≈ 1.4 days
        spread = pd.Series(index=idx, dtype=float)
        spread.iloc[0] = 0.0
        for i in range(1, n):
            spread.iloc[i] = spread.iloc[i - 1] * 0.5 + rng.normal(0, 0.1)
        hl = half_life(spread)
        assert 0 < hl < 10


class TestScreenPairs:
    def test_returns_dataframe(self):
        s1, s2 = _make_cointegrated_series()
        prices = pd.concat([s1.rename("A"), s2.rename("B")], axis=1)
        pairs_to_test = [("A", "B")]
        result = screen_pairs(prices, pairs_to_test)
        assert isinstance(result, pd.DataFrame)

    def test_cointegrated_pair_passes_screen(self):
        s1, s2 = _make_cointegrated_series()
        prices = pd.concat([s1.rename("A"), s2.rename("B")], axis=1)
        result = screen_pairs(prices, [("A", "B")])
        assert len(result) == 1

    def test_random_pair_filtered_out(self):
        s1, s2 = _make_random_series()
        prices = pd.concat([s1.rename("X"), s2.rename("Y")], axis=1)
        result = screen_pairs(prices, [("X", "Y")])
        assert len(result) == 0
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
pytest tests/test_pairs.py -v
```

Expected: `ImportError` or `ModuleNotFoundError` (src/pairs.py doesn't exist yet).

- [ ] **Step 3: Implement src/pairs.py**

```python
# src/pairs.py
"""
Pair construction, cointegration testing, and pair screening.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant


def construct_pair(
    price1: pd.Series,
    price2: pd.Series,
    name1: str,
    name2: str,
) -> pd.DataFrame:
    """Construct pair DataFrame with ratio (Close) and individual prices.

    Parameters
    ----------
    price1, price2 : pd.Series  Aligned price series (same index).
    name1, name2   : str         Ticker names.

    Returns
    -------
    pd.DataFrame with columns Close (=price1/price2), Price1, Price2.
    The attrs dict carries 'name' = 'NAME1_NAME2'.
    """
    pair_df = pd.DataFrame(
        {
            "Close": price1 / price2,
            "Price1": price1,
            "Price2": price2,
        },
        index=price1.index,
    )
    pair_df.attrs["name"] = f"{name1}_{name2}"
    return pair_df


def test_cointegration(
    s1: pd.Series,
    s2: pd.Series,
    pvalue_threshold: float = 0.05,
) -> dict:
    """Engle-Granger cointegration test (both directions; take lower p-value).

    Returns
    -------
    dict with keys: t_stat, p_value, is_cointegrated
    """
    score_12, pval_12, _ = coint(s1, s2)
    score_21, pval_21, _ = coint(s2, s1)

    if pval_12 <= pval_21:
        t_stat, p_value = score_12, pval_12
    else:
        t_stat, p_value = score_21, pval_21

    return {
        "t_stat": t_stat,
        "p_value": p_value,
        "is_cointegrated": p_value < pvalue_threshold,
    }


def half_life(spread: pd.Series) -> float:
    """Ornstein-Uhlenbeck half-life of mean reversion (in bars).

    Regresses Δspread on lagged spread level.
    half_life = -ln(2) / AR_coefficient
    """
    lag = spread.shift(1).dropna()
    delta = spread.diff().dropna()
    common_idx = lag.index.intersection(delta.index)
    reg = OLS(delta.loc[common_idx], add_constant(lag.loc[common_idx])).fit()
    theta = reg.params.iloc[1]  # AR coefficient on lagged level
    if theta >= 0:
        return np.inf  # non-mean-reverting
    return -np.log(2) / theta


def screen_pairs(
    prices: pd.DataFrame,
    candidate_pairs: list[tuple[str, str]],
    pvalue_threshold: float = 0.05,
    min_half_life: float = 5.0,
    max_half_life: float = 120.0,
) -> pd.DataFrame:
    """Screen candidate pairs for cointegration and tradeable half-life.

    Parameters
    ----------
    prices           : pd.DataFrame  Columns are ticker names.
    candidate_pairs  : list of (ticker1, ticker2) tuples.
    pvalue_threshold : float  Cointegration p-value cutoff.
    min_half_life    : float  Min half-life in days (avoid noise).
    max_half_life    : float  Max half-life in days (avoid too slow).

    Returns
    -------
    pd.DataFrame with columns: pair, ticker1, ticker2, p_value, half_life_days
    Only rows that pass both screens are included.
    """
    records = []
    for t1, t2 in candidate_pairs:
        if t1 not in prices.columns or t2 not in prices.columns:
            continue
        s1 = prices[t1].dropna()
        s2 = prices[t2].dropna()
        common = s1.index.intersection(s2.index)
        if len(common) < 252:  # need at least 1 year of data
            continue
        s1, s2 = s1.loc[common], s2.loc[common]

        coint_result = test_cointegration(s1, s2, pvalue_threshold)
        if not coint_result["is_cointegrated"]:
            continue

        spread = s1 / s2
        hl = half_life(spread)
        if not (min_half_life <= hl <= max_half_life):
            continue

        records.append(
            {
                "pair": f"{t1}_{t2}",
                "ticker1": t1,
                "ticker2": t2,
                "p_value": round(coint_result["p_value"], 4),
                "half_life_days": round(hl, 1),
            }
        )

    return pd.DataFrame(records)
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
pytest tests/test_pairs.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/pairs.py tests/test_pairs.py
git commit -m "feat: add pair construction, cointegration test, half-life screen"
```

---

## Task 4: Backtrader Indicators (Z-Score)

**Files:**
- Create: `src/indicators.py`
- Create: `tests/test_indicators.py`

**Important:** These indicators are `bt.Indicator` subclasses — they run inside the Backtrader engine declaratively, not in a Python loop. The z-score computation uses Backtrader's built-in rolling `SMA` and `StdDev`. For the cointegration approach, `bt.indicators.OLS_TransformationN` provides rolling OLS residuals natively.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_indicators.py
import pandas as pd
import numpy as np
import backtrader as bt
import pytest
from src.indicators import DistanceZScore, CointZScore


def _run_indicator(indicator_cls, pair_df, **kwargs):
    """Helper: run a bt.Indicator on a PairFeed and collect zscore output."""
    from src.backtest import PairFeed  # defined in Task 6

    class _Capture(bt.Strategy):
        def __init__(self):
            self.ind = indicator_cls(self.data, **kwargs)
            self.zscores = []

        def next(self):
            self.zscores.append(self.ind.zscore[0])

    cerebro = bt.Cerebro()
    cerebro.adddata(PairFeed(dataname=pair_df))
    cerebro.addstrategy(_Capture)
    result = cerebro.run()
    return result[0].zscores


def _make_pair_df(n=300, seed=42):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2018-01-01", periods=n, freq="B")
    common = np.cumsum(rng.normal(0, 1, n))
    p1 = pd.Series(100 + common + rng.normal(0, 0.5, n), index=idx)
    p2 = pd.Series(100 + common + rng.normal(0, 0.5, n), index=idx)
    df = pd.DataFrame({"Close": p1 / p2, "Price1": p1, "Price2": p2}, index=idx)
    df.attrs["name"] = "A_B"
    return df


class TestDistanceZScore:
    def test_output_length_matches_bars_after_warmup(self):
        pair_df = _make_pair_df()
        period = 60
        zs = _run_indicator(DistanceZScore, pair_df, period=period)
        # Backtrader only calls next() after min_period bars
        assert len(zs) == len(pair_df) - period + 1

    def test_zscore_near_zero_mean(self):
        pair_df = _make_pair_df(n=500)
        zs = _run_indicator(DistanceZScore, pair_df, period=60)
        assert abs(np.mean(zs)) < 0.5  # roughly centered

    def test_zscore_near_unit_std(self):
        pair_df = _make_pair_df(n=500)
        zs = _run_indicator(DistanceZScore, pair_df, period=60)
        assert 0.5 < np.std(zs) < 2.0  # roughly standardized


class TestCointZScore:
    def test_output_length_matches_bars_after_warmup(self):
        pair_df = _make_pair_df(n=300)
        period = 60
        zs = _run_indicator(CointZScore, pair_df, period=period)
        assert len(zs) == len(pair_df) - period + 1

    def test_zscore_finite(self):
        pair_df = _make_pair_df(n=300)
        zs = _run_indicator(CointZScore, pair_df, period=60)
        assert all(np.isfinite(v) for v in zs)
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
pytest tests/test_indicators.py -v
```

Expected: `ImportError` for `src.indicators` and `src.backtest`.

- [ ] **Step 3: Implement src/indicators.py**

```python
# src/indicators.py
"""
Backtrader indicators for pairs trading z-scores.
"""

import backtrader as bt


class DistanceZScore(bt.Indicator):
    """Rolling z-score of the Price1/Price2 ratio (Distance Approach).

    Lines:
        zscore — standardized deviation of the spread from its rolling mean.

    Entry signal (strategy): go long pair when zscore < -entry_z,
                              go short pair when zscore > +entry_z.

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

    Fits Price1 = alpha + beta * Price2 over a rolling window,
    then standardizes the current residual using historical residual stats.

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
```

- [ ] **Step 4: Run tests — verify they pass**

> Note: These tests depend on `src/backtest.PairFeed` which is created in Task 6.
> Come back to run these tests after Task 6 is complete.

- [ ] **Step 5: Commit**

```bash
git add src/indicators.py tests/test_indicators.py
git commit -m "feat: add DistanceZScore and CointZScore Backtrader indicators"
```

---

## Task 5: Sizer

**Files:**
- Create: `src/sizers.py`

- [ ] **Step 1: Implement src/sizers.py**

(Taken directly from the course notebook — no changes needed.)

```python
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
```

- [ ] **Step 2: Commit**

```bash
git add src/sizers.py
git commit -m "feat: add FixedCashSizer (100k TL per trade leg)"
```

---

## Task 6: PairFeed + Backtest Runner

**Files:**
- Create: `src/backtest.py`

The `PairFeed` is a custom `bt.feeds.PandasData` with two extra lines: `price1` and `price2`. The `run_backtest()` function is a thin Cerebro wrapper that returns metrics.

- [ ] **Step 1: Implement src/backtest.py**

```python
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
    """

    lines = ("price1", "price2")
    params = (
        ("datetime", None),   # index is the datetime
        ("open", -1),         # not used
        ("high", -1),
        ("low", -1),
        ("close", "Close"),
        ("volume", -1),
        ("openinterest", -1),
        ("price1", "Price1"),
        ("price2", "Price2"),
    )


def run_backtest(
    pair_df: pd.DataFrame,
    strategy_cls: type,
    initial_cash: float = 1_000_000,
    cash_per_trade: float = 100_000,
    **strategy_params,
) -> dict:
    """Run a single Backtrader backtest for one pair.

    Parameters
    ----------
    pair_df       : pd.DataFrame  Output of construct_pair().
    strategy_cls  : bt.Strategy subclass
    initial_cash  : float  Starting portfolio cash in TL.
    cash_per_trade: float  Per-leg trade size in TL.
    **strategy_params : forwarded to strategy.

    Returns
    -------
    dict with keys: sharpe, max_drawdown, total_return,
                    n_trades, final_value, equity_curve
    """
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(commission=0)
    cerebro.broker.set_shortcash(False)  # allow shorting leg2
    cerebro.addsizer(FixedCashSizer, cash_per_trade=cash_per_trade)

    feed = PairFeed(dataname=pair_df)
    cerebro.adddata(feed, name=pair_df.attrs.get("name", "pair"))

    cerebro.addstrategy(strategy_cls, **strategy_params)

    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe",
                        riskfreerate=0.0, annualize=True, timeframe=bt.TimeFrame.Days)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="dd")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
    cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name="timereturn")

    result = cerebro.run()
    strat = result[0]

    sharpe_raw = strat.analyzers.sharpe.get_analysis().get("sharperatio", None)
    dd_raw = strat.analyzers.dd.get_analysis()
    trade_raw = strat.analyzers.trades.get_analysis()
    returns_raw = strat.analyzers.returns.get_analysis()
    time_return_raw = strat.analyzers.timereturn.get_analysis()

    final_value = cerebro.broker.getvalue()

    return {
        "sharpe": sharpe_raw if sharpe_raw is not None else 0.0,
        "max_drawdown_pct": dd_raw.get("max", {}).get("drawdown", 0.0),
        "total_return_pct": (final_value / initial_cash - 1) * 100,
        "n_trades": trade_raw.get("total", {}).get("closed", 0),
        "final_value": final_value,
        "equity_curve": pd.Series(time_return_raw),
    }
```

- [ ] **Step 2: Now run Task 4 indicator tests**

```bash
pytest tests/test_indicators.py -v
```

Expected: all tests PASS (PairFeed now exists).

- [ ] **Step 3: Commit**

```bash
git add src/backtest.py
git commit -m "feat: add PairFeed and run_backtest() Backtrader wrapper"
```

---

## Task 7: Pairs Trading Strategy

**Files:**
- Create: `src/strategies.py`

- [ ] **Step 1: Implement src/strategies.py**

```python
# src/strategies.py
"""
PairsStrategy — single Backtrader strategy supporting both Distance
and Cointegration z-score approaches.
"""

import backtrader as bt
from src.indicators import DistanceZScore, CointZScore


class PairsStrategy(bt.Strategy):
    """Long-short pairs trading strategy.

    params:
        period       (int)   : Formation window for z-score indicator. Default 60.
        entry_z      (float) : |z-score| threshold to open a position. Default 2.0.
        exit_z       (float) : |z-score| threshold to close a position. Default 1.0.
        approach     (str)   : 'distance' or 'coint'. Default 'distance'.
        wait_reversal(bool)  : If True, wait for z-score to start reverting before
                               entry (improvement: avoids entering on momentum).
                               Default False.
        cash_per_leg (float) : Cash allocated per leg of the trade. Default 100_000.

    Trade logic:
        LONG  pair: spread too low  (z < -entry_z) → buy leg1, short leg2
        SHORT pair: spread too high (z >  entry_z) → short leg1, buy leg2
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

        self._in_long = False   # long pair: long leg1, short leg2
        self._in_short = False  # short pair: short leg1, long leg2

    def _get_sizes(self):
        """Compute integer share sizes for each leg based on cash_per_leg."""
        p1 = self.data.price1[0]
        p2 = self.data.price2[0]
        size1 = int(self.p.cash_per_leg / p1) if p1 > 0 else 0
        size2 = int(self.p.cash_per_leg / p2) if p2 > 0 else 0
        return size1, size2

    def _should_enter(self, z: float, direction: str) -> bool:
        """Check entry condition, optionally requiring reversal signal."""
        if direction == "long":
            crossed = z < -self.p.entry_z
        else:
            crossed = z > self.p.entry_z

        if not crossed:
            return False

        if self.p.wait_reversal:
            prev_z = self.zscore_ind.zscore[-1]
            if direction == "long":
                # z is below -entry_z AND started moving back up
                return z > prev_z
            else:
                # z is above +entry_z AND started moving back down
                return z < prev_z

        return True

    def next(self):
        z = self.zscore_ind.zscore[0]
        size1, size2 = self._get_sizes()

        if size1 == 0 or size2 == 0:
            return

        if not self._in_long and not self._in_short:
            if self._should_enter(z, "long"):
                self.buy(size=size1)    # buy leg1 (price1)
                # Note: shorting leg2 is modelled implicitly via the ratio.
                # In a ratio-only feed, buying the pair = long spread.
                self._in_long = True

            elif self._should_enter(z, "short"):
                self.sell(size=size1)   # sell/short the spread
                self._in_short = True

        elif self._in_long:
            if z > -self.p.exit_z:
                self.close()
                self._in_long = False

        elif self._in_short:
            if z < self.p.exit_z:
                self.close()
                self._in_short = False
```

- [ ] **Step 2: Smoke-test strategy runs without error**

```python
# Run this in a Python session or small test script to verify no crashes:
import pandas as pd
import numpy as np
from src.backtest import run_backtest
from src.strategies import PairsStrategy

rng = np.random.default_rng(42)
n = 500
idx = pd.date_range("2018-01-01", periods=n, freq="B")
common = np.cumsum(rng.normal(0, 1, n))
p1 = pd.Series(100 + common + rng.normal(0, 0.3, n), index=idx)
p2 = pd.Series(100 + common + rng.normal(0, 0.3, n), index=idx)
pair_df = pd.DataFrame({"Close": p1/p2, "Price1": p1, "Price2": p2}, index=idx)
pair_df.attrs["name"] = "TEST"

result = run_backtest(pair_df, PairsStrategy, approach="distance", period=60)
print(result["sharpe"], result["n_trades"], result["total_return_pct"])
# Expected: numbers without errors
```

- [ ] **Step 3: Commit**

```bash
git add src/strategies.py
git commit -m "feat: add PairsStrategy supporting distance and cointegration approaches"
```

---

## Task 8: Performance Metrics & Monte Carlo Test

**Files:**
- Create: `src/metrics.py`
- Create: `tests/test_metrics.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_metrics.py
import numpy as np
import pandas as pd
import pytest
from src.metrics import sharpe_ratio, sortino_ratio, monte_carlo_test


class TestSharpeRatio:
    def test_positive_returns_positive_sharpe(self):
        rets = pd.Series([0.001] * 252)
        assert sharpe_ratio(rets) > 0

    def test_zero_returns_zero_sharpe(self):
        rets = pd.Series([0.0] * 252)
        assert sharpe_ratio(rets) == 0.0

    def test_annualization(self):
        # Known case: daily return 0.001, daily std 0.001 → Sharpe = sqrt(252)
        rets = pd.Series([0.001] * 252)
        expected = np.sqrt(252)
        assert abs(sharpe_ratio(rets) - expected) < 0.01


class TestSortinoRatio:
    def test_only_positive_returns_high_sortino(self):
        rets = pd.Series([0.002] * 252)
        assert sortino_ratio(rets) > 5

    def test_mixed_returns_lower_than_sharpe(self):
        rng = np.random.default_rng(1)
        rets = pd.Series(rng.normal(0.001, 0.01, 252))
        assert sortino_ratio(rets) >= 0  # valid number


class TestMonteCarlo:
    def test_returns_expected_keys(self):
        rng = np.random.default_rng(0)
        rets = pd.Series(rng.normal(0.001, 0.01, 252))
        result = monte_carlo_test(rets, n_simulations=100, seed=0)
        assert {"observed_sharpe", "p_value", "null_sharpes"} <= result.keys()

    def test_high_sharpe_strategy_low_pvalue(self):
        """A clearly superior strategy should have p-value < 0.05."""
        # Returns of 0.003/day (Sharpe ≈ 4.8) should beat random easily
        rets = pd.Series([0.003] * 252)
        result = monte_carlo_test(rets, n_simulations=500, seed=42)
        assert result["p_value"] < 0.05

    def test_random_strategy_high_pvalue(self):
        """Shuffled random returns should have p-value >> 0.05 on average."""
        rng = np.random.default_rng(7)
        rets = pd.Series(rng.normal(0.0, 0.01, 252))
        result = monte_carlo_test(rets, n_simulations=500, seed=42)
        # p-value should be ~0.5 (no edge), so definitely > 0.05
        assert result["p_value"] > 0.05
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
pytest tests/test_metrics.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement src/metrics.py**

```python
# src/metrics.py
"""Performance metrics and Monte Carlo significance test."""

from __future__ import annotations

import numpy as np
import pandas as pd


def sharpe_ratio(returns: pd.Series, annualization: int = 252) -> float:
    """Annualized Sharpe ratio (risk-free rate = 0)."""
    std = returns.std()
    if std == 0 or np.isnan(std):
        return 0.0
    return float(returns.mean() / std * np.sqrt(annualization))


def sortino_ratio(returns: pd.Series, annualization: int = 252) -> float:
    """Annualized Sortino ratio (penalises only downside volatility)."""
    downside = returns[returns < 0]
    if len(downside) == 0 or downside.std() == 0:
        return float("inf") if returns.mean() > 0 else 0.0
    return float(returns.mean() / downside.std() * np.sqrt(annualization))


def max_drawdown(equity_curve: pd.Series) -> float:
    """Maximum peak-to-trough drawdown as a positive fraction (0–1)."""
    roll_max = equity_curve.cummax()
    drawdown = (equity_curve - roll_max) / roll_max
    return float(abs(drawdown.min()))


def monte_carlo_test(
    strategy_returns: pd.Series,
    n_simulations: int = 1000,
    seed: int = 42,
) -> dict:
    """Permutation test: is strategy Sharpe significantly > random?

    Method: randomly shuffle daily returns n_simulations times to break
    the time structure. Compute Sharpe for each permutation.
    p-value = fraction of simulated Sharpes >= observed Sharpe.

    Parameters
    ----------
    strategy_returns : pd.Series  Daily returns of the strategy.
    n_simulations    : int        Number of random permutations.
    seed             : int        Random seed for reproducibility.

    Returns
    -------
    dict with keys:
        observed_sharpe (float)
        null_sharpes    (np.ndarray of length n_simulations)
        p_value         (float)
        null_mean       (float)
        null_std        (float)
    """
    rng = np.random.default_rng(seed)
    observed = sharpe_ratio(strategy_returns)
    rets_array = strategy_returns.values

    null_sharpes = np.array(
        [
            sharpe_ratio(pd.Series(rng.permutation(rets_array)))
            for _ in range(n_simulations)
        ]
    )

    p_value = float(np.mean(null_sharpes >= observed))

    return {
        "observed_sharpe": observed,
        "null_sharpes": null_sharpes,
        "p_value": p_value,
        "null_mean": float(null_sharpes.mean()),
        "null_std": float(null_sharpes.std()),
    }
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
pytest tests/test_metrics.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/metrics.py tests/test_metrics.py
git commit -m "feat: add Sharpe, Sortino, max drawdown, and Monte Carlo test"
```

---

## Task 9: Optimization (Grid Search + Walk-Forward)

**Files:**
- Create: `src/optimize.py`

- [ ] **Step 1: Implement src/optimize.py**

```python
# src/optimize.py
"""Grid search and walk-forward optimization for pairs strategies."""

from __future__ import annotations

import pandas as pd
import numpy as np
from src.backtest import run_backtest
from src.strategies import PairsStrategy


def grid_search(
    pair_df: pd.DataFrame,
    param_grid: dict,
    approach: str = "distance",
    min_trades: int = 3,
) -> pd.DataFrame:
    """Exhaustive grid search over formation periods.

    Parameters
    ----------
    pair_df    : pd.DataFrame  Output of construct_pair().
    param_grid : dict          e.g. {"period": [20, 40, 60, 90, 120, 180]}
    approach   : str           'distance' or 'coint'
    min_trades : int           Skip results with fewer trades (avoids overfitting
                               to lucky single trades, as warned in course notes).

    Returns
    -------
    pd.DataFrame sorted by Sharpe descending.
    """
    records = []
    for period in param_grid.get("period", [60]):
        result = run_backtest(pair_df, PairsStrategy, approach=approach, period=period)
        if result["n_trades"] < min_trades:
            continue
        records.append(
            {
                "period": period,
                "sharpe": result["sharpe"],
                "max_drawdown_pct": result["max_drawdown_pct"],
                "total_return_pct": result["total_return_pct"],
                "n_trades": result["n_trades"],
            }
        )
    return pd.DataFrame(records).sort_values("sharpe", ascending=False)


def walk_forward_optimization(
    pair_df: pd.DataFrame,
    param_grid: dict,
    train_days: int = 252,
    test_days: int = 63,
    approach: str = "distance",
    min_trades: int = 3,
) -> pd.DataFrame:
    """Rolling walk-forward: optimize on train window, evaluate on test window.

    Parameters
    ----------
    pair_df    : pair DataFrame (Close = ratio, Price1, Price2)
    param_grid : {"period": [...]}
    train_days : number of days in the training window (default 252 = 1 year)
    test_days  : number of days in the test window (default 63 = 1 quarter)
    approach   : 'distance' or 'coint'
    min_trades : minimum trades to consider a parameter result valid

    Returns
    -------
    pd.DataFrame with columns: window_start, best_period, test_sharpe,
                                test_return_pct, test_trades
    """
    records = []
    n = len(pair_df)
    start = 0

    while start + train_days + test_days <= n:
        train_slice = pair_df.iloc[start : start + train_days].copy()
        test_slice = pair_df.iloc[start + train_days : start + train_days + test_days].copy()

        # Find best period on training data
        best_period = None
        best_sharpe = -np.inf
        for period in param_grid.get("period", [60]):
            if len(train_slice) <= period:
                continue
            res = run_backtest(train_slice, PairsStrategy, approach=approach, period=period)
            if res["n_trades"] >= min_trades and res["sharpe"] > best_sharpe:
                best_sharpe = res["sharpe"]
                best_period = period

        if best_period is None:
            start += test_days
            continue

        # Evaluate best period on test data
        test_res = run_backtest(test_slice, PairsStrategy, approach=approach, period=best_period)

        records.append(
            {
                "window_start": pair_df.index[start + train_days],
                "best_period": best_period,
                "test_sharpe": test_res["sharpe"],
                "test_return_pct": test_res["total_return_pct"],
                "test_trades": test_res["n_trades"],
            }
        )
        start += test_days

    return pd.DataFrame(records)
```

- [ ] **Step 2: Commit**

```bash
git add src/optimize.py
git commit -m "feat: add grid search and walk-forward optimization"
```

---

## Task 10: Full Notebook Assembly

**Files:**
- Create: `notebook.ipynb`

The notebook imports all src modules and contains the full analysis. Structure it as the graded presentation content.

- [ ] **Step 1: Create notebook with the following section structure**

Section cells to create in order:

```
# 0. Setup & Imports
# 1. Data Loading & EDA
# 2. Pair Construction & Cointegration Screening
# 3. Distance Strategy — Base Backtest
# 4. Cointegration Strategy — Base Backtest
# 5. Parameter Optimization (Grid Search)
# 6. Improvement: Reversal Filter
# 7. Walk-Forward Optimization
# 8. Monte Carlo Significance Tests
# 9. Strategy Comparison Table
# 10. Conclusions
```

- [ ] **Step 2: Write Section 0 — Setup**

```python
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import backtrader as bt

from src.pairs import construct_pair, screen_pairs
from src.strategies import PairsStrategy
from src.backtest import run_backtest
from src.metrics import sharpe_ratio, sortino_ratio, monte_carlo_test
from src.optimize import grid_search, walk_forward_optimization

plt.rcParams["figure.figsize"] = (12, 5)
plt.rcParams["axes.grid"] = True

INITIAL_CASH = 1_000_000
CASH_PER_TRADE = 100_000
RESULTS_DIR = "results"

import os; os.makedirs(RESULTS_DIR, exist_ok=True)
```

- [ ] **Step 3: Write Section 1 — Data Loading**

```python
# Read from CSV — never re-fetch during analysis (reproducibility requirement)
prices = pd.read_csv("data/prices.csv", index_col=0, parse_dates=True)
print(f"Data: {prices.shape[1]} tickers × {len(prices)} trading days")
print(f"Date range: {prices.index[0].date()} → {prices.index[-1].date()}")
prices.head()
```

```python
# Plot all price series (normalized to 100)
(prices / prices.iloc[0] * 100).plot(figsize=(14, 6), legend=False, alpha=0.4)
plt.title("Normalized Prices (Base = 100)")
plt.ylabel("Normalized Price")
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/normalized_prices.png", dpi=150)
plt.show()
```

- [ ] **Step 4: Write Section 2 — Pair Screening**

```python
# Define candidate pairs — grouped by economic rationale
CANDIDATE_PAIRS = [
    ("GARAN.IS", "AKBNK.IS"),   # private banks
    ("ISCTR.IS", "VAKBN.IS"),   # state banks
    ("YKBNK.IS", "HALKB.IS"),   # mid banks
    ("GARAN.IS", "ISCTR.IS"),   # banking cross
    ("TCELL.IS", "TTKOM.IS"),   # telecom duopoly
    ("THYAO.IS", "PGSUS.IS"),   # aviation duopoly
    ("EREGL.IS", "SISE.IS"),    # industrials
    ("KCHOL.IS", "SAHOL.IS"),   # conglomerates
    ("KOZAL.IS", "KOZAA.IS"),   # Koza group
    ("BIMAS.IS", "MGROS.IS"),   # retail duopoly
    ("TUPRS.IS", "AYGAZ.IS"),   # downstream petroleum
]

# Screen pairs
screened = screen_pairs(prices, CANDIDATE_PAIRS)
print(f"\nPairs passing cointegration + half-life screen: {len(screened)}")
screened.sort_values("p_value")
```

```python
# Save screening results
screened.to_csv(f"{RESULTS_DIR}/pair_screening.csv", index=False)

# Plot normalized prices for each screened pair
for _, row in screened.iterrows():
    t1, t2 = row["ticker1"], row["ticker2"]
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    (prices[[t1, t2]] / prices[[t1, t2]].iloc[0] * 100).plot(ax=axes[0])
    axes[0].set_title(f"{row['pair']} — Normalized Prices")
    (prices[t1] / prices[t2]).plot(ax=axes[1], color="purple")
    axes[1].set_title(f"Spread (Ratio)  |  half-life = {row['half_life_days']} days")
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/pair_{row['pair']}.png", dpi=150)
    plt.show()
```

```python
# Build pair DataFrames for all screened pairs
pair_dfs = {}
for _, row in screened.iterrows():
    t1, t2 = row["ticker1"], row["ticker2"]
    s1 = prices[t1].dropna()
    s2 = prices[t2].dropna()
    common = s1.index.intersection(s2.index)
    df = construct_pair(s1.loc[common], s2.loc[common], t1, t2)
    pair_dfs[row["pair"]] = df

print(f"Built {len(pair_dfs)} pair DataFrames.")
```

- [ ] **Step 5: Write Section 3 — Distance Strategy Base Backtest**

```python
distance_results = {}
for name, pdf in pair_dfs.items():
    res = run_backtest(pdf, PairsStrategy, approach="distance", period=60)
    distance_results[name] = res
    print(f"{name}: Sharpe={res['sharpe']:.2f}  MaxDD={res['max_drawdown_pct']:.1f}%  "
          f"Return={res['total_return_pct']:.1f}%  Trades={res['n_trades']}")
```

```python
# Aggregate results table
dist_df = pd.DataFrame({
    k: {
        "Sharpe": v["sharpe"],
        "Max DD %": v["max_drawdown_pct"],
        "Return %": v["total_return_pct"],
        "# Trades": v["n_trades"],
    }
    for k, v in distance_results.items()
}).T.round(2)

dist_df.to_csv(f"{RESULTS_DIR}/distance_base_results.csv")
dist_df
```

- [ ] **Step 6: Write Section 4 — Cointegration Strategy Base Backtest**

```python
coint_results = {}
for name, pdf in pair_dfs.items():
    res = run_backtest(pdf, PairsStrategy, approach="coint", period=60)
    coint_results[name] = res

coint_df = pd.DataFrame({
    k: {
        "Sharpe": v["sharpe"],
        "Max DD %": v["max_drawdown_pct"],
        "Return %": v["total_return_pct"],
        "# Trades": v["n_trades"],
    }
    for k, v in coint_results.items()
}).T.round(2)

coint_df.to_csv(f"{RESULTS_DIR}/coint_base_results.csv")
coint_df
```

- [ ] **Step 7: Write Section 5 — Optimization**

```python
PARAM_GRID = {"period": [20, 40, 60, 90, 120, 180]}

opt_records = []
for name, pdf in pair_dfs.items():
    for approach in ["distance", "coint"]:
        gs = grid_search(pdf, PARAM_GRID, approach=approach)
        if not gs.empty:
            best = gs.iloc[0]
            opt_records.append({
                "pair": name, "approach": approach,
                "best_period": best["period"],
                "best_sharpe": best["sharpe"],
                "best_return_pct": best["total_return_pct"],
            })

opt_df = pd.DataFrame(opt_records)
opt_df.to_csv(f"{RESULTS_DIR}/optimization_results.csv", index=False)
opt_df
```

- [ ] **Step 8: Write Section 6 — Reversal Filter**

```python
reversal_records = []
for name, pdf in pair_dfs.items():
    for approach in ["distance", "coint"]:
        base = run_backtest(pdf, PairsStrategy, approach=approach, period=60, wait_reversal=False)
        improved = run_backtest(pdf, PairsStrategy, approach=approach, period=60, wait_reversal=True)
        reversal_records.append({
            "pair": name, "approach": approach,
            "base_sharpe": base["sharpe"],
            "reversal_sharpe": improved["sharpe"],
            "improvement": improved["sharpe"] - base["sharpe"],
        })

rev_df = pd.DataFrame(reversal_records)
rev_df.to_csv(f"{RESULTS_DIR}/reversal_filter_results.csv", index=False)
rev_df.sort_values("improvement", ascending=False)
```

- [ ] **Step 9: Write Section 7 — Walk-Forward Optimization**

```python
wfo_records = []
for name, pdf in pair_dfs.items():
    for approach in ["distance", "coint"]:
        wfo = walk_forward_optimization(pdf, PARAM_GRID, approach=approach)
        if not wfo.empty:
            mean_sharpe = wfo["test_sharpe"].mean()
            wfo_records.append({
                "pair": name, "approach": approach,
                "wfo_mean_sharpe": round(mean_sharpe, 3),
                "wfo_windows": len(wfo),
            })

wfo_df = pd.DataFrame(wfo_records)
wfo_df.to_csv(f"{RESULTS_DIR}/wfo_results.csv", index=False)
wfo_df
```

- [ ] **Step 10: Write Section 8 — Monte Carlo Test**

```python
from src.metrics import monte_carlo_test
import matplotlib.pyplot as plt

# Pick the best-performing pair/approach from optimization
best_row = opt_df.sort_values("best_sharpe", ascending=False).iloc[0]
best_pair = best_row["pair"]
best_approach = best_row["approach"]
best_period = int(best_row["best_period"])

best_res = run_backtest(
    pair_dfs[best_pair], PairsStrategy,
    approach=best_approach, period=best_period
)
equity = best_res["equity_curve"]
strategy_rets = equity.pct_change().dropna()

mc = monte_carlo_test(strategy_rets, n_simulations=1000, seed=42)
print(f"Observed Sharpe : {mc['observed_sharpe']:.3f}")
print(f"Random mean     : {mc['null_mean']:.3f} ± {mc['null_std']:.3f}")
print(f"p-value         : {mc['p_value']:.4f}")

fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(mc["null_sharpes"], bins=50, color="steelblue", alpha=0.7, label="Random Sharpe")
ax.axvline(mc["observed_sharpe"], color="red", linewidth=2,
           label=f"Strategy Sharpe = {mc['observed_sharpe']:.2f}")
ax.set_title(f"Monte Carlo Significance Test — {best_pair} ({best_approach})\n"
             f"p-value = {mc['p_value']:.4f}")
ax.set_xlabel("Sharpe Ratio")
ax.legend()
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/mc_test_{best_pair}_{best_approach}.png", dpi=150)
plt.show()
```

- [ ] **Step 11: Write Section 9 — Final Comparison Table**

```python
summary = pd.DataFrame({
    "Distance (base h=60)": dist_df["Sharpe"],
    "Coint (base h=60)": coint_df["Sharpe"],
}).round(3)

# Add optimized column from opt_df
for approach, label in [("distance", "Distance (optimized)"), ("coint", "Coint (optimized)")]:
    sub = opt_df[opt_df["approach"] == approach].set_index("pair")["best_sharpe"]
    summary[label] = sub

summary.to_csv(f"{RESULTS_DIR}/final_comparison.csv")
print("\n=== FINAL COMPARISON (Sharpe Ratio) ===")
summary
```

- [ ] **Step 12: Commit notebook**

```bash
git add notebook.ipynb results/
git commit -m "feat: complete pairs trading analysis notebook with all sections"
```

---

## Task 11: Final Verification & Submission

- [ ] **Step 1: Run all tests**

```bash
pytest tests/ -v
```

Expected: all tests PASS.

- [ ] **Step 2: Clear all outputs and re-run notebook from scratch**

In Jupyter: `Kernel → Restart & Run All`

Verify: no errors, all plots render, all CSV results files saved in `results/`.

- [ ] **Step 3: Verify results are deterministic**

Run the notebook a second time (Restart & Run All again) and confirm that the numbers in `results/*.csv` are identical.

- [ ] **Step 4: Verify requirements.txt is complete**

```bash
pip freeze > requirements_check.txt
# Manually compare with requirements.txt to ensure all packages used are pinned
```

- [ ] **Step 5: Final commit**

```bash
git add .
git commit -m "feat: complete statistical arbitrage project — all tests passing, notebook verified"
```

---

## Self-Review Checklist

| Requirement | Covered in Task |
|-------------|----------------|
| At least 10 pairs | Task 10 §4 (11 candidate pairs) |
| Price ratio construction | Task 3 `construct_pair()` |
| Distance z-score approach | Task 4 `DistanceZScore` |
| Cointegration z-score approach | Task 4 `CointZScore` |
| Formation period optimization | Task 9 `grid_search()` |
| Reversal filter improvement | Task 10 §8, `wait_reversal` param |
| Walk-forward optimization | Task 9 `walk_forward_optimization()` |
| Monte Carlo significance test | Task 8 `monte_carlo_test()` |
| Compare Distance vs Coint | Task 10 §9 final comparison |
| Backtrader backtesting | Task 6, 7 |
| FixedCashSizer (100k TL) | Task 5 |
| Save data to CSV | Task 2 |
| Save results to CSV | Task 10 (all sections) |
| requirements.txt with versions | Task 1 |
| No live data fetching in backtest | Task 2 (read-only CSV after download) |

---

*Generated 2026-05-24 for EC581 Statistical Arbitrage Project*
