# Implementation vs. Paper: Findings Report

**Paper:** Gatev, Goetzmann & Rouwenhorst (2006) — *Pairs Trading: Performance of a Relative Value Arbitrage Rule*
**Date:** 2026-05-28

---

## Summary

Our implementation captures the spirit of GGR (2006) but diverges from the paper at several methodologically significant points. Five deviations are critical enough to affect results interpretation. The rest are minor adaptations appropriate for a course project.

---

## Critical Deviations

### 1. Pair Selection: Distance Method vs. Cointegration Test

**Paper:** Pairs are formed by finding, for each stock, the partner stock that **minimizes the sum of squared deviations** between the two normalized cumulative return series over a 12-month formation window. No statistical test is applied — the pair with the smallest distance is always chosen regardless of how statistically "close" the series are.

**Our implementation:** Pairs are pre-specified based on economic rationale (sector membership, corporate group), then screened using the **Engle-Granger cointegration test** (p < 0.10) and a half-life filter.

**Impact:** This is a fundamental difference in pair construction philosophy.
- GGR's distance method is purely mechanical and data-driven: it selects whatever pairs moved together historically, even across sectors. It does not require cointegration.
- Our cointegration filter is statistically more rigorous but requires a longer data history and fails when regime changes break cointegration over the full sample period (as we experienced — only 1 pair passed over 2010–2026, requiring us to restrict to post-2015 data).
- GGR would have selected many more pairs (top 5, 20, 101-120) since there is no statistical rejection criterion.

**What to say in the presentation:** We implement the cointegration approach (which the paper discusses theoretically in Section 1.4) in addition to the distance signal — an extension of the paper, not a replication.

---

### 2. Formation vs. Trading Window Structure

**Paper:** Two explicit phases:
- **Formation period:** 12 months — compute normalized cumulative return for each stock; find minimum-distance partners; estimate historical standard deviation of the spread.
- **Trading period:** 6 months — apply a fixed threshold (2× the formation-period std) to generate signals. The std is **frozen** from the formation period; it does not update during trading.
- Portfolios are re-initiated every month, creating 6 overlapping portfolios simultaneously.

**Our implementation:** A single continuous rolling window. The z-score's mean and standard deviation are recomputed each bar over the last `period` bars (default 60 days). There is no separate formation/trading split.

**Impact:**
- In GGR, the z-score threshold is a fixed number (2× formation std) expressed in price-ratio units. Ours is always exactly ±2 standard deviations by construction (rolling standardization).
- GGR's entry threshold adapts across different formation periods (different pairs have different std), but stays fixed within a trading period. Ours adapts continuously, which may be more reactive to regime changes but also more susceptible to look-ahead bias if not implemented carefully.
- The overlapping-portfolio design in GGR (6 portfolios active at once, averaged) smooths returns and reduces timing luck. Our single-window backtest has higher exposure to entry-point timing.

---

### 3. Exit Rule: Price Crossing vs. Z-Score Threshold

**Paper:** A position is closed when the **spread crosses zero** — i.e., when the normalized prices converge back and the "loser" price catches up to (or surpasses) the "winner" price. The paper explicitly states: *"We unwind the position at the next crossing of the prices."* If prices never cross within the 6-month trading window, the position is force-closed at the end of the period.

**Our implementation:** A position is closed when the **z-score crosses back through ±`exit_z`** (default 1.0). This is a z-score level exit, not a zero-crossing of the spread.

**Impact:**
- GGR's zero-crossing exit is more conservative — it requires actual price convergence. A trade closed at z = 1.0 (our rule) has not necessarily seen the spread return to its historical mean; GGR would keep the position open.
- GGR's trades are on average longer (3.75 months avg holding period per paper). Our rolling z-score exit likely closes positions faster.
- Our exit rule is closer to the **cointegration** literature exit (close when spread mean-reverts partway) than to the GGR distance approach.

---

### 4. Long-Short Implementation: True vs. Synthetic

**Paper:** A genuine dollar-neutral long-short: simultaneously **short $1 of the outperforming stock** and **long $1 of the underperforming stock**. Two separate equity positions in the underlying stocks. Self-financing (the short proceeds fund the long).

**Our implementation:** Trades the **Price1/Price2 ratio** as a single synthetic instrument. When the strategy "buys," it buys the ratio — not the individual stocks. Leg 2 is never actually shorted.

**Impact:**
- We do not compute the actual P&L on both legs separately. The ratio trade proxies the combined position but misses:
  - Short-sale borrow costs (zero in GGR, but non-zero in practice)
  - Dividend payments on short positions
  - Bid-ask bounce effects (GGR discusses this extensively in Section 3.1)
- The ratio instrument is not a standard financial asset — its price can be meaningless in absolute terms (GARAN/AKBNK = 2.5 "TL per TL" is not a real price). The FixedCashSizer's sizing is therefore based on individual leg prices, which is a workaround rather than a true replication.
- For presentation purposes, framing this as "we model the spread as a synthetic long position" is accurate.

---

### 5. Pair Ranking vs. Pre-Selected Pairs

**Paper:** After computing distances for all possible pairs in the CRSP universe, the strategy trades the **top 5** and **top 20** pairs ranked by smallest distance. No prior knowledge of which pairs would score well is used — the ranking is purely mechanical.

**Our implementation:** Pairs are **hand-selected** based on economic rationale (banking sector, Koç group subsidiaries, conglomerates, etc.) and then filtered by cointegration test.

**Impact:**
- Our selection introduces a form of **selection bias**: we are implicitly hypothesizing which pairs should be cointegrated before looking at the data. GGR avoids this by letting the data rank all pairs mechanically.
- However, pre-selecting economically motivated pairs is common in academic literature (and required by the course project: *"Construct at least 10 pairs based on ownership, sector, etc."*).
- Our approach is arguably more robust to spurious correlations because economic rationale provides a prior that the co-movement is structural, not coincidental.

---

## Minor Deviations

### 6. Price Normalization

**Paper:** Prices are normalized to a **cumulative total return index** with dividends reinvested, starting at 1.0 at the beginning of each formation period. The distance is computed in this normalized space.

**Our implementation:** Uses adjusted close prices from yfinance (which incorporate dividend adjustments via a backward price adjustment factor) but does not explicitly normalize prices to start at 1.0. The ratio `Price1 / Price2` implicitly normalizes the relative values but is not the GGR normalization.

**Impact:** Minor for liquid Turkish stocks where dividend yields are relatively low. For dividend-paying stocks like TUPRS.IS or AEFES.IS, not normalizing to total return could slightly distort the spread level over long periods.

---

### 7. Liquidity Filter

**Paper:** Screens out any stock that has **one or more days with no trading** during the formation period. This ensures the price series are complete and the distance calculation is not distorted by illiquid trading days.

**Our implementation:** Forward-fills gaps up to 3 consecutive days (fills trading halts) but does not remove tickers with significant illiquidity.

**Impact:** Minimal for large-cap BIST stocks (GARAN, KCHOL, etc.) which trade continuously. Would matter for smaller-cap pairs.

---

### 8. Return Measurement: Committed vs. Invested Capital

**Paper:** Reports two distinct return metrics:
- **Committed capital return:** Divides payoffs by total number of selected pairs (including pairs that never opened a position). More conservative.
- **Fully-invested return:** Divides payoffs only by pairs that actually opened a position.

**Our implementation:** Reports total portfolio return from Backtrader, equivalent to fully-invested return. Committed capital return is not computed.

**Impact:** Our returns are likely slightly more optimistic than the committed capital measure. For a fair comparison with GGR's Table 1, the committed capital figures (0.78–0.81% per month) are the appropriate benchmark, not the fully-invested figures (1.31–1.44%).

---

### 9. Monte Carlo Test Design

**Paper:** Uses a **bootstrap** methodology to distinguish pairs trading profits from momentum/reversal effects documented in prior literature.

**Our implementation:** Uses a **sign-randomization** permutation test — randomly flip the sign of each daily return to create a null distribution of Sharpe ratios. Tests whether our Sharpe is significantly above the zero-mean null.

**Impact:** Our test answers a different question than GGR's bootstrap. GGR tests whether profits are distinct from known risk factors; ours tests whether returns have a positive mean. Both are valid for a course project but are not directly comparable.

---

## What the Paper Does That We Don't Implement (Out of Scope)

| Paper Component | Status |
|-----------------|--------|
| Unrestricted pair selection from full CRSP universe | Not implemented (pre-selected pairs) |
| Top-5, Top-20, pairs 101-120 portfolio analysis | Not implemented |
| Overlapping monthly portfolios (6 simultaneous) | Not implemented |
| Sector-neutral pairs analysis (Section 3.5) | Partial (our pairs are already sector-grouped) |
| Bootstrap significance vs. known risk factors | Not implemented |
| Transaction cost sensitivity analysis (Section 3.3) | Not implemented |
| Long-only vs. short-only decomposition | Not implemented |
| Bid-ask bounce correction (Panel B of Table 1) | Not implemented |

---

## What We Implemented That the Paper Doesn't Have

| Our Extension | Description |
|---------------|-------------|
| **Cointegration test screening** | Engle-Granger test + half-life filter before pair inclusion |
| **Cointegration z-score strategy** | OLS residual z-score as signal (vs. GGR's distance signal) |
| **Walk-forward optimization** | Rolling train/test split for out-of-sample parameter validation |
| **Reversal filter** | Wait for z-score to begin reverting before entry |
| **Formation period grid search** | Optimize window length `h` over [20, 40, 60, 90, 120, 180] days |
| **BIST market adaptation** | Applied to Turkish equities 2015–2026 rather than US equities 1962–2002 |

---

## Performance Comparison

| Metric | GGR Paper (US, 1962–2002) | Our Implementation (BIST, 2015–2026) |
|--------|--------------------------|--------------------------------------|
| Pair selection | Min. distance (exhaustive) | Economic rationale + EG cointegration |
| Pairs tested | Top 5 / Top 20 of ~full US universe | 8 pairs from 53 candidates |
| Best Sharpe (optimized) | Not directly reported | **1.011** (KCHOL/SAHOL, distance, h=20) |
| Avg monthly excess return | 0.78–1.44% (top-20 portfolio) | Per-pair backtest only |
| Statistical significance | Bootstrap p-value significant | p = 0.000 (sign randomization) |
| Transaction costs | 0 (Panel A); ~162 bp per round-trip (est.) | 0 (commission=0 by course requirement) |

---

## Key Takeaway for the Presentation

Our implementation is best described as a **course-adapted extension** of GGR (2006):

- We replicate the core intuition faithfully: normalize price series, identify pairs that move together, trade on z-score divergence, and close on convergence.
- We implement the **cointegration approach** that GGR discusses theoretically (Section 1.4) but does not implement empirically — making this a genuine extension.
- The main methodological gap is in pair formation: GGR's distance-based exhaustive matching over the full universe is not replicated. Instead we use a pre-specified, economically motivated universe screened by cointegration, which is the method required by the course notebook.
- Zero transaction cost assumption (course requirement) means our Sharpe ratios are optimistic. GGR estimates ~162 bps per round-trip; applying this to BIST (higher bid-ask spreads than US) would significantly reduce reported profits.
