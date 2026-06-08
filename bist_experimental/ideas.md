# Experimental Ideas — BIST Pairs Trading

---

## Idea 1: TCMB Policy Shock Filter

### Concept

Standard GGR pairs trading assumes the spread between two stocks is stationary and will revert to its historical mean. This assumption holds in calm markets but breaks down when a market-wide macro shock hits — both legs of a pair move in the same direction (or opposite directions for unrelated reasons), and the spread diverges without reverting during the trading window.

BIST is uniquely vulnerable to this. The Turkish Central Bank (TCMB) has made surprise rate decisions of 200–500 bps in a single meeting. TRY/USD crises have caused entire market sectors to gap down simultaneously. During these episodes, pairs that were comoving during the formation period decouple — not because of a temporary mispricing, but because the macro environment has structurally changed.

**Research question:** Do GGR pairs opened during TCMB shock windows lose money? If yes, does filtering them out improve the Sharpe ratio?

---

### What Constitutes a "Shock Window"

Three triggers, any one of which blocks new pair openings for N days after the event:

| Trigger | Definition | Block duration |
|---|---|---|
| TCMB surprise rate decision | Actual rate change differs from consensus by >50 bps | 5 trading days |
| TRY/USD daily move | USD/TRY daily return > 3% in either direction | 3 trading days |
| High CPI print | Monthly CPI (TurkStat) >2% above prior month | 3 trading days |

Existing open positions are **not closed** on a shock — only new entries are blocked. Pairs already in a trade continue to their zero-crossing exit as usual.

---

### Data Needed

- **TCMB meeting dates and rate decisions** — available from TCMB's website (evds2.tcmb.gov.tr). Downloadable as CSV.
- **USD/TRY daily prices** — downloadable via yfinance as `USDTRY=X`.
- **TurkStat CPI releases** — available from EVDS (Turkish Statistical Institute data system). Monthly series, downloadable as CSV.

---

### Implementation Plan

**Step 1 — Build shock calendar**
```python
# shock_filter.py
def build_shock_calendar(tcmb_decisions, usdtry_prices, cpi_data, block_days=5):
    """Returns a DatetimeIndex of dates on which new positions are blocked."""
```

**Step 2 — Modify `run_ggr_portfolio`**
Add an optional `shock_dates` parameter. Inside the trading loop, before executing a new pair entry, check if the current date is in the shock calendar. If yes, skip the entry.

**Step 3 — Compare portfolios**
Run two portfolios side by side — baseline GGR and shock-filtered GGR — and compare:
- Sharpe ratio (committed and fully-invested)
- Max drawdown
- Number of trades opened
- Win rate on filtered-out trades (did we dodge losers?)

---

### Expected Outcome

Filtering should reduce the number of trades (fewer entries during volatile periods) and improve the Sharpe ratio by cutting the worst drawdown windows. The key question is whether the improvement in risk is worth the reduction in returns. If filtered-out trades have a lower win rate than baseline, that confirms the filter is economically justified.

---

---

## Idea 2: Real (CPI-Adjusted) Pair Formation

### Concept

GGR pair selection ranks all stock pairs by the sum of squared differences (SSD) of their normalized price paths over the 12-month formation window. On BIST, this is distorted by Turkey's chronic inflation. When annual CPI runs at 20–85%, every TRY-denominated price series trends upward in nominal terms. Two stocks can appear to comove tightly in a formation window simply because inflation pushed both upward together — not because of any fundamental economic link. The pair then falls apart in the trading window as company-specific factors dominate and the shared inflation signal disappears.

The fix: deflate every price series by a CPI index before pair selection. Normalized real prices capture genuine co-movement in purchasing-power-adjusted terms. Pairs selected this way share a real economic driver (sector exposure, factor loading) rather than a shared inflation trend.

**Research question:** Do pairs selected on real (CPI-deflated) prices outperform pairs selected on nominal prices in terms of Sharpe ratio and trade win rate on BIST? Is the improvement larger in high-inflation subperiods (2018, 2021–2023)?

This idea has no direct precedent in the literature — GGR and all major replications use US data where inflation is low enough to be irrelevant for pair selection. Applying it to a high-inflation emerging market is genuinely novel.

---

### What Changes in Pair Formation

Only the formation step changes. Everything else (σ-scaling, entry/exit rules, portfolio construction) stays exactly as in GGR.

| Step | Baseline GGR | Real-prices variant |
|---|---|---|
| Formation: normalize prices | `P_t / P_0` (nominal) | `(P_t / CPI_t) / (P_0 / CPI_0)` (real) |
| Formation: compute SSD | On normalized nominal prices | On normalized real prices |
| Trading: compute spread | On normalized **nominal** prices | On normalized **nominal** prices (unchanged) |
| Trading: entry/exit | ±2σ / zero crossing | ±2σ / zero crossing (unchanged) |

The trading window reverts to nominal prices — the real-price adjustment is only a lens for identifying which pairs have genuine co-movement, not a change to how the spread is actually traded.

---

### Data Needed

- **Turkey CPI monthly index** — available from EVDS (evds2.tcmb.gov.tr), series `TP.FG.J0`. The TurkStat base-year monthly CPI index. Downloadable as CSV.
- **Daily interpolation**: CPI is monthly; prices are daily. Interpolate using the last published CPI value for each trading day (i.e., hold the monthly value constant until the next release — a "step" interpolation that respects the information constraint: you cannot use a CPI print before it is released).

---

### Implementation Plan

**Step 1 — Download and interpolate CPI**
```python
# cpi_deflator.py
def load_cpi_daily(cpi_monthly: pd.Series, price_index: pd.DatetimeIndex) -> pd.Series:
    """Reindex monthly CPI to daily using forward-fill (no look-ahead)."""
    cpi = cpi_monthly.reindex(price_index, method="ffill")
    return cpi
```

**Step 2 — Deflate prices in the formation window**
```python
def deflate_prices(prices: pd.DataFrame, cpi_daily: pd.Series) -> pd.DataFrame:
    """Divide each price by the CPI index to get real (constant-TRY) prices."""
    cpi_aligned = cpi_daily.reindex(prices.index).ffill()
    return prices.div(cpi_aligned, axis=0)
```

**Step 3 — Modify `select_pairs`**
Add a `real_prices` parameter. When provided, compute SSD on `real_prices` instead of `prices` during the formation window. Return the same pair objects (with nominal σ) for trading.

**Step 4 — Compare portfolios**
Run three portfolios over the full BIST history:
- Baseline GGR (nominal pair selection)
- Real-prices GGR (CPI-deflated pair selection)
- Subperiod breakdown: low-inflation years (2010–2017) vs. high-inflation years (2018–2023)

Key metrics: Sharpe ratio, max drawdown, number of pairs selected per round, pair overlap between the two methods (how many pairs are the same?).

---

### Expected Outcome

In high-inflation periods, the real-prices method should select fewer "inflation phantom" pairs and more pairs with genuine fundamental linkage. The expected result is a higher win rate per trade and a higher Sharpe ratio, with the improvement concentrated in the 2018–2023 high-inflation subperiod. In low-inflation years the two methods should converge (similar pair selection, similar performance), which would serve as an internal validity check.

---

---

## Idea 3: Cross-Listing Arbitrage (BIST vs. LSE GDRs)

### Concept

Several major Turkish firms trade simultaneously on BIST and the London Stock Exchange as Global Depositary Receipts (GDRs): Garanti BBVA, İşbank, Sabancı Holding, Turkcell, and others. Each GDR represents a fixed number of underlying BIST shares. In theory, after converting for the GDR ratio and USD/TRY exchange rate, the LSE price and the BIST price should be identical. In practice, they diverge — timezone gaps (BIST closes earlier), liquidity differences, and capital flow restrictions create short-lived mispricings.

This is a fundamentally different flavor from intra-BIST distance pairs: the same company across two markets, with a hard theoretical arbitrage bound. Convergence is driven by arbitrageurs rather than mean reversion of a statistical spread — which makes the signal cleaner but execution harder.

**Research question:** How large and persistent are BIST/LSE price discrepancies for liquid Turkish GDRs? Can a statistical strategy (hold until parity) generate risk-adjusted returns after accounting for FX conversion costs and the BIST T+2 / LSE T+2 settlement mismatch?

---

### What Constitutes a Trade

| Leg | Instrument | When to go long |
|---|---|---|
| BIST leg | BIST ordinary share (e.g., GARAN.IS) | BIST price < theoretical parity |
| LSE leg | GDR (e.g., GARAN.L) | LSE GDR price > theoretical parity |

Theoretical parity: `GDR_price_USD = (BIST_price_TRY / GDR_ratio) / USD_TRY`

A divergence exists when the actual GDR price differs from theoretical parity by more than transaction costs (typically >0.5%).

---

### Data Needed

- **BIST prices** — already in `bist/data/prices.csv` for GARAN.IS, ISCTR.IS, SAHOL.IS, TCELL.IS
- **LSE GDR prices** — downloadable via yfinance: `GARAN.L`, `ISCTR.L`, `SAHOL.L`, `TCELL.L`
- **USD/TRY daily prices** — yfinance `USDTRY=X`
- **GDR ratios** — fixed conversion ratios (e.g., Garanti: 1 GDR = 5 BIST shares). Available from LSE GDR prospectuses.

---

### Implementation Plan

**Step 1 — Compute theoretical parity spread**
```python
# gdr_arbitrage.py
def compute_parity_spread(bist_price, gdr_price_usd, usdtry, gdr_ratio):
    """Spread = GDR_actual - GDR_theoretical. Positive = GDR overpriced vs. BIST."""
    theoretical_gdr = (bist_price / gdr_ratio) / usdtry
    return gdr_price_usd - theoretical_gdr
```

**Step 2 — Entry/exit rules**
- Enter when `|spread| > threshold` (e.g., 0.5% of theoretical price)
- Exit when spread reverts to zero (parity restored)
- Track in USD terms (convert BIST P&L back to USD at daily close)

**Step 3 — Compare to GGR baseline**
Key metrics: annualized return, Sharpe, average holding period, number of round-trips per year per pair, correlation with BIST market return.

---

### Expected Outcome

GDR discrepancies should be smaller and shorter-lived than intra-BIST statistical spreads — this is near-hard arbitrage, not statistical. Expected holding periods of 1–3 days. The main risk is FX: a sharp TRY move during a holding period can wipe out the arbitrage gain. The strategy should have low market beta (market-neutral by construction) but high FX beta.

---

## Idea 4: OU Half-Life Adaptive Entry Threshold

### Concept

GGR uses a single formation-σ threshold to enter all pairs: open when the spread exceeds 2σ, regardless of how fast that spread typically reverts. Two pairs can both hit 2σ divergence, but one might revert in 5 days while the other takes 60. The slow reverter ties up capital for two months on a trade with low expected annualized return; the fast reverter is genuinely profitable.

The fix: fit an Ornstein-Uhlenbeck (OU) process to each pair's spread during the formation window. The OU fit gives two parameters — the equilibrium mean-reversion speed θ (which determines half-life) and the long-run volatility σ_OU. Use these to set pair-specific thresholds: only trade pairs with half-life < H_max days, and scale the entry threshold by σ_OU rather than formation σ.

**Research question:** Does filtering on OU half-life and using pair-specific thresholds improve Sharpe ratio on BIST compared to uniform 2σ entry?

---

### OU Model

The spread `S_t` follows: `dS = θ(μ - S)dt + σ dW`

Key derived quantities:
- **Half-life**: `t_{1/2} = ln(2) / θ` — expected time for spread to revert halfway to mean
- **OU σ**: `σ_OU = σ / sqrt(2θ)` — long-run equilibrium standard deviation of the spread
- **Entry threshold**: `2 × σ_OU` (replaces formation σ)

---

### Data Needed

No new data — only the existing BIST `prices.csv`. The OU parameters are estimated from the formation-period spread of each candidate pair.

---

### Implementation Plan

**Step 1 — Fit OU process to each formation-period spread**
```python
from scipy.optimize import minimize

def fit_ou(spread: pd.Series) -> dict:
    """MLE fit of OU parameters to formation-period spread."""
    # Discretized OU: S_{t+1} = a + b*S_t + eps, eps ~ N(0, s^2)
    # theta = -log(b)/dt, mu = a/(1-b), sigma_ou = s/sqrt(1 - b^2) * sqrt(dt)
    dt = 1  # daily
    S = spread.values
    b = np.cov(S[1:], S[:-1])[0, 1] / np.var(S[:-1])
    a = np.mean(S[1:]) - b * np.mean(S[:-1])
    resid = S[1:] - (a + b * S[:-1])
    s = np.std(resid)
    theta = -np.log(b) / dt
    half_life = np.log(2) / theta
    sigma_ou = s / np.sqrt(1 - b**2)
    return {"theta": theta, "half_life": half_life, "sigma_ou": sigma_ou}
```

**Step 2 — Filter and threshold**
- Discard pairs with `half_life > 40` trading days (≈2 months)
- Use `2 × sigma_ou` as entry threshold instead of formation σ
- Exit at zero crossing (unchanged from GGR)

**Step 3 — Compare portfolios**
- Baseline GGR (uniform 2σ, no half-life filter)
- OU-adaptive GGR (pair-specific threshold + half-life screen)

Key metrics: Sharpe, average holding period, number of trades, trade win rate.

---

### Expected Outcome

The half-life filter should remove slow-reverting pairs that tie up capital. The OU-σ threshold should produce more consistent entry points across heterogeneous pairs. Expected result: fewer trades, shorter average holding periods, higher win rate per trade, improved Sharpe. The trade-off is reduced number of active positions per round.

---

## Idea 5: 3σ Stop-Loss (Broken Pair Detection)

### Concept

GGR has no stop-loss. Once a trade is open, it stays open until the spread crosses zero or the 6-month trading window ends. Do & Faff (2010) — the most comprehensive academic follow-up to GGR — show empirically that pairs which reach 3σ divergence rarely revert: they are structurally broken. The spread keeps widening because a fundamental change occurred (merger, earnings shock, regulatory change), not a temporary mispricing.

Adding a hard stop at 3σ should cut the worst losses with minimal impact on profitable trades, since very few winning trades ever reach 3σ before reverting.

**Research question:** What fraction of GGR trades on BIST reach 3σ divergence? Among those that do, what fraction eventually converge vs. stay diverged? Does the 3σ stop improve the Sharpe ratio and max drawdown on BIST?

---

### What Changes

Only the exit logic changes. Everything else stays identical to GGR.

| Exit condition | Baseline GGR | With 3σ stop |
|---|---|---|
| Spread crosses zero | Close for profit ✓ | Close for profit ✓ |
| 6-month window ends | Close at mark-to-market ✓ | Close at mark-to-market ✓ |
| Spread reaches 3σ | No exit — hold | **Close for loss (new)** |

---

### Data Needed

No new data — only existing prices. The 3σ stop is computed from the locked formation-period σ, same as the entry threshold.

---

### Implementation Plan

**Step 1 — Modify the trade exit logic**
In `backtest.py`, inside the trading loop, add a check:
```python
if abs(current_spread) >= 3 * pair.sigma:
    # Close both legs at loss — spread has broken down
    close_trade(pair, reason="stop_loss_3sigma")
```

**Step 2 — Track stop-loss statistics**
Log each stop-loss exit separately from normal zero-crossing exits and window-end exits. Compute:
- Fraction of trades that hit 3σ
- Average P&L on stop-loss trades vs. normal exits
- Whether stopped trades subsequently converge (retrospective test)

**Step 3 — Compare portfolios**
- Baseline GGR (no stop)
- GGR + 3σ stop

Key metrics: Sharpe, max drawdown, win rate split by exit type.

---

### Expected Outcome

Do & Faff (2010) find that adding a 3σ stop improves the Sharpe ratio significantly on US data. On BIST — where macro shocks (TRY crises, rate decisions) cause genuine structural breaks — the improvement should be even larger. The 3σ threshold should be rare enough (by construction) that it does not cut many profitable trades short.

---

## Idea 6: Volume Confirmation Filter

### Concept

GGR enters a trade whenever the normalized price spread exceeds 2σ, regardless of whether the divergence is accompanied by trading activity. A large price move with no volume is a stale quote or data artifact. A large price move in the over-performing leg on high volume means real demand — that leg may have repriced fundamentally and will not revert. A large price move on low volume in the under-performing leg means that leg simply hasn't been traded yet — it is likely to catch up.

The filter: only enter a divergence trade when the over-performing leg has above-average volume AND the under-performing leg has below-average volume. This combination — one leg repriced on real volume, the other lagging on thin volume — is the cleanest signal that the divergence is temporary.

**Research question:** Do volume-confirmed entries have higher win rates and faster reversion than unfiltered GGR entries on BIST? Does the improvement persist after accounting for the reduced number of trades?

---

### Volume Rule

For a pair (A, B) at divergence where A is over-performing and B is under-performing:

| Condition | Check |
|---|---|
| A has high volume | A's today volume > A's 20-day average volume × 1.0 |
| B has low volume | B's today volume < B's 20-day average volume × 1.0 |

Both conditions must hold to enter. If only one holds, skip the entry.

The 20-day window is forward-consistent (uses only past volume), avoiding look-ahead bias.

---

### Data Needed

- **BIST volume data** — already available in `bist/data/` (volume.csv alongside prices.csv, downloaded via yfinance)

---

### Implementation Plan

**Step 1 — Compute rolling average volume**
```python
def volume_confirmed(volume: pd.DataFrame, outperformer: str,
                     underperformer: str, date, window=20) -> bool:
    """Return True if volume pattern supports entry."""
    avg_vol = volume.loc[:date].tail(window).mean()
    today_vol = volume.loc[date]
    high_vol = today_vol[outperformer] > avg_vol[outperformer]
    low_vol = today_vol[underperformer] < avg_vol[underperformer]
    return high_vol and low_vol
```

**Step 2 — Modify entry logic**
Add an optional `require_volume_confirm=True` flag to `run_ggr_portfolio`. When enabled, call `volume_confirmed()` before each entry.

**Step 3 — Compare portfolios**
- Baseline GGR (no volume filter)
- GGR + volume confirmation

Key metrics: Sharpe, win rate, average trade P&L, number of trades entered (volume filter will reduce this).

---

### Expected Outcome

Volume-confirmed entries should have a higher win rate per trade (cleaner signal) but fewer total trades (many divergences are filtered out). The net effect on Sharpe depends on how many good trades are excluded. On BIST, where thin liquidity and large-block trades are common, volume signals should be noisier but still informative. The expectation is a moderate improvement in win rate (~5–10 percentage points) with a ~20–30% reduction in number of trades.

---

## Idea 7: Earnings Announcement Entry Prioritization

### Concept

Standard GGR enters purely on spread divergence — it doesn't know why two stocks diverged. When one leg just had an earnings announcement and the other hasn't yet, the divergence is likely transient post-earnings noise: the market is repricing one stock on new information while the other stock simply hasn't moved yet (waiting for its own announcement, or the market hasn't fully processed the cross-company implications). These entries should converge faster than the average GGR trade.

The idea: track earnings announcement dates for all BIST stocks. When a spread entry signal fires and one leg reported earnings within the past 2 trading days while the other did not, flag this as an "earnings-triggered" entry. Compare the win rate and holding period of earnings-triggered entries vs. non-earnings entries.

**Research question:** Do earnings-triggered divergences revert faster and more reliably than other GGR entries on BIST? Can upweighting these entries improve the portfolio Sharpe?

---

### What Changes

The core GGR logic is unchanged. The earnings calendar adds a classification layer on top of entry signals:

| Entry type | Condition | Action |
|---|---|---|
| Earnings-triggered | One leg reported earnings within 2 days, other did not | Enter — expected fast reversion |
| Non-earnings | Neither leg had recent earnings | Enter as normal |
| Both-earnings | Both legs reported earnings recently | Skip — spread may be fundamental |

---

### Data Needed

- **BIST earnings calendar** — quarterly earnings announcement dates for all BIST stocks. Available from KAP (Public Disclosure Platform, kap.org.tr) or IsYatirim's earnings calendar. Downloadable as CSV or via web scraping.
- **No new price data needed.**

---

### Implementation Plan

**Step 1 — Build earnings calendar**
```python
# earnings_calendar.py
def load_earnings_calendar(path: str) -> pd.DataFrame:
    """Load KAP earnings announcement dates. Returns DataFrame with
    columns [ticker, announcement_date]."""

def is_recent_earnings(ticker: str, date, calendar: pd.DataFrame,
                       window=2) -> bool:
    """True if ticker had an earnings announcement within `window` trading days."""
    recent = calendar[calendar["ticker"] == ticker]
    diffs = (date - recent["announcement_date"]).dt.days
    return (diffs >= 0).any() and (diffs[diffs >= 0].min() <= window)
```

**Step 2 — Classify entries**
At each entry signal, call `is_recent_earnings()` for both legs. Label the trade as `earnings_triggered`, `non_earnings`, or `both_earnings` (skip).

**Step 3 — Compare entry types**
Run the full portfolio with earnings classification. Analyze:
- Win rate by entry type
- Average holding period by entry type
- Average P&L by entry type
- Whether upweighting earnings-triggered entries (or skipping non-earnings entries) improves Sharpe

---

### Expected Outcome

Earnings-triggered entries should have shorter average holding periods (the under-reacting leg catches up within 1–5 days once the market processes the news) and higher win rates. The improvement should be most visible in sub-portfolios formed around Turkey's quarterly earnings season (March, May, August, November). The main risk: on BIST, post-earnings moves can be large and prolonged due to thin liquidity, potentially making some earnings-triggered divergences *more* persistent rather than less.

---

*More ideas to be added.*
