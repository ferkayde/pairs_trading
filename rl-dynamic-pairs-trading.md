# Dynamic Pairs Trading with Reinforcement Learning

Replacing the static open-at-2σ / close-at-mean rule from Gatev, Goetzmann & Rouwenhorst (2006) with a learned trading policy. The research question: **does a learned policy beat the static threshold rule out of sample, net of transaction costs?**

---

## 1. Project Overview

Classic pairs trading identifies co-moving pairs, tracks the normalized spread between them, and trades on fixed deviation thresholds. This is a hand-crafted policy. Reinforcement learning lets us learn the policy directly from data: the agent observes the spread's state and decides whether to open a long-spread position, open a short-spread position, hold, or close.

The static GGR rule becomes the **benchmark baseline**, which is a major advantage: you always have a meaningful comparison, and the paper practically writes itself around the comparison table.

## 2. Pipeline Architecture

```
Stage 1: Pair formation (reuse existing infrastructure)
  └── Distance method / cointegration on formation window
Stage 2: Spread construction
  └── Normalized spread series + features per pair
Stage 3: RL environment (Gymnasium)
  └── One episode = one trading window for one pair
Stage 4: Training (Stable-Baselines3: DQN or PPO)
Stage 5: Walk-forward evaluation vs. static 2σ baseline
Stage 6: Analysis: PnL, Sharpe, drawdowns, trade stats, significance tests
```

## 3. Environment Design

### State space (observation vector)

Per timestep, for a given pair:

| Feature | Rationale |
|---|---|
| z-score of spread | The core signal the static rule uses |
| z-score lags (e.g., t-1, t-5) | Lets the agent see momentum/reversion of the spread |
| Rolling spread volatility (e.g., 20d) | Regime information; thresholds should adapt to vol |
| Half-life of mean reversion (rolling OU fit) | How fast reversion is expected |
| Days since position opened | Time-stop information |
| Current position (-1, 0, +1) | Makes the problem Markovian w.r.t. the policy |
| Unrealized PnL of open position | Enables learned stop-loss / take-profit behavior |
| Days remaining in trading window | End-of-episode awareness (forced close at end) |

Normalize all features (z-score or min-max over the formation window only — never using future data).

### Action space (discrete, size 3 or 4)

- `0` — flat / close any open position
- `1` — long spread (long undervalued leg, short overvalued leg)
- `2` — short spread
- (optional) `3` — hold current position explicitly, if you want to separate "stay" from "close"

Keep it discrete. Continuous position sizing is a later extension, not v1.

### Reward function

```
r_t = ΔPnL_t (mark-to-market, in spread units or bps)
      - c · |Δposition_t|          # transaction cost per position change
      - λ · |position_t| · σ_t     # optional risk penalty (start with λ = 0)
```

- Transaction cost `c`: use a realistic per-leg cost for BIST (commission + half-spread). Results **flip sign** without this — it is the single most important modeling choice.
- Consider reward scaling/clipping for training stability.
- Alternative: differential Sharpe ratio reward — try only after plain PnL works.

### Episode structure

- Mirror GGR: 12-month formation window → 6-month trading window.
- One episode = one (pair, trading window) tuple. Forced position close at episode end (matches GGR delisting/window-end handling).
- Training set = many episodes across many pairs and many historical windows. This is critical: a single pair gives far too little data.

## 4. Training Setup

- **Framework:** Gymnasium environment + Stable-Baselines3.
- **Algorithm:** start with **DQN** (discrete actions, simple). PPO as a robustness check.
- **Network:** small MLP (2 hidden layers, 64 units). Bigger networks overfit noise.
- **Hyperparameters to actually tune:** learning rate, exploration schedule (DQN), reward cost coefficient `c`. Leave the rest at SB3 defaults initially.
- **Seeds:** train with ≥5 random seeds and report mean ± std across seeds. RL results in finance are notoriously seed-sensitive; single-seed results are not credible.

## 5. Evaluation Protocol (the part that makes or breaks the project)

1. **Walk-forward splits only.** E.g., train on windows from 2010–2018, validate on 2019–2020, test on 2021–2024 (adjust to your BIST data range). Never a random train/test split — spread dynamics are autocorrelated and you will leak information.
2. **Baselines to beat:**
   - Static 2σ open / 0 close (GGR rule) — the headline comparison
   - Static rule with tuned threshold (grid-search 1.5σ–3σ on validation) — a fairer fight
   - Buy-and-hold-flat (do nothing) — sanity floor
3. **Metrics:** cumulative PnL net of costs, annualized Sharpe, max drawdown, number of trades, win rate, average holding period, turnover.
4. **Significance:** reuse the Monte Carlo machinery from the GGR replication — bootstrap the PnL difference between RL policy and static rule, report confidence intervals.
5. **Behavioral analysis:** plot the learned policy's effective open/close thresholds as a function of volatility state. If the agent has learned something interpretable (e.g., wider thresholds in high-vol regimes), that's the most interesting figure in the whole project.

## 6. Known Pitfalls

- **Overfitting is the default outcome.** Small data + flexible policy = memorized noise. Defenses: small network, many pairs/windows in training, early stopping on validation Sharpe, multiple seeds.
- **Look-ahead bias in features.** Every rolling statistic must use only past data. Audit each feature.
- **Non-stationarity.** A policy trained pre-2020 may fail post-2020. Report performance by sub-period, not just aggregate.
- **Survivorship bias.** Handle delisted BIST names the same way as in the GGR replication.
- **Reward hacking via cost model.** If costs are too low, the agent learns to churn. If the trained agent trades every step, the cost model is wrong.
- **Short-selling constraints on BIST.** The short-selling ban periods already researched are directly relevant — either exclude those periods or force the agent flat/long-only during bans and report both.

## 7. Suggested Milestones

1. **Week 1–2:** Gymnasium environment + static-rule agent implemented *inside the environment* (validates the env: static agent's PnL must match the existing GGR replication numbers).
2. **Week 3–4:** DQN training loop, single pair, overfit on purpose to confirm learning works.
3. **Week 5–6:** Full multi-pair training, walk-forward evaluation, seed study.
4. **Week 7:** Analysis, plots, significance tests, write-up.

Milestone 1 is the secret weapon: if the static rule inside your env reproduces your existing replication results, every downstream comparison is trustworthy.

## 8. Stack

```
python 3.11+
gymnasium
stable-baselines3
pandas, numpy
statsmodels        # OU half-life, cointegration
matplotlib
```

## 9. Possible Extensions (only after v1 works)

- Continuous position sizing (PPO with Box action space)
- Cross-pair portfolio agent (allocates risk budget across pairs)
- Regime features (market index vol, rates) in the state
- Comparison of policies trained on distance-method pairs vs. cointegration pairs
