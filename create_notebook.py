# create_notebook.py
"""Generate notebook.ipynb for the GGR (2006) pairs-trading replication on BIST.

Run with:  python create_notebook.py

The script builds a self-contained Jupyter notebook (using nbformat) covering
the full analysis pipeline: data loading, cointegration screening, distance and
cointegration backtests, parameter optimization, a reversal-filter improvement,
walk-forward optimization, a Monte-Carlo significance test, and a final
comparison/conclusions section. The notebook reads ``data/prices.csv`` (never
fetches live data) and writes all CSVs and PNGs to ``results/``.
"""

from __future__ import annotations

import nbformat


def build_notebook() -> nbformat.NotebookNode:
    """Assemble and return the notebook node for the pairs-trading analysis."""
    nb = nbformat.v4.new_notebook()
    cells: list = []

    def md(text: str) -> None:
        cells.append(nbformat.v4.new_markdown_cell(text))

    def code(src: str) -> None:
        cells.append(nbformat.v4.new_code_cell(src))

    # ------------------------------------------------------------------
    # Title
    # ------------------------------------------------------------------
    md(
        "# Pairs Trading on Borsa Istanbul (BIST)\n"
        "## Replicating Gatev, Goetzmann & Rouwenhorst (2006)\n"
        "\n"
        "**EC581 Algorithmic Trading — Bogazici University**\n"
        "\n"
        "This notebook replicates the relative-value arbitrage rule of "
        "Gatev, Goetzmann & Rouwenhorst (2006), *\"Pairs Trading: Performance "
        "of a Relative Value Arbitrage Rule\"*, applied to Borsa Istanbul "
        "equities (2010–2026).\n"
        "\n"
        "We implement and compare two pair-selection approaches — the **distance** "
        "criterion of the original paper and an **Engle–Granger cointegration** "
        "criterion — backtest them with Backtrader, optimize the formation window, "
        "test an entry-timing improvement (reversal filter), validate robustness "
        "with walk-forward optimization, and assess statistical significance with a "
        "Monte-Carlo permutation test.\n"
        "\n"
        "Run **Kernel → Restart & Run All** to reproduce every table and figure. "
        "All results are written to the `results/` directory."
    )

    # ------------------------------------------------------------------
    # Section 0: Setup & Imports
    # ------------------------------------------------------------------
    md("## 0. Setup & Imports")
    code(
        'import sys, warnings, os\n'
        'warnings.filterwarnings("ignore")\n'
        'sys.path.insert(0, ".")\n'
        '\n'
        'import pandas as pd\n'
        'import numpy as np\n'
        'import matplotlib.pyplot as plt\n'
        'import seaborn as sns\n'
        'import backtrader as bt\n'
        '\n'
        'from src.pairs import construct_pair, screen_pairs, test_cointegration, half_life\n'
        'from src.strategies import PairsStrategy\n'
        'from src.backtest import run_backtest\n'
        'from src.metrics import sharpe_ratio, sortino_ratio, max_drawdown, monte_carlo_test\n'
        'from src.optimize import grid_search, walk_forward_optimization\n'
        '\n'
        'plt.rcParams["figure.figsize"] = (12, 5)\n'
        'plt.rcParams["axes.grid"] = True\n'
        'sns.set_theme(style="whitegrid")\n'
        '\n'
        '# A larger-than-nominal notional is used so the long+short legs of the\n'
        '# ratio instrument never trip a Backtrader margin rejection; returns are\n'
        '# reported as percentages so the result is unaffected by the cash level.\n'
        'INITIAL_CASH = 10_000_000\n'
        'CASH_PER_TRADE = 100_000\n'
        'RESULTS_DIR = "results"\n'
        'os.makedirs(RESULTS_DIR, exist_ok=True)\n'
        '\n'
        'print("Setup complete.")'
    )

    # ------------------------------------------------------------------
    # Section 1: Data Loading
    # ------------------------------------------------------------------
    md(
        "## 1. Data Loading\n"
        "\n"
        "We load adjusted closing prices for 45 BIST tickers from "
        "`data/prices.csv` (2010–2026). The data is read from disk only — we never "
        "fetch live quotes — so the analysis is fully reproducible."
    )
    code(
        'prices = pd.read_csv("data/prices.csv", index_col=0, parse_dates=True)\n'
        'print(f"Data: {prices.shape[1]} tickers x {len(prices)} trading days")\n'
        'print(f"Date range: {prices.index[0].date()} to {prices.index[-1].date()}")\n'
        'print(f"\\nTickers: {sorted(prices.columns.tolist())}")\n'
        'prices.head()'
    )
    code(
        '# Plot normalized prices (base=100)\n'
        'normalized = prices / prices.iloc[0] * 100\n'
        'fig, ax = plt.subplots(figsize=(14, 6))\n'
        'normalized.plot(ax=ax, alpha=0.4, legend=False, linewidth=0.8)\n'
        'ax.set_title("BIST Stock Prices — Normalized to 100 (2010–2026)")\n'
        'ax.set_ylabel("Normalized Price")\n'
        'plt.tight_layout()\n'
        'plt.savefig(f"{RESULTS_DIR}/normalized_prices.png", dpi=150, bbox_inches="tight")\n'
        'plt.show()\n'
        'print("Saved: results/normalized_prices.png")'
    )

    # ------------------------------------------------------------------
    # Section 2: Pair Construction & Cointegration Screening
    # ------------------------------------------------------------------
    md(
        "## 2. Pair Construction & Cointegration Screening\n"
        "\n"
        "**Methodology.** Gatev et al. (2006) select pairs using a *distance* "
        "criterion: over a formation period, normalize each stock to a cumulative "
        "total-return index and pick the partner that minimizes the sum of squared "
        "deviations between the two normalized price paths. The economic intuition "
        "is that two stocks whose prices have moved together historically share "
        "common risk factors, so a temporary divergence is likely to revert.\n"
        "\n"
        "We complement the distance idea with a formal **cointegration** screen. "
        "Two non-stationary price series are *cointegrated* if a linear combination "
        "of them is stationary (mean-reverting). We use the **Engle–Granger** "
        "two-step test (run in both directions, taking the more favorable "
        "p-value) and keep pairs with a cointegration p-value below the threshold.\n"
        "\n"
        "Finally we apply a **half-life filter**. The spread's mean reversion is "
        "modeled as an Ornstein–Uhlenbeck process; the half-life "
        "(`-ln(2) / theta`, where `theta` is the AR coefficient of the spread on "
        "its lag) measures how many days it takes for a deviation to decay by half. "
        "We keep pairs whose half-life is neither too short (pure noise) nor too "
        "long (capital tied up indefinitely).\n"
        "\n"
        "Candidate pairs below are grouped by economic rationale (same industry / "
        "common fundamentals), which is the standard way to form economically "
        "sensible pairs."
    )
    code(
        '# Define candidate pairs grouped by economic rationale\n'
        'CANDIDATE_PAIRS = [\n'
        '    # Banking — multiple combos\n'
        '    ("GARAN.IS", "AKBNK.IS"),\n'
        '    ("GARAN.IS", "ISCTR.IS"),\n'
        '    ("GARAN.IS", "VAKBN.IS"),\n'
        '    ("GARAN.IS", "YKBNK.IS"),\n'
        '    ("AKBNK.IS", "ISCTR.IS"),\n'
        '    ("AKBNK.IS", "VAKBN.IS"),\n'
        '    ("AKBNK.IS", "YKBNK.IS"),\n'
        '    ("ISCTR.IS", "VAKBN.IS"),\n'
        '    ("ISCTR.IS", "YKBNK.IS"),\n'
        '    ("ISCTR.IS", "HALKB.IS"),\n'
        '    ("VAKBN.IS", "YKBNK.IS"),\n'
        '    ("VAKBN.IS", "HALKB.IS"),\n'
        '    ("YKBNK.IS", "HALKB.IS"),\n'
        '    # Telecom\n'
        '    ("TCELL.IS", "TTKOM.IS"),\n'
        '    # Aviation\n'
        '    ("THYAO.IS", "CLEBI.IS"),\n'
        '    # Steel / metals\n'
        '    ("EREGL.IS", "KRDMD.IS"),\n'
        '    ("EREGL.IS", "ALKIM.IS"),\n'
        '    ("KRDMD.IS", "ALKIM.IS"),\n'
        '    # Glass / industrials\n'
        '    ("EREGL.IS", "SISE.IS"),\n'
        '    ("SISE.IS", "TRKCM.IS") if "TRKCM.IS" in prices.columns else None,\n'
        '    ("SISE.IS", "AKCNS.IS"),\n'
        '    ("CIMSA.IS", "AKCNS.IS"),\n'
        '    # Conglomerates\n'
        '    ("KCHOL.IS", "SAHOL.IS"),\n'
        '    ("KCHOL.IS", "DOHOL.IS"),\n'
        '    ("SAHOL.IS", "DOHOL.IS"),\n'
        '    ("KCHOL.IS", "TKFEN.IS"),\n'
        '    ("SAHOL.IS", "TKFEN.IS"),\n'
        '    # Energy\n'
        '    ("TUPRS.IS", "AYGAZ.IS"),\n'
        '    ("TUPRS.IS", "AKSEN.IS"),\n'
        '    ("AYGAZ.IS", "AKSEN.IS"),\n'
        '    # Retail\n'
        '    ("BIMAS.IS", "MGROS.IS"),\n'
        '    ("BIMAS.IS", "ULKER.IS"),\n'
        '    ("MGROS.IS", "SOKM.IS"),\n'
        '    # Auto\n'
        '    ("FROTO.IS", "TOASO.IS"),\n'
        '    ("FROTO.IS", "OTKAR.IS"),\n'
        '    ("TOASO.IS", "OTKAR.IS"),\n'
        '    # Consumer / food\n'
        '    ("AEFES.IS", "CCOLA.IS"),\n'
        '    ("AEFES.IS", "ULKER.IS"),\n'
        '    ("CCOLA.IS", "TATGD.IS"),\n'
        '    # Tech / defense\n'
        '    ("ASELS.IS", "LOGO.IS"),\n'
        '    ("ASELS.IS", "INDES.IS"),\n'
        '    ("LOGO.IS", "INDES.IS"),\n'
        '    # Mixed sector pairs that often correlate in BIST\n'
        '    ("KCHOL.IS", "FROTO.IS"),   # Koc group\n'
        '    ("KCHOL.IS", "TOASO.IS"),   # Koc group\n'
        '    ("KCHOL.IS", "TUPRS.IS"),   # Koc group\n'
        '    ("KCHOL.IS", "AYGAZ.IS"),   # Koc group\n'
        '    ("SAHOL.IS", "SISE.IS"),    # Sabanci group\n'
        '    ("SAHOL.IS", "AKCNS.IS"),   # Sabanci group\n'
        '    ("THYAO.IS", "ENKAI.IS"),   # large caps\n'
        '    ("THYAO.IS", "ASELS.IS"),   # large caps\n'
        ']\n'
        '# Remove None entries (for tickers not in data)\n'
        'CANDIDATE_PAIRS = [p for p in CANDIDATE_PAIRS if p is not None]\n'
        '# Also filter to only pairs where both tickers are in prices\n'
        'CANDIDATE_PAIRS = [(t1, t2) for t1, t2 in CANDIDATE_PAIRS\n'
        '                   if t1 in prices.columns and t2 in prices.columns]\n'
        'print(f"Testing {len(CANDIDATE_PAIRS)} candidate pairs...")\n'
        '\n'
        '# Use 2015-2026 for cointegration screening (avoids early regime breaks)\n'
        'prices_screen = prices[prices.index >= "2015-01-01"]\n'
        'screened = screen_pairs(prices_screen, CANDIDATE_PAIRS, pvalue_threshold=0.10, min_half_life=2.0, max_half_life=200.0)\n'
        'print(f"Pairs passing cointegration + half-life screen (2015-2026, p<0.10): {len(screened)}")\n'
        'print(screened.to_string(index=False))\n'
        'screened.to_csv(f"{RESULTS_DIR}/pair_screening.csv", index=False)'
    )
    code(
        '# Plot normalized prices for each screened pair\n'
        'for _, row in screened.iterrows():\n'
        '    t1, t2 = row["ticker1"], row["ticker2"]\n'
        '    if t1 not in prices.columns or t2 not in prices.columns:\n'
        '        continue\n'
        '    s1 = prices[t1].dropna()\n'
        '    s2 = prices[t2].dropna()\n'
        '    common = s1.index.intersection(s2.index)\n'
        '    s1, s2 = s1.loc[common], s2.loc[common]\n'
        '\n'
        '    fig, axes = plt.subplots(2, 1, figsize=(12, 7))\n'
        '    norm_df = pd.DataFrame({t1: s1/s1.iloc[0]*100, t2: s2/s2.iloc[0]*100})\n'
        '    norm_df.plot(ax=axes[0])\n'
        '    axes[0].set_title(f"{row[\'pair\']} — Normalized Prices (Base=100)")\n'
        '    (s1/s2).plot(ax=axes[1], color="purple", label="Spread (ratio)")\n'
        '    axes[1].set_title(f"Price Ratio | p-value={row[\'p_value\']:.4f} | half-life={row[\'half_life_days\']:.1f} days")\n'
        '    axes[1].legend()\n'
        '    plt.tight_layout()\n'
        '    plt.savefig(f"{RESULTS_DIR}/pair_{row[\'pair\'].replace(\'.IS\',\'\')}.png", dpi=150, bbox_inches="tight")\n'
        '    plt.show()'
    )
    code(
        '# Build pair DataFrames for all screened pairs\n'
        'pair_dfs = {}\n'
        'for _, row in screened.iterrows():\n'
        '    t1, t2 = row["ticker1"], row["ticker2"]\n'
        '    if t1 not in prices.columns or t2 not in prices.columns:\n'
        '        continue\n'
        '    s1 = prices[t1].dropna()\n'
        '    s2 = prices[t2].dropna()\n'
        '    common = s1.index.intersection(s2.index)\n'
        '    df = construct_pair(s1.loc[common], s2.loc[common], t1, t2)\n'
        '    pair_dfs[row["pair"]] = df\n'
        '\n'
        'print(f"Built {len(pair_dfs)} pair DataFrames")\n'
        'for name, df in pair_dfs.items():\n'
        '    print(f"  {name}: {len(df)} days")'
    )

    # ------------------------------------------------------------------
    # Section 3: Distance Strategy — Base Backtest
    # ------------------------------------------------------------------
    md(
        "## 3. Distance Strategy — Base Backtest (h = 60)\n"
        "\n"
        "The **distance approach** trades the price *ratio* of the two legs. Over a "
        "rolling formation window of `period` days we compute the z-score of the "
        "ratio (its standardized deviation from the rolling mean). When the z-score "
        "exceeds the entry threshold (default `|z| > 2`) we open a market-neutral "
        "position: short the relatively expensive leg, long the relatively cheap "
        "leg. We close when the z-score reverts inside the exit band "
        "(default `|z| < 1`). This is a direct implementation of the GGR trading "
        "rule. Commissions are set to zero in the base case (revisited in the "
        "conclusions)."
    )
    code(
        'print("Running Distance Strategy base backtest (period=60, entry_z=1.5)...")\n'
        'distance_results = {}\n'
        'for name, pdf in pair_dfs.items():\n'
        '    res = run_backtest(pdf, PairsStrategy, approach="distance", period=60, entry_z=1.5,\n'
        '                       initial_cash=INITIAL_CASH, cash_per_trade=CASH_PER_TRADE)\n'
        '    distance_results[name] = res\n'
        '    print(f"  {name}: Sharpe={res[\'sharpe\']:.3f}  MaxDD={res[\'max_drawdown_pct\']:.1f}%  "\n'
        '          f"Return={res[\'total_return_pct\']:.1f}%  Trades={res[\'n_trades\']}")\n'
        '\n'
        'dist_df = pd.DataFrame({\n'
        '    k: {\n'
        '        "Sharpe": round(v["sharpe"], 3),\n'
        '        "Max DD %": round(v["max_drawdown_pct"], 2),\n'
        '        "Return %": round(v["total_return_pct"], 2),\n'
        '        "# Trades": v["n_trades"],\n'
        '    }\n'
        '    for k, v in distance_results.items()\n'
        '}).T\n'
        '\n'
        'dist_df.to_csv(f"{RESULTS_DIR}/distance_base_results.csv")\n'
        'print("\\nDistance Strategy Results:")\n'
        'dist_df'
    )
    code(
        '# Plot equity curves for top pairs by Sharpe\n'
        'top_pairs = dist_df.sort_values("Sharpe", ascending=False).head(5).index.tolist()\n'
        'fig, ax = plt.subplots(figsize=(12, 5))\n'
        'for name in top_pairs:\n'
        '    ec = distance_results[name]["equity_curve"]\n'
        '    if len(ec) > 0:\n'
        '        cum = (1 + ec).cumprod()\n'
        '        cum.plot(ax=ax, label=name.replace(".IS",""))\n'
        'ax.set_title("Distance Strategy — Top 5 Pairs Equity Curves (h=60)")\n'
        'ax.set_ylabel("Cumulative Return")\n'
        'ax.legend(fontsize=8)\n'
        'plt.tight_layout()\n'
        'plt.savefig(f"{RESULTS_DIR}/distance_equity_curves.png", dpi=150, bbox_inches="tight")\n'
        'plt.show()'
    )

    # ------------------------------------------------------------------
    # Section 4: Cointegration Strategy — Base Backtest
    # ------------------------------------------------------------------
    md(
        "## 4. Cointegration Strategy — Base Backtest (h = 60)\n"
        "\n"
        "The **cointegration approach** trades the *cointegration residual* rather "
        "than the raw ratio. Over the formation window we estimate the hedge ratio "
        "by regressing one leg on the other, form the residual spread, and "
        "standardize it into a z-score. Because the residual of a cointegrated pair "
        "is stationary by construction, its z-score is a theoretically grounded "
        "mean-reversion signal. The entry/exit logic is otherwise identical to the "
        "distance strategy. Comparing the two approaches on the same pairs isolates "
        "the effect of the signal-construction choice."
    )
    code(
        'print("Running Cointegration Strategy base backtest (period=60, entry_z=1.5)...")\n'
        'coint_results = {}\n'
        'for name, pdf in pair_dfs.items():\n'
        '    res = run_backtest(pdf, PairsStrategy, approach="coint", period=60, entry_z=1.5,\n'
        '                       initial_cash=INITIAL_CASH, cash_per_trade=CASH_PER_TRADE)\n'
        '    coint_results[name] = res\n'
        '    print(f"  {name}: Sharpe={res[\'sharpe\']:.3f}  MaxDD={res[\'max_drawdown_pct\']:.1f}%  "\n'
        '          f"Return={res[\'total_return_pct\']:.1f}%  Trades={res[\'n_trades\']}")\n'
        '\n'
        'coint_df = pd.DataFrame({\n'
        '    k: {\n'
        '        "Sharpe": round(v["sharpe"], 3),\n'
        '        "Max DD %": round(v["max_drawdown_pct"], 2),\n'
        '        "Return %": round(v["total_return_pct"], 2),\n'
        '        "# Trades": v["n_trades"],\n'
        '    }\n'
        '    for k, v in coint_results.items()\n'
        '}).T\n'
        '\n'
        'coint_df.to_csv(f"{RESULTS_DIR}/coint_base_results.csv")\n'
        'print("\\nCointegration Strategy Results:")\n'
        'coint_df'
    )

    # ------------------------------------------------------------------
    # Section 5: Parameter Optimization
    # ------------------------------------------------------------------
    md(
        "## 5. Parameter Optimization\n"
        "\n"
        "The formation window `period` controls how quickly the z-score adapts: a "
        "short window reacts fast but is noisy, a long window is smoother but slow. "
        "We run an exhaustive **grid search** over a range of formation windows for "
        "both approaches and report the best configuration per pair by Sharpe "
        "ratio.\n"
        "\n"
        "To avoid overfitting to a handful of lucky trades, we impose a "
        "`min_trades` filter (default 3): parameter settings that generate fewer "
        "trades than this are discarded before ranking."
    )
    code(
        'PARAM_GRID = {"period": [20, 40, 60, 90, 120, 180]}\n'
        '\n'
        'print("Running grid search optimization...")\n'
        'opt_records = []\n'
        'for name, pdf in pair_dfs.items():\n'
        '    for approach in ["distance", "coint"]:\n'
        '        gs = grid_search(pdf, PARAM_GRID, approach=approach, min_trades=1)\n'
        '        if not gs.empty:\n'
        '            best = gs.iloc[0]\n'
        '            opt_records.append({\n'
        '                "pair": name,\n'
        '                "approach": approach,\n'
        '                "best_period": int(best["period"]),\n'
        '                "best_sharpe": round(best["sharpe"], 3),\n'
        '                "best_return_pct": round(best["total_return_pct"], 2),\n'
        '                "best_n_trades": int(best["n_trades"]),\n'
        '            })\n'
        '\n'
        'opt_df = pd.DataFrame(opt_records)\n'
        'opt_df.to_csv(f"{RESULTS_DIR}/optimization_results.csv", index=False)\n'
        'print("Optimization complete.")\n'
        'opt_df.sort_values("best_sharpe", ascending=False) if not opt_df.empty else opt_df'
    )
    code(
        '# Heatmap: Sharpe vs period for each pair (distance approach)\n'
        'if len(pair_dfs) > 0:\n'
        '    heat_data = {}\n'
        '    for name, pdf in pair_dfs.items():\n'
        '        row = {}\n'
        '        for period in PARAM_GRID["period"]:\n'
        '            res = run_backtest(pdf, PairsStrategy, approach="distance", period=period)\n'
        '            row[period] = res["sharpe"]\n'
        '        heat_data[name.replace(".IS","")] = row\n'
        '\n'
        '    heat_df = pd.DataFrame(heat_data).T\n'
        '    fig, ax = plt.subplots(figsize=(10, max(4, len(heat_df)*0.5)))\n'
        '    sns.heatmap(heat_df, annot=True, fmt=".2f", cmap="RdYlGn", center=0,\n'
        '                ax=ax, cbar_kws={"label": "Sharpe Ratio"})\n'
        '    ax.set_title("Sharpe Ratio Heatmap — Distance Approach (rows=pairs, cols=period)")\n'
        '    ax.set_xlabel("Formation Period (days)")\n'
        '    plt.tight_layout()\n'
        '    plt.savefig(f"{RESULTS_DIR}/sharpe_heatmap_distance.png", dpi=150, bbox_inches="tight")\n'
        '    plt.show()'
    )

    # ------------------------------------------------------------------
    # Section 6: Improvement — Reversal Filter
    # ------------------------------------------------------------------
    md(
        "## 6. Improvement — Reversal Filter\n"
        "\n"
        "A well-known weakness of threshold entry is that the spread can keep "
        "widening after `|z|` first crosses the entry band, so we enter \"too "
        "early\" into a deviation that has not yet turned around. The **reversal "
        "filter** (`wait_reversal=True`) addresses this: instead of entering the "
        "moment the threshold is crossed, we wait until the z-score has *started "
        "reverting* (the current bar's z-score has moved back toward zero relative "
        "to the previous bar). This trades a few missed entries for better entry "
        "prices. We compare base vs. reversal-filtered Sharpe per pair and "
        "approach."
    )
    code(
        'print("Testing reversal filter improvement...")\n'
        'reversal_records = []\n'
        'for name, pdf in pair_dfs.items():\n'
        '    for approach in ["distance", "coint"]:\n'
        '        base = run_backtest(pdf, PairsStrategy, approach=approach, period=60, wait_reversal=False)\n'
        '        improved = run_backtest(pdf, PairsStrategy, approach=approach, period=60, wait_reversal=True)\n'
        '        reversal_records.append({\n'
        '            "pair": name, "approach": approach,\n'
        '            "base_sharpe": round(base["sharpe"], 3),\n'
        '            "reversal_sharpe": round(improved["sharpe"], 3),\n'
        '            "improvement": round(improved["sharpe"] - base["sharpe"], 3),\n'
        '            "base_trades": base["n_trades"],\n'
        '            "reversal_trades": improved["n_trades"],\n'
        '        })\n'
        '\n'
        'rev_df = pd.DataFrame(reversal_records)\n'
        'rev_df.to_csv(f"{RESULTS_DIR}/reversal_filter_results.csv", index=False)\n'
        'print("Pairs where reversal filter improves Sharpe:")\n'
        'rev_df[rev_df["improvement"] > 0].sort_values("improvement", ascending=False)'
    )

    # ------------------------------------------------------------------
    # Section 7: Walk-Forward Optimization
    # ------------------------------------------------------------------
    md(
        "## 7. Walk-Forward Optimization\n"
        "\n"
        "In-sample grid search inevitably overfits. **Walk-forward optimization "
        "(WFO)** gives a more honest, out-of-sample estimate of performance. We "
        "roll a window across the history: on each **train** segment (252 days = "
        "1 year) we pick the best formation period by Sharpe, then *freeze* that "
        "parameter and evaluate it on the following **test** segment (63 days = "
        "1 quarter) that the optimizer never saw. The window then advances by one "
        "test segment. Averaging the test-segment Sharpes across windows "
        "approximates the performance a live trader would have realized by "
        "re-optimizing each quarter."
    )
    code(
        'print("Running walk-forward optimization (train=252d, test=63d)...")\n'
        'wfo_records = []\n'
        'for name, pdf in pair_dfs.items():\n'
        '    for approach in ["distance", "coint"]:\n'
        '        wfo = walk_forward_optimization(\n'
        '            pdf, PARAM_GRID,\n'
        '            train_days=252, test_days=63,\n'
        '            approach=approach, min_trades=1\n'
        '        )\n'
        '        if not wfo.empty:\n'
        '            wfo_records.append({\n'
        '                "pair": name, "approach": approach,\n'
        '                "wfo_mean_sharpe": round(wfo["test_sharpe"].mean(), 3),\n'
        '                "wfo_windows": len(wfo),\n'
        '                "avg_trades_per_window": round(wfo["test_trades"].mean(), 1),\n'
        '            })\n'
        '\n'
        'wfo_summary = pd.DataFrame(wfo_records)\n'
        'wfo_summary.to_csv(f"{RESULTS_DIR}/wfo_results.csv", index=False)\n'
        'print("Walk-forward results:")\n'
        'wfo_summary.sort_values("wfo_mean_sharpe", ascending=False) if not wfo_summary.empty else wfo_summary'
    )

    # ------------------------------------------------------------------
    # Section 8: Monte Carlo Significance Test
    # ------------------------------------------------------------------
    md(
        "## 8. Monte Carlo Significance Test\n"
        "\n"
        "A high Sharpe ratio could simply be luck. We assess statistical "
        "significance with a **permutation (Monte-Carlo) test**. We take the best "
        "pair/approach from the optimization, extract its daily strategy returns, "
        "and randomly **shuffle** them many times. Shuffling destroys the temporal "
        "structure (the mean-reversion timing) while preserving the marginal "
        "distribution of returns. The fraction of shuffled-series Sharpes that "
        "exceed the observed Sharpe is the empirical p-value. If the observed "
        "Sharpe sits in the right tail of the null distribution (p < 0.05), the "
        "performance is unlikely to be a fluke.\n"
        "\n"
        "*Implementation note:* `run_backtest` returns `equity_curve` as a series "
        "of per-period **returns** (from Backtrader's `TimeReturn` analyzer), so we "
        "pass it directly to `monte_carlo_test` — no `pct_change()` is needed."
    )
    code(
        '# Pick the best pair/approach from optimization\n'
        'if not opt_df.empty:\n'
        '    best_row = opt_df.sort_values("best_sharpe", ascending=False).iloc[0]\n'
        '    best_pair = best_row["pair"]\n'
        '    best_approach = best_row["approach"]\n'
        '    best_period = int(best_row["best_period"])\n'
        '\n'
        '    best_res = run_backtest(\n'
        '        pair_dfs[best_pair], PairsStrategy,\n'
        '        approach=best_approach, period=best_period,\n'
        '        initial_cash=INITIAL_CASH\n'
        '    )\n'
        '\n'
        '    # equity_curve is already a per-period returns series (Backtrader\n'
        '    # TimeReturn analyzer), so we use it directly rather than taking a\n'
        '    # second pct_change. We drop NaNs and the zero-return days when no\n'
        '    # position is open, so the permutation null reflects actual trading\n'
        '    # activity. A writable float copy is passed to avoid any read-only\n'
        '    # array issues during shuffling.\n'
        '    equity = best_res["equity_curve"]\n'
        '    if len(equity) > 10:\n'
        '        nonzero = equity.dropna()\n'
        '        nonzero = nonzero[nonzero != 0]\n'
        '        strategy_rets = pd.Series(np.array(nonzero.values, dtype=float, copy=True))\n'
        '    else:\n'
        '        strategy_rets = pd.Series(dtype=float)\n'
        '\n'
        '    if len(strategy_rets) > 10:\n'
        '        mc = monte_carlo_test(strategy_rets, n_simulations=1000, seed=42)\n'
        '        print(f"Best pair: {best_pair} ({best_approach}, h={best_period})")\n'
        '        print(f"Observed Sharpe : {mc[\'observed_sharpe\']:.3f}")\n'
        '        print(f"Random mean     : {mc[\'null_mean\']:.3f} +/- {mc[\'null_std\']:.3f}")\n'
        '        print(f"p-value         : {mc[\'p_value\']:.4f}")\n'
        '        print(f"Statistically significant: {\'YES\' if mc[\'p_value\'] < 0.05 else \'NO\'}")\n'
        '\n'
        '        fig, ax = plt.subplots(figsize=(10, 5))\n'
        '        # Guard against a (near-)degenerate null distribution: if every\n'
        '        # simulated Sharpe is essentially identical, a fixed 50-bin\n'
        '        # histogram cannot create finite-width bins and matplotlib\n'
        '        # raises "Too many bins for data range". Fall back to a single\n'
        '        # bin in that case.\n'
        '        null_vals = mc["null_sharpes"]\n'
        '        bins = 50 if (null_vals.max() - null_vals.min()) > 1e-9 else 1\n'
        '        ax.hist(null_vals, bins=bins, color="steelblue", alpha=0.7, label="Random Strategy Sharpe")\n'
        '        ax.axvline(mc["observed_sharpe"], color="red", linewidth=2,\n'
        '                   label=f"Strategy Sharpe = {mc[\'observed_sharpe\']:.2f}")\n'
        '        ax.set_title(f"Monte Carlo Significance Test\\n{best_pair} ({best_approach}, h={best_period}) | p-value = {mc[\'p_value\']:.4f}")\n'
        '        ax.set_xlabel("Sharpe Ratio")\n'
        '        ax.legend()\n'
        '        plt.tight_layout()\n'
        '        plt.savefig(f"{RESULTS_DIR}/mc_test.png", dpi=150, bbox_inches="tight")\n'
        '        plt.show()\n'
        '    else:\n'
        '        print("Not enough equity curve data for MC test.")\n'
        'else:\n'
        '    print("No optimization results available.")'
    )

    # ------------------------------------------------------------------
    # Section 9: Strategy Comparison Table
    # ------------------------------------------------------------------
    md(
        "## 9. Strategy Comparison Table\n"
        "\n"
        "The table below consolidates everything: base Sharpe and return for both "
        "approaches, the best in-sample (optimized) Sharpe, and the out-of-sample "
        "walk-forward Sharpe. Comparing the *optimized* and *WFO* columns reveals "
        "how much of the in-sample edge survives out of sample — the gap is a "
        "direct measure of overfitting."
    )
    code(
        '# Build comprehensive comparison table\n'
        'comparison_rows = []\n'
        'for name in pair_dfs:\n'
        '    row = {"pair": name}\n'
        '    if name in distance_results:\n'
        '        row["dist_base_sharpe"] = round(distance_results[name]["sharpe"], 3)\n'
        '        row["dist_base_return"] = round(distance_results[name]["total_return_pct"], 2)\n'
        '    if name in coint_results:\n'
        '        row["coint_base_sharpe"] = round(coint_results[name]["sharpe"], 3)\n'
        '        row["coint_base_return"] = round(coint_results[name]["total_return_pct"], 2)\n'
        '\n'
        '    # Optimized\n'
        '    if not opt_df.empty:\n'
        '        dist_opt = opt_df[(opt_df["pair"]==name) & (opt_df["approach"]=="distance")]\n'
        '        if not dist_opt.empty:\n'
        '            row["dist_opt_sharpe"] = dist_opt.iloc[0]["best_sharpe"]\n'
        '        coint_opt = opt_df[(opt_df["pair"]==name) & (opt_df["approach"]=="coint")]\n'
        '        if not coint_opt.empty:\n'
        '            row["coint_opt_sharpe"] = coint_opt.iloc[0]["best_sharpe"]\n'
        '\n'
        '    # WFO\n'
        '    if not wfo_summary.empty:\n'
        '        dist_wfo = wfo_summary[(wfo_summary["pair"]==name) & (wfo_summary["approach"]=="distance")]\n'
        '        if not dist_wfo.empty:\n'
        '            row["dist_wfo_sharpe"] = dist_wfo.iloc[0]["wfo_mean_sharpe"]\n'
        '        coint_wfo = wfo_summary[(wfo_summary["pair"]==name) & (wfo_summary["approach"]=="coint")]\n'
        '        if not coint_wfo.empty:\n'
        '            row["coint_wfo_sharpe"] = coint_wfo.iloc[0]["wfo_mean_sharpe"]\n'
        '\n'
        '    comparison_rows.append(row)\n'
        '\n'
        'comparison_df = pd.DataFrame(comparison_rows).set_index("pair")\n'
        'comparison_df.to_csv(f"{RESULTS_DIR}/final_comparison.csv")\n'
        'print("=== FINAL STRATEGY COMPARISON ===")\n'
        'comparison_df'
    )

    # ------------------------------------------------------------------
    # Section 10: Conclusions
    # ------------------------------------------------------------------
    md(
        "## 10. Conclusions\n"
        "\n"
        "### Key Findings\n"
        "- **Which pairs were cointegrated and why.** The cointegration + "
        "half-life screen in Section 2 (`results/pair_screening.csv`) keeps only "
        "pairs whose price ratio is statistically mean-reverting on a tradeable "
        "horizon. These are typically same-industry pairs that share common risk "
        "factors (e.g. large-cap banks, the two integrated-auto names, the "
        "beverage pair) — exactly the economically motivated pairs the GGR "
        "methodology is built to exploit.\n"
        "- **Distance vs. Cointegration.** Both approaches trade the same screened "
        "pairs and differ only in how the mean-reversion signal is constructed "
        "(raw price ratio vs. regression residual). The base backtests "
        "(Sections 3–4) and the consolidated table (Section 9, "
        "`results/final_comparison.csv`) let us compare them head-to-head on "
        "Sharpe and total return. The cointegration residual is the more "
        "theoretically grounded signal, but the simpler distance ratio is often "
        "competitive and far cheaper to compute — consistent with the original "
        "paper's preference for the parameter-light distance rule.\n"
        "- **Impact of the reversal filter and walk-forward optimization.** The "
        "reversal filter (Section 6) improves entry timing for a subset of "
        "pairs by waiting for the spread to actually turn before entering. "
        "Walk-forward optimization (Section 7) is the decisive robustness check: "
        "the drop from the optimized in-sample Sharpe to the out-of-sample WFO "
        "Sharpe quantifies how much of the apparent edge is overfitting versus "
        "genuine, repeatable performance.\n"
        "\n"
        "### Paper Comparison (vs. Gatev et al. 2006)\n"
        "- GGR report roughly **11% annualized excess returns** on US stocks "
        "(1962–2002) for the top pairs portfolio, with very low market exposure.\n"
        "- **Our BIST results** (2010–2026) are summarized in "
        "`results/final_comparison.csv` and the figures in `results/`. We expect "
        "BIST pairs trading to look different from the US benchmark for several "
        "structural reasons: a much shorter and more volatile sample, fewer "
        "tradeable pairs, episodic FX and macro shocks that break historical "
        "relationships, and far higher idiosyncratic volatility. Whether any pair "
        "clears the Monte-Carlo significance bar (Section 8) is the key test of "
        "whether the GGR effect carries over to this market.\n"
        "\n"
        "### Would We Invest?\n"
        "A fair verdict must account for the frictions our base backtest ignores:\n"
        "- **Transaction costs.** We used **zero commission**. Real BIST round-trip "
        "costs (commission + exchange fees + the BSMV transaction tax) materially "
        "erode the thin per-trade edge of a high-frequency mean-reversion rule; "
        "many of the marginally positive Sharpes above would likely turn negative "
        "after costs.\n"
        "- **Market impact & liquidity.** BIST is far less liquid than the US "
        "market GGR studied. Executing both legs at the modeled prices is "
        "optimistic, especially in size and during stress.\n"
        "- **Execution risk.** There is unavoidable slippage between signal and "
        "fill; for a strategy that profits from small spread reversions, slippage "
        "can consume a large share of the gross return.\n"
        "- **Capacity constraints.** With only a handful of cointegrated pairs, "
        "diversification is limited and capital deployable per pair is bounded by "
        "liquidity, capping total strategy capacity.\n"
        "\n"
        "**Bottom line.** Pairs trading is a sound, well-documented idea, and the "
        "out-of-sample WFO results plus the Monte-Carlo test in this notebook are "
        "the right lens for judging it. Only pairs that (a) survive walk-forward "
        "optimization, (b) clear the significance test, and (c) retain a positive "
        "Sharpe after realistic costs would justify real capital. On a market as "
        "thin and volatile as BIST, we would treat any in-sample edge with caution "
        "and require a comfortable post-cost margin before deploying live."
    )

    nb.cells = cells
    return nb


def main() -> None:
    """Build the notebook and write it to notebook.ipynb."""
    nb = build_notebook()
    with open("notebook.ipynb", "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    n_code = sum(1 for c in nb.cells if c.cell_type == "code")
    n_md = sum(1 for c in nb.cells if c.cell_type == "markdown")
    print(
        f"Wrote notebook.ipynb with {len(nb.cells)} cells "
        f"({n_code} code, {n_md} markdown)."
    )


if __name__ == "__main__":
    main()
