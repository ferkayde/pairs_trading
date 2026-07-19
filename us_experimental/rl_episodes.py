"""
rl_episodes.py — walk-forward (pair, trading-window) episode builder.

Reuses the GGR pair-formation machinery from src.pairs unchanged (liquidity +
activity filters, SSD ranking, locked sigma), then packages each selected pair's
trading window as one RL episode: price arrays, formation constants, and the
precomputed static feature matrix from rl_features.

The trading bars of an episode are exactly the bars run_ggr_portfolio would
trade (dropna intersection of both legs inside the trading window), so PnL
computed on an episode is directly comparable to the replication engine.
"""

from __future__ import annotations

import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# Make `src` importable when run from us_experimental/.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.pairs import (  # noqa: E402
    activity_filter,
    compute_locked_sigma,
    compute_ssd,
    liquidity_filter,
    normalize_prices,
    select_top_pairs,
)
from us_experimental.rl_features import build_static_features  # noqa: E402


@dataclass
class Episode:
    """One (pair, trading window) tuple — a single RL episode."""

    ticker1: str
    ticker2: str
    formation_start: pd.Timestamp
    trading_start: pd.Timestamp
    trading_end: pd.Timestamp
    dates: pd.DatetimeIndex   # trading bars where both legs have valid prices
    p1: np.ndarray            # leg-1 prices on those bars
    p2: np.ndarray
    p1_0: float               # prices at formation start (normalization base)
    p2_0: float
    locked_sigma: float       # formation std of the normalized spread
    features: np.ndarray      # (length, len(STATIC_FEATURES)) float32

    @property
    def length(self) -> int:
        return len(self.dates)


def build_episodes(
    prices: pd.DataFrame,
    formation_days: int = 252,
    trading_days: int = 126,
    roll_days: int = 21,
    top_n: int = 20,
    min_trading_bars: int = 5,
    vol_window: int = 20,
    hl_window: int = 60,
    verbose: bool = True,
) -> list[Episode]:
    """Walk-forward episode generation over the full price history.

    Steps through history every roll_days, replicating run_ggr_portfolio's
    formation stage (filters -> normalize -> SSD -> top_n -> locked sigma),
    then builds one Episode per selected pair. Pairs whose trading bars are
    fewer than min_trading_bars or contain non-positive prices are skipped.
    """
    all_dates = prices.index
    n_total = len(all_dates)
    episodes: list[Episode] = []
    n_windows_total = max(1, (n_total - formation_days - trading_days) // roll_days + 1)
    window_num = 0

    start_idx = 0
    while start_idx + formation_days + trading_days <= n_total:
        f_start = all_dates[start_idx]
        f_end = all_dates[start_idx + formation_days - 1]
        t_start = all_dates[start_idx + formation_days]
        t_end = all_dates[min(start_idx + formation_days + trading_days - 1, n_total - 1)]

        window_num += 1
        if verbose:
            print(
                f"  Episode window {window_num}/{n_windows_total}  "
                f"trading {t_start.date()} -> {t_end.date()}  "
                f"({len(episodes)} episodes)",
                end="\r", flush=True,
            )

        liquid = liquidity_filter(prices, f_start, f_end)
        liquid = activity_filter(prices.loc[f_start:f_end], liquid)
        if len(liquid) < 4:
            start_idx += roll_days
            continue

        norm_form = normalize_prices(prices.loc[f_start:f_end, liquid])
        ssd_df = compute_ssd(norm_form)
        top_pairs = select_top_pairs(ssd_df, n=min(top_n, len(ssd_df)))
        sigmas = compute_locked_sigma(norm_form, top_pairs)
        p0_row = prices.loc[f_start]

        for t1, t2, _ in top_pairs:
            sigma = sigmas.get((t1, t2))
            if sigma is None or sigma < 1e-10:
                continue
            p1_0 = p0_row.get(t1, np.nan)
            p2_0 = p0_row.get(t2, np.nan)
            if pd.isna(p1_0) or pd.isna(p2_0) or p1_0 <= 0 or p2_0 <= 0:
                continue

            s1 = prices.loc[f_start:t_end, t1]
            s2 = prices.loc[f_start:t_end, t2]
            common = s1.dropna().index.intersection(s2.dropna().index)
            s1c, s2c = s1.loc[common], s2.loc[common]
            trading_mask = common >= t_start
            trading_dates = common[trading_mask]
            if len(trading_dates) < min_trading_bars:
                continue
            p1_arr = s1c.loc[trading_dates].to_numpy(dtype=float)
            p2_arr = s2c.loc[trading_dates].to_numpy(dtype=float)
            if (p1_arr <= 0).any() or (p2_arr <= 0).any():
                continue

            feats = build_static_features(
                s1c, s2c, float(p1_0), float(p2_0), float(sigma), t_start,
                vol_window=vol_window, hl_window=hl_window,
                max_half_life=float(trading_days),
            )
            if len(feats) != len(trading_dates):
                continue  # defensive: feature rows must align with trading bars

            episodes.append(Episode(
                ticker1=t1, ticker2=t2,
                formation_start=f_start, trading_start=t_start, trading_end=t_end,
                dates=trading_dates,
                p1=p1_arr, p2=p2_arr,
                p1_0=float(p1_0), p2_0=float(p2_0),
                locked_sigma=float(sigma),
                features=feats.to_numpy(dtype=np.float32),
            ))

        start_idx += roll_days

    if verbose:
        print(f"\n  Done: {len(episodes)} episodes from {window_num} windows.")
    return episodes


def split_episodes(
    episodes: list[Episode],
    train_end: str = "2015-01-01",
    val_end: str = "2020-01-01",
) -> dict[str, list[Episode]]:
    """Walk-forward split by trading-window dates; straddling windows dropped.

    train : trading_end   <  train_end   (fully out of the validation era)
    val   : trading_start >= train_end and trading_end < val_end
    test  : trading_start >= val_end
    """
    tr_end = pd.Timestamp(train_end)
    v_end = pd.Timestamp(val_end)
    return {
        "train": [e for e in episodes if e.trading_end < tr_end],
        "val": [e for e in episodes
                if e.trading_start >= tr_end and e.trading_end < v_end],
        "test": [e for e in episodes if e.trading_start >= v_end],
    }


def save_episodes(episodes: list[Episode], path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(episodes, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_episodes(path) -> list[Episode]:
    with open(path, "rb") as f:
        return pickle.load(f)
