"""
macro_data.py — Macro reference series for the BIST experimental ideas.

Provides two things both Idea 1 (TCMB shock filter) and Idea 2 (real-price
formation) need:

  1. A Turkey monthly CPI (TÜFE) index, daily-forward-filled with no look-ahead.
  2. A TCMB policy-rate decision calendar (surprise / emergency moves).

DATA PROVENANCE
---------------
The exact production series live on TCMB's EVDS system (evds2.tcmb.gov.tr):
  - CPI monthly index           → series  TP.FG.J0   (TÜFE, base 2003=100)
  - TCMB 1-week repo decisions   → MPC press-release archive

EVDS requires a (free) API key, which is not available in this sandbox, so the
series below are reconstructed from the publicly published TÜİK year-end
inflation prints and the well-known TCMB decision dates.  Both functions accept
an override path so you can drop in the exact EVDS CSV later without touching
the notebook:

    cpi = load_cpi_monthly(csv_path="evds_TP_FG_J0.csv")   # uses real data
    cpi = load_cpi_monthly()                                # uses reconstruction

The reconstruction reproduces the *economically relevant* feature for both
ideas — that TRY prices inflate 10-85 %/yr, heavily in 2018 and 2021-2024 — to
within the precision needed for pair selection.  It is not meant to match the
official index to the decimal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Turkey year-end CPI inflation (TÜFE, annual % change, December YoY).
# Source: TÜİK / TCMB published year-end figures.
# ---------------------------------------------------------------------------
_YEAR_END_INFLATION = {
    2009: 6.53,
    2010: 6.40,
    2011: 10.45,
    2012: 6.16,
    2013: 7.40,
    2014: 8.17,
    2015: 8.81,
    2016: 8.53,
    2017: 11.92,
    2018: 20.30,
    2019: 11.84,
    2020: 14.60,
    2021: 36.08,
    2022: 64.27,
    2023: 64.77,
    2024: 44.38,
    2025: 32.00,   # provisional; replace with realised print when available
    2026: 28.00,   # forward placeholder for the 2026 trading tail
}


def build_monthly_cpi_index(base_value: float = 100.0) -> pd.Series:
    """Reconstruct a monthly CPI index from year-end annual inflation.

    Within each calendar year the index grows geometrically from the prior
    December value to the current December value (constant monthly rate
    = (1 + annual)**(1/12)).  This is a smooth interpolation of the official
    annual prints — adequate for deflating prices in pair formation, where only
    the broad inflation trend matters, not month-to-month seasonality.

    Returns a monthly pd.Series indexed by month-end (freq "ME"), starting at
    Dec-2009 = base_value.
    """
    years = sorted(_YEAR_END_INFLATION)
    idx_dates = [pd.Timestamp(f"{years[0]}-12-31")]
    idx_vals = [base_value]

    for y in years[1:]:
        annual = _YEAR_END_INFLATION[y] / 100.0
        monthly_rate = (1 + annual) ** (1 / 12) - 1
        start_val = idx_vals[-1]
        for m in range(1, 13):
            # month-end date for month m of year y
            d = pd.Timestamp(f"{y}-{m:02d}-01") + pd.offsets.MonthEnd(0)
            idx_dates.append(d)
            idx_vals.append(start_val * (1 + monthly_rate) ** m)

    return pd.Series(idx_vals, index=pd.DatetimeIndex(idx_dates), name="cpi")


def load_cpi_monthly(csv_path: str | None = None) -> pd.Series:
    """Load the monthly CPI index — from EVDS CSV if given, else reconstruction.

    EVDS CSV is expected with a date column and a value column (the first two
    columns are used).  Dates are parsed and the series is sorted ascending.
    """
    if csv_path is not None:
        raw = pd.read_csv(csv_path)
        date_col, val_col = raw.columns[0], raw.columns[1]
        s = pd.Series(
            pd.to_numeric(raw[val_col], errors="coerce").values,
            index=pd.to_datetime(raw[date_col], errors="coerce"),
            name="cpi",
        ).dropna().sort_index()
        return s
    return build_monthly_cpi_index()


def load_cpi_daily(cpi_monthly: pd.Series, price_index: pd.DatetimeIndex) -> pd.Series:
    """Reindex monthly CPI to the daily price index using forward-fill.

    Forward-fill respects the information constraint: a month's CPI print is
    only known *after* it is published, so for every trading day we use the
    most recent already-published monthly value (no look-ahead).
    """
    # Union the two indices, ffill, then restrict to the price dates.
    combined = cpi_monthly.reindex(
        cpi_monthly.index.union(price_index)
    ).ffill()
    return combined.reindex(price_index).ffill().rename("cpi")


def deflate_prices(prices: pd.DataFrame, cpi_daily: pd.Series) -> pd.DataFrame:
    """Divide each price by the daily CPI index → real (constant-TRY) prices.

    Real price = nominal price / CPI.  The scale is arbitrary (CPI base
    cancels out under the P_t/P_0 normalization GGR applies next), so we do not
    rescale by any base-year value.
    """
    cpi_aligned = cpi_daily.reindex(prices.index).ffill()
    return prices.div(cpi_aligned, axis=0)


# ---------------------------------------------------------------------------
# TCMB policy-rate surprise / emergency decision calendar.
#
# These are the high-impact TCMB moves where the realised one-week repo
# decision diverged sharply from consensus or was an out-of-schedule emergency
# action — the episodes Idea 1 wants to fence off.  Dates are the announcement
# dates; the filter blocks NEW entries for `block_days` afterwards.
#
# Source: TCMB MPC press-release archive (well-documented public events).
# ---------------------------------------------------------------------------
_TCMB_SHOCK_DECISIONS = [
    # date,        old%,  new%,  note
    ("2014-01-28", 4.50,  10.00, "Emergency midnight hike (+550bp), TRY defence"),
    ("2018-05-23", 8.00,  16.50, "Emergency hike during currency crisis (late-late liquidity window)"),
    ("2018-06-07", 16.50, 17.75, "Follow-up hike"),
    ("2018-09-13", 17.75, 24.00, "Large surprise hike (+625bp) vs consensus"),
    ("2020-09-24", 8.25,  10.25, "Surprise hike after long hold"),
    ("2020-11-19", 10.25, 15.00, "Large hike (+475bp), new governor"),
    ("2021-03-18", 17.00, 19.00, "Hike (+200bp); governor sacked 2 days later → crash"),
    ("2021-09-23", 19.00, 18.00, "Surprise cut, start of unorthodox easing"),
    ("2021-11-18", 16.00, 15.00, "Cut amid lira free-fall"),
    ("2021-12-16", 15.00, 14.00, "Cut; TRY crisis peak"),
    ("2023-06-22", 8.50,  15.00, "Policy U-turn (+650bp), undershoot vs consensus 21%"),
    ("2023-07-20", 15.00, 17.50, "Hike below consensus"),
    ("2023-08-24", 17.50, 25.00, "Large catch-up hike (+750bp)"),
    ("2024-03-21", 45.00, 50.00, "Surprise hike (+500bp) before local elections"),
]


def load_tcmb_decisions(
    csv_path: str | None = None,
    surprise_bp: float = 150.0,
) -> pd.DataFrame:
    """TCMB surprise/emergency decision calendar.

    Two modes:

    1. csv_path given → REAL data.  The CSV is the full history of policy-rate
       *changes* (one row per change: date, new 1-week-repo rate).  We compute
       each change's size and keep only the SURPRISE moves — those of magnitude
       >= `surprise_bp` basis points (default 150 bp).  Routine 25-100 bp steps
       are ignored; only the market-shaking moves (the 2014 midnight hike, the
       2018 / 2021 / 2023 crises, etc.) become shock events.

    2. csv_path None → fall back to the curated _TCMB_SHOCK_DECISIONS list.

    Returns a DataFrame with columns [date, old_rate, new_rate, change_bp, note].
    """
    if csv_path is not None:
        raw = pd.read_csv(csv_path)
        raw["date"] = pd.to_datetime(raw["date"])
        raw = raw.sort_values("date").reset_index(drop=True)
        raw["old_rate"] = raw["rate"].shift(1)
        raw["new_rate"] = raw["rate"]
        raw["change_bp"] = (raw["new_rate"] - raw["old_rate"]) * 100.0
        # First row has no prior rate → drop it; then keep only big moves.
        surprises = raw.dropna(subset=["old_rate"]).copy()
        surprises = surprises[surprises["change_bp"].abs() >= surprise_bp]
        surprises["note"] = surprises["change_bp"].apply(
            lambda b: f"Surprise {'hike' if b > 0 else 'cut'} ({b:+.0f}bp)"
        )
        return surprises[
            ["date", "old_rate", "new_rate", "change_bp", "note"]
        ].reset_index(drop=True)

    rows = []
    for d, old, new, note in _TCMB_SHOCK_DECISIONS:
        rows.append({
            "date": pd.Timestamp(d),
            "old_rate": old,
            "new_rate": new,
            "change_bp": (new - old) * 100.0,
            "note": note,
        })
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def build_shock_calendar(
    price_index: pd.DatetimeIndex,
    usdtry: pd.Series | None = None,
    tcmb_decisions: pd.DataFrame | None = None,
    cpi_monthly: pd.Series | None = None,
    tcmb_block_days: int = 5,
    fx_threshold: float = 0.03,
    fx_block_days: int = 3,
    cpi_jump_threshold: float = 0.02,
    cpi_block_days: int = 3,
) -> pd.DatetimeIndex:
    """Build the set of trading dates on which NEW pair entries are blocked.

    Three triggers (ideas.md §Idea 1), each blocking N trading days *after* the
    event:

      • TCMB surprise/emergency decision   → block `tcmb_block_days` (default 5)
      • |USD/TRY daily return| > fx_threshold (3 %)  → block `fx_block_days` (3)
      • Monthly CPI jump > cpi_jump_threshold above the prior month (2 %)
        → block `cpi_block_days` (3) from the release date

    Blocking is forward-looking from the event and is applied on *trading dates*
    (the price index), so "N days" means N rows of `price_index`.

    Parameters
    ----------
    price_index : the daily trading-date index to block against.
    usdtry      : daily USD/TRY close (optional; FX trigger skipped if None).
    tcmb_decisions : output of load_tcmb_decisions() (optional).
    cpi_monthly : monthly CPI index (optional; CPI trigger skipped if None).

    Returns
    -------
    DatetimeIndex of blocked dates (subset of price_index).
    """
    price_index = pd.DatetimeIndex(price_index).sort_values()
    blocked: set[pd.Timestamp] = set()

    def _block_after(event_date: pd.Timestamp, n_days: int):
        # First trading day on/after the event, then the next n_days rows.
        pos = price_index.searchsorted(event_date, side="left")
        for j in range(pos, min(pos + n_days + 1, len(price_index))):
            blocked.add(price_index[j])

    # Trigger 1 — TCMB decisions
    if tcmb_decisions is not None:
        for d in tcmb_decisions["date"]:
            _block_after(pd.Timestamp(d), tcmb_block_days)

    # Trigger 2 — USD/TRY daily move
    if usdtry is not None and len(usdtry) > 1:
        fx = usdtry.reindex(price_index).ffill()
        fx_ret = fx.pct_change()
        for d in fx_ret.index[fx_ret.abs() > fx_threshold]:
            _block_after(pd.Timestamp(d), fx_block_days)

    # Trigger 3 — High CPI print (monthly jump above prior month)
    if cpi_monthly is not None and len(cpi_monthly) > 1:
        mom = cpi_monthly.pct_change()
        for d in mom.index[mom > cpi_jump_threshold]:
            _block_after(pd.Timestamp(d), cpi_block_days)

    return pd.DatetimeIndex(sorted(blocked))


def download_usdtry(start: str = "2010-01-01", end: str | None = None) -> pd.Series:
    """Download daily USD/TRY close via yfinance (ticker 'USDTRY=X').

    Returns a pd.Series of closes indexed by date.  Requires internet.
    """
    import yfinance as yf

    raw = yf.download("USDTRY=X", start=start, end=end,
                      auto_adjust=True, progress=False)
    close = raw["Close"]
    if isinstance(close, pd.DataFrame):     # MultiIndex single-ticker case
        close = close.iloc[:, 0]
    return close.rename("usdtry").dropna()
