"""
data_download.py — Download US equity data for GGR (2006) exact replication.

Fetches the current S&P 500 constituent list from Wikipedia, then downloads
daily adjusted close prices from Yahoo Finance. Run once; the notebook reads
from data/prices.csv on every subsequent run.

S&P 500 tickers are used as a practical approximation of GGR's CRSP universe.
Note: this introduces survivorship bias (only current constituents), consistent
with our BIST implementation.
"""

import yfinance as yf
import pandas as pd
from pathlib import Path

START = "1990-01-01"
PRICES_PATH = Path("data/prices.csv")
VOLUME_PATH = Path("data/volume.csv")
FF_PATH = Path("data/ff_factors.csv")
SECTORS_PATH = Path("data/sectors.csv")

# GGR use 4 broad sector groups based on SIC codes.
# We map GICS sectors (from Wikipedia S&P 500 table) to the same 4 groups.
GICS_TO_GGR = {
    "Utilities":                   "Utilities",
    "Financials":                  "Financials",
    "Real Estate":                 "Financials",
    "Industrials":                 "Industrials",
    "Materials":                   "Industrials",
    "Energy":                      "Industrials",
    "Consumer Discretionary":      "Industrials",
    "Consumer Staples":            "Industrials",
    "Health Care":                 "Industrials",
    "Information Technology":      "Industrials",
    "Communication Services":      "Industrials",
}


def load_sp500_tickers() -> list:
    """Scrape current S&P 500 tickers from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    table = pd.read_html(url)[0]
    tickers = table["Symbol"].tolist()
    # Yahoo Finance uses '-' not '.' for tickers like BRK.B -> BRK-B
    return [t.replace(".", "-") for t in tickers]


def load_sp500_sectors() -> pd.DataFrame:
    """Return DataFrame mapping Yahoo Finance ticker -> GGR sector group."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    table = pd.read_html(url)[0]
    df = table[["Symbol", "GICS Sector"]].copy()
    df["Symbol"] = df["Symbol"].str.replace(".", "-", regex=False)
    df["ggr_sector"] = df["GICS Sector"].map(GICS_TO_GGR).fillna("Industrials")
    df = df.rename(columns={"Symbol": "ticker", "GICS Sector": "gics_sector"})
    df = df.set_index("ticker")
    SECTORS_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(SECTORS_PATH)
    print(f"Saved sector info: {len(df)} stocks -> {SECTORS_PATH}")
    return df


def download_ff_factors(start: str = START) -> pd.DataFrame:
    """Download Fama-French daily factors from Ken French's data library.

    Downloads: Mkt-RF, SMB, HML (F-F 3-factor) + Momentum + Short-Term Reversal.
    GGR Table 4 uses all five factors in their risk-regression.

    Requires pandas_datareader. Falls back to a warning if unavailable.
    Returns daily factor returns as fractions (not percentages).
    """
    try:
        import pandas_datareader.data as web
    except ImportError:
        print("pandas_datareader not installed — skipping FF factor download.")
        print("Install with: pip install pandas-datareader")
        return pd.DataFrame()

    print("Downloading Fama-French daily factors from Ken French data library ...")
    dfs = {}
    sources = {
        "ff3":  "F-F_Research_Data_Factors_daily",
        "mom":  "F-F_Momentum_Factor_daily",
        "strev":"F-F_ST_Reversal_Factor_daily",
    }
    for key, name in sources.items():
        try:
            raw = web.DataReader(name, "famafrench", start=start)[0]
            raw.index = pd.to_datetime(raw.index, format="%Y%m%d")
            dfs[key] = raw / 100.0    # convert from % to fractions
            print(f"  {name}: {len(raw)} days")
        except Exception as e:
            print(f"  WARNING: could not download {name}: {e}")

    if not dfs:
        return pd.DataFrame()

    # Combine into single DataFrame
    factors = dfs.get("ff3", pd.DataFrame())
    if "mom" in dfs:
        factors = factors.join(dfs["mom"][["Mom"]], how="left")
    if "strev" in dfs:
        factors = factors.join(dfs["strev"][["ST_Rev"]], how="left")

    # Keep Mkt-RF, SMB, HML, Mom, ST_Rev (drop RF — already used for excess returns)
    keep = [c for c in ["Mkt-RF", "SMB", "HML", "Mom", "ST_Rev"] if c in factors.columns]
    factors = factors[keep].dropna(how="all")

    FF_PATH.parent.mkdir(parents=True, exist_ok=True)
    factors.to_csv(FF_PATH)
    print(f"Saved FF factors: {factors.shape} -> {FF_PATH}")
    return factors


def download_data(tickers: list, start: str) -> tuple:
    """Download adjusted close prices and volume for all tickers.

    Applies the same GGR data preparation used in the BIST implementation:
    - No forward-filling (liquidity + activity filters screen per window).
    - Keeps tickers with at least one full formation+trading cycle of data.
    - Drops market-closure rows (holidays included by Yahoo Finance as NaN rows).

    Returns (prices, volume) DataFrames.
    """
    print(f"Downloading {len(tickers)} S&P 500 tickers from {start} ...")
    raw = yf.download(tickers, start=start, auto_adjust=True, progress=True)

    prices = raw["Close"].copy()
    volume = raw["Volume"].copy()

    # Keep tickers with at least one full formation+trading cycle (252+126 days)
    min_days = 252 + 126
    prices = prices.dropna(axis=1, thresh=min_days)
    volume = volume[prices.columns]

    # Drop rows where ALL prices are missing
    prices = prices.dropna(how="all")

    # Drop market-closure days: US federal holidays that Yahoo Finance sometimes
    # includes as rows with NaN. Measured against already-listed (active) tickers
    # so early rows with many un-IPO'd stocks are not mistakenly removed.
    first_valid = prices.apply(lambda col: col.first_valid_index())
    first_valid = first_valid.fillna(prices.index[-1] + pd.Timedelta(days=1))
    active = pd.DataFrame(
        {col: prices.index >= first_valid[col] for col in prices.columns},
        index=prices.index,
    )
    n_active = active.sum(axis=1).clip(lower=1)
    nan_among_active = (prices.isna() & active).sum(axis=1) / n_active
    market_open = nan_among_active <= 0.5
    n_dropped = (~market_open).sum()
    if n_dropped:
        print(f"Dropping {n_dropped} market-closure rows (>50% of active tickers NaN)")
    prices = prices.loc[market_open]
    volume = volume.reindex(prices.index)

    PRICES_PATH.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(PRICES_PATH)
    volume.to_csv(VOLUME_PATH)

    print(f"Saved {prices.shape[1]} tickers x {len(prices)} days")
    print(f"  -> {PRICES_PATH}")
    print(f"  -> {VOLUME_PATH}")
    return prices, volume


if __name__ == "__main__":
    tickers = load_sp500_tickers()
    download_data(tickers, START)
    load_sp500_sectors()
    download_ff_factors(START)
