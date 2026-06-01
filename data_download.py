"""
data_download.py — Run ONCE to fetch price and volume data and save to CSV.
All subsequent notebook runs read from data/prices.csv and data/volume.csv.

Reads tickers from clean_turkish_tickers.txt (one ticker per line, no .IS suffix).
Appends .IS suffix for Yahoo Finance Istanbul Stock Exchange lookup.
"""

import yfinance as yf
import pandas as pd
from pathlib import Path

TICKERS_FILE = Path("clean_turkish_tickers.txt")
FALLBACK_TICKERS = [
    "GARAN", "AKBNK", "ISCTR", "YKBNK", "VAKBN", "HALKB",
    "TCELL", "TTKOM", "THYAO", "PGSUS",
    "EREGL", "KRDMD", "SISE", "TRKCM", "CIMSA", "AKCNS",
    "KCHOL", "SAHOL", "DOHOL", "TKFEN",
    "TUPRS", "AYGAZ", "AKSEN",
    "KOZAL", "KOZAA",
    "BIMAS", "MGROS", "SOKM", "ULKER",
    "TOASO", "FROTO", "OTKAR",
    "EMLAK", "ENKAI",
    "ASELS", "LOGO",
    "AEFES", "CCOLA", "TATGD",
    "ALKIM",
]

START = "2010-01-01"
PRICES_PATH = Path("data/prices.csv")
VOLUME_PATH = Path("data/volume.csv")


def load_tickers() -> list:
    """Load ticker symbols from file, falling back to hardcoded list."""
    if TICKERS_FILE.exists():
        symbols = [
            line.strip() for line in TICKERS_FILE.read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ]
        return [f"{s}.IS" for s in symbols]
    return [f"{s}.IS" for s in FALLBACK_TICKERS]


def download_data(tickers: list, start: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download adjusted close prices and volume for all tickers.

    Applies the GGR (2006) data preparation:
    - No forward-filling (liquidity filter will screen out tickers with gaps).
    - Drops tickers with >10% missing price observations.
    - Saves prices to data/prices.csv and volume to data/volume.csv.

    Returns (prices, volume) DataFrames.
    """
    print(f"Downloading {len(tickers)} tickers from {start} …")
    raw = yf.download(tickers, start=start, auto_adjust=True, progress=True)

    prices = raw["Close"].copy()
    volume = raw["Volume"].copy()

    # Drop tickers with >10% missing price observations
    thresh = int(0.90 * len(prices))
    prices = prices.dropna(axis=1, thresh=thresh)
    volume = volume[prices.columns]  # keep aligned

    # Drop rows where ALL prices are missing
    prices = prices.dropna(how="all")
    volume = volume.reindex(prices.index)

    PRICES_PATH.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(PRICES_PATH)
    volume.to_csv(VOLUME_PATH)

    print(f"Saved {prices.shape[1]} tickers × {len(prices)} days")
    print(f"  → {PRICES_PATH}")
    print(f"  → {VOLUME_PATH}")
    print(f"Tickers: {sorted(prices.columns.tolist())}")
    return prices, volume


if __name__ == "__main__":
    tickers = load_tickers()
    download_data(tickers, START)
