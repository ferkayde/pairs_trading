"""
data_download.py — Run ONCE to fetch price data and save to CSV.
All subsequent runs read from data/prices.csv.
"""

import yfinance as yf
import pandas as pd
from pathlib import Path

TICKERS = [
    # Banking — private
    "GARAN.IS", "AKBNK.IS", "ISCTR.IS", "YKBNK.IS",
    # Banking — state
    "VAKBN.IS", "HALKB.IS", "ZRGYO.IS",
    # Insurance & financial services
    "ALARK.IS", "EGEEN.IS",
    # Telecom
    "TCELL.IS", "TTKOM.IS",
    # Aviation
    "THYAO.IS", "PGSUS.IS",
    # Steel & metals
    "EREGL.IS", "KRDMD.IS", "ALKIM.IS",
    # Glass, chemicals, building materials
    "SISE.IS", "TRKCM.IS", "CIMSA.IS", "AKCNS.IS",
    # Conglomerates / holdings
    "KCHOL.IS", "SAHOL.IS", "DOHOL.IS", "TKFEN.IS",
    # Energy & petroleum
    "TUPRS.IS", "AYGAZ.IS", "AKSEN.IS", "ZOREN.IS",
    # Mining — Koza group
    "KOZAL.IS", "KOZAA.IS",
    # Retail & consumer
    "BIMAS.IS", "MGROS.IS", "SOKM.IS", "ULKER.IS",
    # Automotive & industrials
    "TOASO.IS", "FROTO.IS", "OTKAR.IS",
    # Real estate & construction
    "EMLAK.IS", "ENKAI.IS", "TABGD.IS",
    # Healthcare & pharma
    "ECILC.IS", "DEVA.IS", "SELEC.IS",
    # Technology & defense
    "ASELS.IS", "LOGO.IS", "INDES.IS",
    # Media & entertainment
    "NTTUR.IS", "KLNMA.IS",
    # Logistics
    "CLEBI.IS", "ULAS.IS",
    # Agriculture & food
    "AEFES.IS", "CCOLA.IS", "TATGD.IS",
    # Cement
    "BOLUC.IS", "GOLTS.IS",
    # Index
    "XU100.IS",
]

START = "2010-01-01"
OUT_PATH = Path("data/prices.csv")


def download_prices(tickers: list, start: str, out_path: Path) -> pd.DataFrame:
    """Download adjusted close prices, align dates, forward-fill gaps <= 3 days.

    Drops any ticker with >5% missing values after forward-filling.
    Saves result to out_path as CSV.
    """
    raw = yf.download(tickers, start=start, auto_adjust=True, progress=True)
    prices = raw["Close"]

    # Forward-fill trading halts up to 3 consecutive days
    prices = prices.ffill(limit=3)

    # Drop columns with >5% missing
    thresh = int(0.95 * len(prices))
    prices = prices.dropna(axis=1, thresh=thresh)

    # Drop rows where ALL prices are missing
    prices = prices.dropna(how="all")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(out_path)
    print(f"Saved {prices.shape[1]} tickers x {len(prices)} days -> {out_path}")
    print(f"Tickers retained: {sorted(prices.columns.tolist())}")
    return prices


if __name__ == "__main__":
    download_prices(TICKERS, START, OUT_PATH)
