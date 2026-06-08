"""
merge_external_data.py — Add delisted and missing BIST tickers from external source.

Reads price data from bist_others_data/Kitap*.xlsx (IsYatirim export format),
extracts the 7 tickers not covered by Yahoo Finance, and merges them into
bist/data/prices.csv.

Source format (each Kitap file):
  Row 0-1: headers
  Rows 2+: ticker (col 1), open (col 2), close (col 3), date (col 4)

Prices are dividend-adjusted (verified: ratio to Yahoo ≈ 0.998, std ≈ 0.002).

Run from the project root:
    python bist/merge_external_data.py
"""

from pathlib import Path
import pandas as pd

KITAP_FILES = [
    "bist_others_data/Kitap1.xlsx",
    "bist_others_data/Kitap2.xlsx",
    "bist_others_data/Kitap3.xlsx",
    "bist_others_data/Kitap4.xlsx",
]
PRICES_PATH = Path("bist/data/prices.csv")

# Tickers present in Kitap files but missing from bist/data/prices.csv.
# ROYAL.IS: delisted 2022-12-22 (genuine survivorship bias fix)
# YGYO.IS: removed from XUTUM index Feb 2025 (survivorship bias fix)
# Others: active stocks not captured in Yahoo Finance download.
NEW_TICKERS = {"OSMEN.IS", "GLBMD.IS", "ROYAL.IS", "QNBTR.IS", "YGYO.IS", "GATEG.IS", "KLNMA.IS"}


def load_new_tickers() -> pd.DataFrame:
    """Parse all Kitap files and return wide-format DataFrame for NEW_TICKERS only."""
    all_rows = []
    for fpath in KITAP_FILES:
        df = pd.read_excel(fpath, sheet_name="Sayfa1", header=None)
        data = df.iloc[2:, [1, 3, 4]].copy()
        data.columns = ["ticker", "close", "date"]
        data = data[data["ticker"].isin(NEW_TICKERS)].copy()
        data["date"] = pd.to_datetime(data["date"], errors="coerce")
        data["close"] = pd.to_numeric(data["close"], errors="coerce")
        data = data.dropna()
        all_rows.append(data)

    combined = pd.concat(all_rows).drop_duplicates(subset=["ticker", "date"])
    # Pivot to wide format: index=date, columns=ticker, values=close
    wide = combined.pivot(index="date", columns="ticker", values="close")
    wide.index.name = "Date"
    wide.columns.name = None
    return wide.sort_index()


def merge_into_prices(new_wide: pd.DataFrame) -> None:
    print(f"Loading existing prices: {PRICES_PATH}")
    prices = pd.read_csv(PRICES_PATH, index_col=0, parse_dates=True)
    print(f"  Existing: {prices.shape[1]} tickers x {len(prices)} days")

    # Reindex new tickers to match existing date index (NaN for dates outside range)
    new_aligned = new_wide.reindex(prices.index)

    # Concatenate new columns
    merged = pd.concat([prices, new_aligned], axis=1)
    print(f"  After merge: {merged.shape[1]} tickers x {len(merged)} days")

    for ticker in NEW_TICKERS:
        if ticker in new_wide.columns:
            n_days = new_wide[ticker].count()
            print(f"  Added {ticker}: {n_days} trading days "
                  f"({new_wide.index[new_wide[ticker].notna()].min().date()} to "
                  f"{new_wide.index[new_wide[ticker].notna()].max().date()})")

    merged.to_csv(PRICES_PATH)
    print(f"Saved merged prices -> {PRICES_PATH}")


if __name__ == "__main__":
    print("Parsing external price data from Kitap files ...")
    new_wide = load_new_tickers()
    print(f"Extracted {len(new_wide.columns)} new tickers, {len(new_wide)} unique dates\n")
    merge_into_prices(new_wide)
    print("\nDone. Run bist/notebook.ipynb to regenerate results with the expanded universe.")
