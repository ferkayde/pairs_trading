# tests/test_pairs.py
import numpy as np
import pandas as pd
import pytest

from src.pairs import (
    normalize_prices,
    compute_ssd,
    select_top_pairs,
    compute_locked_sigma,
    liquidity_filter,
    activity_filter,
)


def _prices(data: dict, n: int | None = None) -> pd.DataFrame:
    length = n if n is not None else len(next(iter(data.values())))
    idx = pd.date_range("2020-01-01", periods=length, freq="B")
    return pd.DataFrame(data, index=idx)


# ---------------------------------------------------------------------------
# normalize_prices
# ---------------------------------------------------------------------------

class TestNormalizePrices:
    def test_first_row_is_one(self):
        df = _prices({"A": [100.0, 110.0, 90.0], "B": [50.0, 55.0, 45.0]})
        norm = normalize_prices(df)
        assert norm["A"].iloc[0] == pytest.approx(1.0)
        assert norm["B"].iloc[0] == pytest.approx(1.0)

    def test_ratios_preserved(self):
        df = _prices({"A": [100.0, 120.0, 80.0]})
        norm = normalize_prices(df)
        pd.testing.assert_series_equal(
            norm["A"], pd.Series([1.0, 1.2, 0.8], index=df.index), check_names=False
        )

    def test_nan_in_first_row_handled(self):
        idx = pd.date_range("2020-01-01", periods=3, freq="B")
        df = pd.DataFrame({"A": [np.nan, 100.0, 110.0]}, index=idx)
        norm = normalize_prices(df)
        # first non-NaN is 100 → second row = 1.0, third = 1.1
        assert norm["A"].iloc[1] == pytest.approx(1.0)
        assert norm["A"].iloc[2] == pytest.approx(1.1)


# ---------------------------------------------------------------------------
# compute_ssd
# ---------------------------------------------------------------------------

class TestComputeSSD:
    def test_identical_series_zero_ssd(self):
        df = _prices({"A": [1.0, 1.1, 1.2], "B": [1.0, 1.1, 1.2]})
        ssd_df = compute_ssd(df)
        assert ssd_df["ssd"].iloc[0] == pytest.approx(0.0, abs=1e-10)

    def test_known_value(self):
        # A=[1,2], B=[1,1] → diff=[0,1] → SSD=1
        df = _prices({"A": [1.0, 2.0], "B": [1.0, 1.0]}, n=2)
        ssd_df = compute_ssd(df)
        assert ssd_df["ssd"].iloc[0] == pytest.approx(1.0)

    def test_sorted_ascending(self):
        df = _prices({"A": [1.0, 2.0], "B": [1.0, 1.5], "C": [1.0, 5.0]}, n=2)
        ssd_df = compute_ssd(df)
        assert list(ssd_df["ssd"]) == sorted(ssd_df["ssd"])

    def test_pair_count(self):
        n_tickers = 6
        df = _prices({f"T{i}": [1.0, 1.0] for i in range(n_tickers)}, n=2)
        ssd_df = compute_ssd(df)
        assert len(ssd_df) == n_tickers * (n_tickers - 1) // 2

    def test_output_columns(self):
        df = _prices({"A": [1.0, 1.1], "B": [1.0, 1.2]}, n=2)
        ssd_df = compute_ssd(df)
        assert set(ssd_df.columns) == {"ticker1", "ticker2", "ssd"}

    def test_symmetry(self):
        # SSD(A,B) == SSD(B,A) — only the upper triangle is returned, but value is symmetric
        df = _prices({"A": [1.0, 1.3, 0.9], "B": [1.0, 1.0, 1.0]})
        ssd_df = compute_ssd(df)
        assert len(ssd_df) == 1  # only one pair
        assert ssd_df["ssd"].iloc[0] > 0


# ---------------------------------------------------------------------------
# select_top_pairs
# ---------------------------------------------------------------------------

class TestSelectTopPairs:
    def test_returns_top_n(self):
        ssd_df = pd.DataFrame({
            "ticker1": ["A", "A", "B"],
            "ticker2": ["B", "C", "C"],
            "ssd": [0.1, 0.3, 0.5],
        })
        top = select_top_pairs(ssd_df, n=2)
        assert len(top) == 2

    def test_smallest_ssd_first(self):
        ssd_df = pd.DataFrame({
            "ticker1": ["A", "A"],
            "ticker2": ["B", "C"],
            "ssd": [0.1, 0.9],
        })
        top = select_top_pairs(ssd_df, n=2)
        assert top[0][2] < top[1][2]

    def test_returns_tuples_of_three(self):
        ssd_df = pd.DataFrame({"ticker1": ["A"], "ticker2": ["B"], "ssd": [0.5]})
        top = select_top_pairs(ssd_df, n=1)
        assert isinstance(top[0], tuple) and len(top[0]) == 3

    def test_n_larger_than_df_returns_all(self):
        ssd_df = pd.DataFrame({"ticker1": ["A"], "ticker2": ["B"], "ssd": [0.5]})
        top = select_top_pairs(ssd_df, n=50)
        assert len(top) == 1


# ---------------------------------------------------------------------------
# compute_locked_sigma
# ---------------------------------------------------------------------------

class TestComputeLockedSigma:
    def test_zero_spread_gives_zero_sigma(self):
        df = _prices({"A": [1.0, 1.1, 1.2], "B": [1.0, 1.1, 1.2]})
        sigmas = compute_locked_sigma(df, [("A", "B", 0.0)])
        assert sigmas[("A", "B")] == pytest.approx(0.0)

    def test_positive_sigma_for_diverging_pair(self):
        df = _prices({"A": [1.0, 1.1, 1.2], "B": [1.0, 1.0, 1.0]})
        sigmas = compute_locked_sigma(df, [("A", "B", 0.0)])
        assert sigmas[("A", "B")] > 0

    def test_missing_ticker_skipped(self):
        df = _prices({"A": [1.0, 1.1]}, n=2)
        sigmas = compute_locked_sigma(df, [("A", "MISSING", 0.0)])
        assert ("A", "MISSING") not in sigmas

    def test_keys_are_ticker_tuples(self):
        df = _prices({"X": [1.0, 1.05], "Y": [1.0, 0.95]}, n=2)
        sigmas = compute_locked_sigma(df, [("X", "Y", 0.1)])
        assert ("X", "Y") in sigmas


# ---------------------------------------------------------------------------
# liquidity_filter
# ---------------------------------------------------------------------------

class TestLiquidityFilter:
    def _make_prices(self, n=5):
        idx = pd.date_range("2020-01-01", periods=n, freq="B")
        return pd.DataFrame({"A": [100.0] * n, "B": [50.0] * n}, index=idx)

    def test_complete_prices_pass(self):
        px = self._make_prices()
        result = liquidity_filter(px, px.index[0], px.index[-1])
        assert set(result) == {"A", "B"}

    def test_nan_price_rejected(self):
        px = self._make_prices()
        px.iloc[2, 0] = np.nan  # NaN price for A on day 2
        result = liquidity_filter(px, px.index[0], px.index[-1])
        assert "A" not in result
        assert "B" in result

    def test_all_nan_column_rejected(self):
        px = self._make_prices()
        px["A"] = np.nan
        result = liquidity_filter(px, px.index[0], px.index[-1])
        assert "A" not in result

    def test_no_tickers_with_all_nan(self):
        idx = pd.date_range("2020-01-01", periods=3, freq="B")
        px = pd.DataFrame({"A": [np.nan, np.nan, np.nan]}, index=idx)
        result = liquidity_filter(px, idx[0], idx[-1])
        assert result == []


# ---------------------------------------------------------------------------
# activity_filter
# ---------------------------------------------------------------------------

class TestActivityFilter:
    def _make_window(self, n=20):
        idx = pd.date_range("2020-01-01", periods=n, freq="B")
        # Active: prices move every day
        active = pd.Series([100.0 + i * 0.5 for i in range(n)], index=idx)
        # Dormant: constant price → 100% zero returns
        dormant = pd.Series([100.0] * n, index=idx)
        return pd.DataFrame({"ACTIVE": active, "DORMANT": dormant})

    def test_active_stock_passes(self):
        px = self._make_window()
        result = activity_filter(px, ["ACTIVE", "DORMANT"])
        assert "ACTIVE" in result

    def test_dormant_stock_removed(self):
        px = self._make_window()
        result = activity_filter(px, ["ACTIVE", "DORMANT"])
        assert "DORMANT" not in result

    def test_threshold_respected(self):
        # 5 zero-return days out of 9 returns = ~56% zeros
        # [100,100,100,100,100,101,102,103,104,105] → first 4 returns are 0
        idx = pd.date_range("2020-01-01", periods=10, freq="B")
        prices = [100.0] * 5 + [101.0, 102.0, 103.0, 104.0, 105.0]
        px = pd.DataFrame({"A": prices}, index=idx)
        assert activity_filter(px, ["A"], max_zero_frac=0.30) == []
        assert activity_filter(px, ["A"], max_zero_frac=0.60) == ["A"]

    def test_missing_ticker_skipped(self):
        px = self._make_window()
        result = activity_filter(px, ["ACTIVE", "MISSING"])
        assert "MISSING" not in result
        assert "ACTIVE" in result

    def test_empty_tickers_returns_empty(self):
        px = self._make_window()
        assert activity_filter(px, []) == []
