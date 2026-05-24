# tests/test_pairs.py
import pandas as pd
import numpy as np
import pytest
from src.pairs import construct_pair, test_cointegration, half_life, screen_pairs


def _make_cointegrated_series(n=500, seed=42):
    """Generate two cointegrated price series for testing."""
    rng = np.random.default_rng(seed)
    common = np.cumsum(rng.normal(0, 1, n))
    s1 = common + rng.normal(0, 0.5, n)
    s2 = common + rng.normal(0, 0.5, n)
    idx = pd.date_range("2015-01-01", periods=n, freq="B")
    return pd.Series(np.exp(s1 / 100), index=idx), pd.Series(np.exp(s2 / 100), index=idx)


def _make_random_series(n=500, seed=99):
    """Generate two independent random walks."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2015-01-01", periods=n, freq="B")
    s1 = pd.Series(np.exp(np.cumsum(rng.normal(0, 1, n)) / 100), index=idx)
    s2 = pd.Series(np.exp(np.cumsum(rng.normal(0, 1, n)) / 100), index=idx)
    return s1, s2


class TestConstructPair:
    def test_ratio_is_price1_over_price2(self):
        s1, s2 = _make_cointegrated_series()
        pair = construct_pair(s1, s2, "A", "B")
        pd.testing.assert_series_equal(pair["Close"], s1 / s2, check_names=False)

    def test_has_required_columns(self):
        s1, s2 = _make_cointegrated_series()
        pair = construct_pair(s1, s2, "A", "B")
        assert set(pair.columns) == {"Close", "Price1", "Price2"}

    def test_name_attribute(self):
        s1, s2 = _make_cointegrated_series()
        pair = construct_pair(s1, s2, "GARAN", "AKBNK")
        assert pair.attrs["name"] == "GARAN_AKBNK"

    def test_index_preserved(self):
        s1, s2 = _make_cointegrated_series()
        pair = construct_pair(s1, s2, "A", "B")
        pd.testing.assert_index_equal(pair.index, s1.index)


class TestCointegration:
    def test_cointegrated_pair_low_pvalue(self):
        s1, s2 = _make_cointegrated_series()
        result = test_cointegration(s1, s2)
        assert result["p_value"] < 0.05

    def test_random_pair_high_pvalue(self):
        s1, s2 = _make_random_series()
        result = test_cointegration(s1, s2)
        assert result["p_value"] > 0.05

    def test_result_has_required_keys(self):
        s1, s2 = _make_cointegrated_series()
        result = test_cointegration(s1, s2)
        assert {"t_stat", "p_value", "is_cointegrated"} <= result.keys()


class TestHalfLife:
    def test_half_life_positive(self):
        s1, s2 = _make_cointegrated_series()
        spread = s1 / s2
        hl = half_life(spread)
        assert hl > 0

    def test_fast_reverting_spread_short_half_life(self):
        """A spread with strong mean reversion should have a short half-life."""
        rng = np.random.default_rng(7)
        n = 500
        idx = pd.date_range("2015-01-01", periods=n, freq="B")
        spread = pd.Series(index=idx, dtype=float)
        spread.iloc[0] = 0.0
        for i in range(1, n):
            spread.iloc[i] = spread.iloc[i - 1] * 0.5 + rng.normal(0, 0.1)
        hl = half_life(spread)
        assert 0 < hl < 10


class TestScreenPairs:
    def test_returns_dataframe(self):
        s1, s2 = _make_cointegrated_series()
        prices = pd.concat([s1.rename("A"), s2.rename("B")], axis=1)
        pairs_to_test = [("A", "B")]
        result = screen_pairs(prices, pairs_to_test)
        assert isinstance(result, pd.DataFrame)

    def test_cointegrated_pair_passes_screen(self):
        s1, s2 = _make_cointegrated_series()
        prices = pd.concat([s1.rename("A"), s2.rename("B")], axis=1)
        result = screen_pairs(prices, [("A", "B")])
        assert len(result) == 1

    def test_random_pair_filtered_out(self):
        s1, s2 = _make_random_series()
        prices = pd.concat([s1.rename("X"), s2.rename("Y")], axis=1)
        result = screen_pairs(prices, [("X", "Y")])
        assert len(result) == 0
