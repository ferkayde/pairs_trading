# tests/test_metrics.py
import numpy as np
import pandas as pd
import pytest
from src.metrics import sharpe_ratio, sortino_ratio, monte_carlo_test


class TestSharpeRatio:
    def test_positive_returns_positive_sharpe(self):
        rets = pd.Series([0.001] * 252)
        assert sharpe_ratio(rets) > 0

    def test_zero_returns_zero_sharpe(self):
        rets = pd.Series([0.0] * 252)
        assert sharpe_ratio(rets) == 0.0

    def test_annualization(self):
        # daily return 0.001, daily std 0.001 -> Sharpe = sqrt(252)
        rets = pd.Series([0.001] * 252)
        expected = np.sqrt(252)
        assert abs(sharpe_ratio(rets) - expected) < 0.01


class TestSortinoRatio:
    def test_only_positive_returns_high_sortino(self):
        rets = pd.Series([0.002] * 252)
        assert sortino_ratio(rets) > 5

    def test_mixed_returns_nonnegative(self):
        rng = np.random.default_rng(1)
        rets = pd.Series(rng.normal(0.001, 0.01, 252))
        assert sortino_ratio(rets) >= 0


class TestMonteCarlo:
    def test_returns_expected_keys(self):
        rng = np.random.default_rng(0)
        rets = pd.Series(rng.normal(0.001, 0.01, 252))
        result = monte_carlo_test(rets, n_simulations=100, seed=0)
        assert {"observed_sharpe", "p_value", "null_sharpes"} <= result.keys()

    def test_high_sharpe_strategy_low_pvalue(self):
        rets = pd.Series([0.003] * 252)
        result = monte_carlo_test(rets, n_simulations=500, seed=42)
        assert result["p_value"] < 0.05

    def test_random_strategy_high_pvalue(self):
        rng = np.random.default_rng(7)
        rets = pd.Series(rng.normal(0.0, 0.01, 252))
        result = monte_carlo_test(rets, n_simulations=500, seed=42)
        assert result["p_value"] > 0.05
