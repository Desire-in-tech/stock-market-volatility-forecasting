"""
tests/test_model.py — Unit tests for model.py (GarchModel).

Run with:
    pytest tests/test_model.py -v
"""

import sqlite3
import sys
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data import SQLRepository
from model import GarchModel


# Fixtures

@pytest.fixture
def sample_returns() -> pd.Series:
    """Synthetic daily returns series for testing (no network calls)."""
    rng = np.random.default_rng(42)
    n = 500
    returns = rng.normal(loc=0.05, scale=1.0, size=n) * 100  # in percentage
    idx = pd.date_range("2020-01-02", periods=n, freq="B")
    return pd.Series(returns, index=idx, name="returns")


@pytest.fixture
def repo_with_returns(sample_returns):
    """In-memory SQLite repo pre-loaded with synthetic OHLCV + returns."""
    conn = sqlite3.connect(":memory:")
    repo = SQLRepository(connection=conn)
    df = pd.DataFrame(
        {
            "Open": 100 + sample_returns.cumsum() * 0.01,
            "High": 101 + sample_returns.cumsum() * 0.01,
            "Low": 99 + sample_returns.cumsum() * 0.01,
            "Close": 100 + sample_returns.cumsum() * 0.01,
            "Volume": 1_000_000,
            "returns": sample_returns.values,
        },
        index=sample_returns.index,
    )
    df.index.name = "Date"
    repo.insert_table("TEST.NS", df, if_exists="replace")
    return repo


@pytest.fixture
def fitted_model(repo_with_returns):
    """A GarchModel that has been wrangled and fitted."""
    gm = GarchModel(ticker="TEST.NS", repo=repo_with_returns)
    gm.wrangle_data(n_observations=400)
    gm.fit(p=1, q=1)
    return gm


# wrangle_data

class TestWrangleData:
    def test_sets_data_attribute(self, repo_with_returns):
        gm = GarchModel(ticker="TEST.NS", repo=repo_with_returns)
        gm.wrangle_data(n_observations=300)
        assert gm.data is not None

    def test_data_is_series(self, repo_with_returns):
        gm = GarchModel(ticker="TEST.NS", repo=repo_with_returns)
        gm.wrangle_data(n_observations=300)
        assert isinstance(gm.data, pd.Series)

    def test_no_nan_in_data(self, repo_with_returns):
        gm = GarchModel(ticker="TEST.NS", repo=repo_with_returns)
        gm.wrangle_data(n_observations=300)
        assert gm.data.isna().sum() == 0

    def test_raises_without_repo(self):
        gm = GarchModel(ticker="TEST.NS")
        with pytest.raises(ValueError, match="No repository attached"):
            gm.wrangle_data()


# fit

class TestFit:
    def test_sets_model_fit(self, fitted_model):
        assert fitted_model.model_fit is not None

    def test_sets_aic(self, fitted_model):
        assert fitted_model.aic is not None
        assert isinstance(fitted_model.aic, float)

    def test_sets_bic(self, fitted_model):
        assert fitted_model.bic is not None
        assert isinstance(fitted_model.bic, float)

    def test_raises_without_data(self, repo_with_returns):
        gm = GarchModel(ticker="TEST.NS", repo=repo_with_returns)
        with pytest.raises(ValueError, match="No data loaded"):
            gm.fit()


# predict_volatility

class TestPredictVolatility:
    def test_returns_series(self, fitted_model):
        vol = fitted_model.predict_volatility(horizon=5)
        assert isinstance(vol, pd.Series)

    def test_correct_length(self, fitted_model):
        horizon = 5
        vol = fitted_model.predict_volatility(horizon=horizon)
        assert len(vol) == horizon

    def test_values_are_positive(self, fitted_model):
        vol = fitted_model.predict_volatility(horizon=5)
        assert (vol > 0).all(), "All volatility forecasts should be positive"

    def test_raises_without_fit(self, repo_with_returns):
        gm = GarchModel(ticker="TEST.NS", repo=repo_with_returns)
        gm.wrangle_data()
        with pytest.raises(ValueError, match="No fitted model found"):
            gm.predict_volatility()


# dump / load

class TestDumpLoad:
    def test_dump_creates_file(self, fitted_model, tmp_path):
        filepath = str(tmp_path / "models" / "test_model.pkl")
        saved_path = fitted_model.dump(filepath)
        assert os.path.isfile(saved_path)

    def test_load_restores_model(self, fitted_model, tmp_path):
        filepath = str(tmp_path / "models" / "test_model.pkl")
        fitted_model.dump(filepath)

        gm2 = GarchModel(ticker="TEST.NS")
        gm2.load(filepath)
        assert gm2.model_fit is not None

    def test_loaded_model_can_predict(self, fitted_model, tmp_path):
        filepath = str(tmp_path / "models" / "test_model.pkl")
        fitted_model.dump(filepath)

        gm2 = GarchModel(ticker="TEST.NS")
        gm2.load(filepath)
        vol = gm2.predict_volatility(horizon=3)
        assert len(vol) == 3


# build_model_path

class TestBuildModelPath:
    def test_returns_string(self):
        path = GarchModel.build_model_path("RELIANCE.NS")
        assert isinstance(path, str)

    def test_ends_with_pkl(self):
        path = GarchModel.build_model_path("RELIANCE.NS")
        assert path.endswith(".pkl")

    def test_contains_ticker(self):
        path = GarchModel.build_model_path("TCS.NS", models_dir="models")
        assert "TCS" in path
