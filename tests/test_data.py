"""
tests/test_data.py — Unit tests for data.py (ETL layer).

Run with:
    pytest tests/test_data.py -v
"""

import sqlite3
import sys
import os

import pandas as pd
import pytest

# Make src importable when running pytest from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data import YFinanceAPI, SQLRepository, get_stock_data


# YFinanceAPI

class TestYFinanceAPI:
    """Tests for YFinanceAPI.get_daily_data()."""

    def test_returns_dataframe(self):
        api = YFinanceAPI()
        df = api.get_daily_data("^BSESN", start="2023-01-01", end="2023-03-31")
        assert isinstance(df, pd.DataFrame), "Expected a DataFrame"

    def test_expected_columns(self):
        api = YFinanceAPI()
        df = api.get_daily_data("^BSESN", start="2023-01-01", end="2023-03-31")
        for col in ["Open", "High", "Low", "Close", "Volume", "returns"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_no_null_returns(self):
        api = YFinanceAPI()
        df = api.get_daily_data("^BSESN", start="2023-01-01", end="2023-03-31")
        assert df["returns"].isna().sum() == 0, "Found NaN values in returns"

    def test_index_is_datetime(self):
        api = YFinanceAPI()
        df = api.get_daily_data("^BSESN", start="2023-01-01", end="2023-03-31")
        assert isinstance(df.index, pd.DatetimeIndex), "Index should be DatetimeIndex"

    def test_invalid_ticker_raises(self):
        api = YFinanceAPI()
        with pytest.raises(ValueError, match="No data returned"):
            api.get_daily_data(
                "INVALID_TICKER_XYZ_999",
                start="2023-01-01",
                end="2023-03-31",
            )


# SQLRepository

class TestSQLRepository:
    """Tests for SQLRepository using an in-memory SQLite database."""

    @pytest.fixture
    def repo_with_data(self):
        """Return a fresh repo pre-loaded with two rows of data."""
        conn = sqlite3.connect(":memory:")
        repo = SQLRepository(connection=conn)
        # Build a small dummy DataFrame
        df = pd.DataFrame(
            {
                "Open": [100.0, 101.0],
                "High": [102.0, 103.0],
                "Low": [99.0, 100.0],
                "Close": [101.0, 102.0],
                "Volume": [1_000_000, 1_100_000],
                "returns": [1.0, 0.99],
            },
            index=pd.to_datetime(["2023-01-02", "2023-01-03"]),
        )
        df.index.name = "Date"
        repo.insert_table("TEST_TICKER", df, if_exists="replace")
        return repo

    def test_insert_returns_success(self):
        conn = sqlite3.connect(":memory:")
        repo = SQLRepository(connection=conn)
        df = pd.DataFrame(
            {"Close": [100.0], "returns": [0.5]},
            index=pd.to_datetime(["2023-01-02"]),
        )
        df.index.name = "Date"
        result = repo.insert_table("T", df, if_exists="replace")
        assert result["transaction_successful"] is True

    def test_read_returns_dataframe(self, repo_with_data):
        df = repo_with_data.read_table("TEST_TICKER")
        assert isinstance(df, pd.DataFrame)

    def test_read_correct_shape(self, repo_with_data):
        df = repo_with_data.read_table("TEST_TICKER")
        assert len(df) == 2, f"Expected 2 rows, got {len(df)}"

    def test_read_limit(self, repo_with_data):
        df = repo_with_data.read_table("TEST_TICKER", limit=1)
        assert len(df) == 1

    def test_table_exists(self, repo_with_data):
        assert repo_with_data.table_exists("TEST_TICKER") is True
        assert repo_with_data.table_exists("NONEXISTENT") is False


# get_stock_data convenience function

class TestGetStockData:
    def test_returns_dataframe(self):
        df = get_stock_data("^BSESN", start="2023-06-01", end="2023-06-30")
        assert isinstance(df, pd.DataFrame)

    def test_has_returns_column(self):
        df = get_stock_data("^BSESN", start="2023-06-01", end="2023-06-30")
        assert "returns" in df.columns
