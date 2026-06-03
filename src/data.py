"""
data.py — Extract, Transform, Load (ETL) for BSE stock data.

Classes
-------
YFinanceAPI     : Fetch daily OHLCV data from Yahoo Finance.
SQLRepository   : Read/write stock data to a SQLite database.

Standalone helper
-----------------
get_stock_data  : One-call convenience wrapper used in notebooks.
"""

import os
import sqlite3

import numpy as np
import pandas as pd
import yfinance as yf
from sqlalchemy import create_engine, text


# ---------------------------------------------------------------------------
# Yahoo Finance API wrapper
# ---------------------------------------------------------------------------

class YFinanceAPI:
    """Fetch and clean daily stock data from Yahoo Finance."""

    def get_daily_data(
        self,
        ticker: str,
        start: str,
        end: str,
    ) -> pd.DataFrame:
        """
        Download OHLCV data and compute daily percentage returns.

        Parameters
        ----------
        ticker : str
            Yahoo Finance symbol, e.g. ``"RELIANCE.NS"``, ``"^BSESN"``.
        start : str
            Start date in ``YYYY-MM-DD`` format.
        end : str
            End date in ``YYYY-MM-DD`` format.

        Returns
        -------
        pd.DataFrame
            Columns: Open, High, Low, Close, Volume, returns.
            Index: DatetimeIndex named ``Date``.

        Raises
        ------
        ValueError
            If Yahoo Finance returns an empty response for the given ticker.
        """
        raw = yf.download(
            ticker,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
        )

        if raw.empty:
            raise ValueError(
                f"No data returned for ticker '{ticker}'. "
                "Check the symbol and date range."
            )

        # Flatten MultiIndex columns that yfinance sometimes returns
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.index.name = "Date"
        df.dropna(inplace=True)

        # Daily percentage returns (not fractional) — used by GARCH
        df["returns"] = df["Close"].pct_change() * 100
        df.dropna(inplace=True)

        return df


# ---------------------------------------------------------------------------
# SQLite repository
# ---------------------------------------------------------------------------

class SQLRepository:
    """
    Thin wrapper around a SQLite connection for reading/writing DataFrames.

    Parameters
    ----------
    connection : sqlite3.Connection
        An open ``sqlite3`` connection (or a SQLAlchemy engine/connection).
    """

    def __init__(self, connection):
        self.connection = connection

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def insert_table(
        self,
        table_name: str,
        records: pd.DataFrame,
        if_exists: str = "fail",
    ) -> dict:
        """
        Persist a DataFrame to a SQLite table.

        Parameters
        ----------
        table_name : str
            Target table name (typically the ticker symbol).
        records : pd.DataFrame
            Data to write. Must have a DatetimeIndex named ``Date``.
        if_exists : {"fail", "replace", "append"}
            Behaviour when the table already exists.

        Returns
        -------
        dict
            ``{"transaction_successful": bool, "records_inserted": int}``
        """
        n_inserted = records.to_sql(
            name=table_name,
            con=self.connection,
            if_exists=if_exists,
            index=True,
            index_label="Date",
        )
        return {
            "transaction_successful": True,
            "records_inserted": n_inserted or len(records),
        }

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def read_table(
        self,
        table_name: str,
        limit: int | None = None,
    ) -> pd.DataFrame:
        """
        Load a table (or its most-recent rows) into a DataFrame.

        Parameters
        ----------
        table_name : str
            Table to query.
        limit : int, optional
            If given, return only the most recent ``limit`` rows.

        Returns
        -------
        pd.DataFrame
            Data with a DatetimeIndex named ``Date``.
        """
        if limit:
            query = (
                f"SELECT * FROM '{table_name}' "
                f"ORDER BY Date DESC LIMIT {limit}"
            )
        else:
            query = f"SELECT * FROM '{table_name}' ORDER BY Date ASC"

        df = pd.read_sql(
            query,
            con=self.connection,
            index_col="Date",
            parse_dates=["Date"],
        )
        df.sort_index(inplace=True)
        return df

    def table_exists(self, table_name: str) -> bool:
        """Return True if *table_name* exists in the database."""
        query = (
            "SELECT name FROM sqlite_master "
            f"WHERE type='table' AND name='{table_name}'"
        )
        result = pd.read_sql(query, con=self.connection)
        return not result.empty


# ---------------------------------------------------------------------------
# Convenience helper used directly in notebooks
# ---------------------------------------------------------------------------

def get_stock_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    One-call wrapper: fetch, clean, and return BSE stock data.

    Parameters
    ----------
    ticker : str
        Yahoo Finance symbol.  For BSE stocks append ``.BO``; for NSE use
        ``.NS``.  Examples: ``"RELIANCE.NS"``, ``"TCS.BO"``, ``"^BSESN"``.
    start : str
        Start date ``YYYY-MM-DD``.
    end : str
        End date ``YYYY-MM-DD``.

    Returns
    -------
    pd.DataFrame
        Cleaned OHLCV + returns DataFrame.

    Examples
    --------
    >>> df = get_stock_data("RELIANCE.NS", "2018-01-01", "2023-12-31")
    >>> df.head()
    """
    api = YFinanceAPI()
    return api.get_daily_data(ticker=ticker, start=start, end=end)
