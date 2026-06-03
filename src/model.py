"""
model.py — GARCH volatility model: fit, forecast, save, load.

Class
-----
GarchModel : Wraps the ``arch`` library's GARCH(p, q) implementation with
             a clean interface for notebooks and the FastAPI server.
"""

import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from arch import arch_model


class GarchModel:
    """
    GARCH(p, q) model for conditional volatility forecasting on BSE stocks.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol used to look up data in the repository
        (e.g. ``"RELIANCE.NS"``).
    repo : SQLRepository, optional
        Data repository instance.  Required when calling :meth:`wrangle_data`.
    use_new_data : bool, default True
        If True, prefer fresh data from Yahoo Finance when available.

    Attributes
    ----------
    data : pd.Series
        Percentage daily returns used to train the model.
    model_fit : ARCHModelResult | None
        Fitted model result object (set after calling :meth:`fit`).
    aic : float | None
        Akaike Information Criterion of the fitted model.
    bic : float | None
        Bayesian Information Criterion of the fitted model.

    Examples
    --------
    Fit and forecast:

    >>> gm = GarchModel(ticker="RELIANCE.NS", repo=repo)
    >>> gm.wrangle_data(n_observations=1000)
    >>> gm.fit(p=1, q=1)
    >>> vol = gm.predict_volatility(horizon=5)

    Save and reload:

    >>> path = gm.dump("models/RELIANCE.NS_2024-01-01.pkl")
    >>> gm2 = GarchModel(ticker="RELIANCE.NS")
    >>> gm2.load(path)
    >>> vol = gm2.predict_volatility(horizon=5)
    """

    def __init__(
        self,
        ticker: str,
        repo=None,
        use_new_data: bool = True,
    ):
        self.ticker = ticker
        self.repo = repo
        self.use_new_data = use_new_data

        self.data: pd.Series | None = None
        self.model = None
        self.model_fit = None
        self.aic: float | None = None
        self.bic: float | None = None

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    def wrangle_data(self, n_observations: int = 2500) -> "GarchModel":
        """
        Load returns from the repository and attach them to ``self.data``.

        Parameters
        ----------
        n_observations : int, default 2500
            Maximum number of rows to load (most-recent rows).

        Returns
        -------
        GarchModel
            Self, for method chaining.

        Raises
        ------
        ValueError
            If no repository has been set.
        """
        if self.repo is None:
            raise ValueError(
                "No repository attached.  Pass a SQLRepository instance "
                "when constructing GarchModel."
            )

        df = self.repo.read_table(
            table_name=self.ticker,
            limit=n_observations,
        )
        df.sort_index(inplace=True)

        # Recompute returns in case the stored column is stale
        df["returns"] = df["Close"].pct_change() * 100
        df.dropna(inplace=True)

        self.data = df["returns"]
        return self

    # ------------------------------------------------------------------
    # Model fitting
    # ------------------------------------------------------------------

    def fit(self, p: int = 1, q: int = 1) -> "GarchModel":
        """
        Fit a GARCH(p, q) model to ``self.data``.

        Parameters
        ----------
        p : int, default 1
            Order of the ARCH (lagged squared residual) terms.
        q : int, default 1
            Order of the GARCH (lagged variance) terms.

        Returns
        -------
        GarchModel
            Self, for method chaining.

        Raises
        ------
        ValueError
            If :meth:`wrangle_data` has not been called first.
        """
        if self.data is None:
            raise ValueError(
                "No data loaded.  Call wrangle_data() before fit()."
            )

        self.model = arch_model(
            self.data,
            p=p,
            q=q,
            rescale=False,
        )
        self.model_fit = self.model.fit(disp="off")
        self.aic = self.model_fit.aic
        self.bic = self.model_fit.bic
        return self

    # ------------------------------------------------------------------
    # Forecasting
    # ------------------------------------------------------------------

    def predict_volatility(self, horizon: int = 5) -> pd.Series:
        """
        Produce an annualised volatility forecast.

        Parameters
        ----------
        horizon : int, default 5
            Number of trading days ahead to forecast.

        Returns
        -------
        pd.Series
            Annualised volatility forecast for each period in the horizon.
            Index labels are ``h.1``, ``h.2``, …, ``h.<horizon>``.

        Raises
        ------
        ValueError
            If :meth:`fit` (or :meth:`load`) has not been called first.
        """
        if self.model_fit is None:
            raise ValueError(
                "No fitted model found.  Call fit() or load() first."
            )

        forecast = self.model_fit.forecast(horizon=horizon, reindex=False)
        variance_forecast = forecast.variance.iloc[-1]

        # Annualise: daily variance → annual volatility (√252 scaling)
        volatility_forecast = np.sqrt(variance_forecast) * np.sqrt(252)
        return volatility_forecast

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def dump(self, filepath: str) -> str:
        """
        Serialise the fitted model to ``filepath`` using joblib.

        The parent directory is created automatically if it does not exist.

        Parameters
        ----------
        filepath : str
            Destination path, e.g. ``"models/RELIANCE.NS_2024-01-15.pkl"``.

        Returns
        -------
        str
            The resolved absolute path of the saved file.
        """
        if self.model_fit is None:
            raise ValueError("Nothing to save.  Call fit() first.")

        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        joblib.dump(self.model_fit, filepath)
        return os.path.abspath(filepath)

    def load(self, filepath: str) -> "GarchModel":
        """
        Deserialise a previously saved model from ``filepath``.

        Parameters
        ----------
        filepath : str
            Path to a joblib-serialised ``ARCHModelResult`` object.

        Returns
        -------
        GarchModel
            Self, for method chaining.
        """
        self.model_fit = joblib.load(filepath)
        return self

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def build_model_path(ticker: str, models_dir: str = "models") -> str:
        """
        Return a canonical path for a model checkpoint.

        Format: ``<models_dir>/<ticker>_<YYYY-MM-DD>.pkl``
        """
        datestamp = datetime.today().strftime("%Y-%m-%d")
        filename = f"{ticker.replace('/', '-')}_{datestamp}.pkl"
        return os.path.join(models_dir, filename)
