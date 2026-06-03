"""
main.py — FastAPI server for BSE volatility predictions.

Run locally:
    uvicorn main:app --reload --workers 1 --host localhost --port 8008

Endpoints
---------
GET  /hello           Health check / greeting.
POST /fit             Fetch data → fit GARCH → save model.
POST /predict         Load latest model → forecast volatility.
"""

import glob
import os
import sqlite3
from datetime import datetime
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Resolve paths relative to this file so the server works from any cwd
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_DB_PATH = os.path.join(_ROOT, "database", "stock_data.db")
_MODELS_DIR = os.path.join(_ROOT, "models")

# ---------------------------------------------------------------------------
# Lazy imports — heavy ML deps only loaded when a request comes in
# ---------------------------------------------------------------------------

def _get_repo():
    from data import SQLRepository  # noqa: PLC0415
    conn = sqlite3.connect(_DB_PATH)
    return SQLRepository(connection=conn)


def _get_api():
    from data import YFinanceAPI  # noqa: PLC0415
    return YFinanceAPI()


def _get_model(ticker: str, repo=None):
    from model import GarchModel  # noqa: PLC0415
    return GarchModel(ticker=ticker, repo=repo)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="BSE Volatility Forecasting API",
    description=(
        "Predict Bombay Stock Exchange stock volatility using a GARCH model "
        "trained on Yahoo Finance data."
    ),
    version="1.0.0",
)

os.makedirs(_MODELS_DIR, exist_ok=True)
os.makedirs(os.path.dirname(_DB_PATH), exist_ok=True)


# ---------------------------------------------------------------------------
# /hello
# ---------------------------------------------------------------------------

@app.get("/hello", summary="Health check")
def hello() -> dict[str, str]:
    """Return a greeting to confirm the server is running."""
    return {
        "message": "Hello! BSE Volatility Forecasting API is live.",
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


# ---------------------------------------------------------------------------
# /fit
# ---------------------------------------------------------------------------

class FitIn(BaseModel):
    ticker: str
    start_date: str
    end_date: str
    n_observations: int = 2500
    p: int = 1
    q: int = 1


class FitOut(FitIn):
    success: bool
    message: str
    model_path: str
    aic: float
    bic: float


@app.post("/fit", response_model=FitOut, summary="Fit a GARCH model")
def fit_model(request: FitIn) -> FitOut:
    """
    Fetch data for *ticker* from Yahoo Finance, store it in SQLite,
    fit a GARCH(p, q) model, and persist the result to disk.

    Example body::

        {
            "ticker": "RELIANCE.NS",
            "start_date": "2018-01-01",
            "end_date": "2024-12-31",
            "n_observations": 1500,
            "p": 1,
            "q": 1
        }
    """
    try:
        # 1. Fetch raw data from Yahoo Finance
        api = _get_api()
        df = api.get_daily_data(
            ticker=request.ticker,
            start=request.start_date,
            end=request.end_date,
        )

        # 2. Persist to SQLite (replace if already stored)
        repo = _get_repo()
        repo.insert_table(
            table_name=request.ticker,
            records=df,
            if_exists="replace",
        )

        # 3. Build and fit the GARCH model
        model = _get_model(ticker=request.ticker, repo=repo)
        model.wrangle_data(n_observations=request.n_observations)
        model.fit(p=request.p, q=request.q)

        # 4. Save model to disk with a datestamped filename
        from model import GarchModel  # noqa: PLC0415
        model_path = GarchModel.build_model_path(
            ticker=request.ticker,
            models_dir=_MODELS_DIR,
        )
        model.dump(model_path)

        return FitOut(
            **request.model_dump(),
            success=True,
            message=f"Model fitted and saved for {request.ticker}.",
            model_path=model_path,
            aic=round(model.aic, 4),
            bic=round(model.bic, 4),
        )

    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# /predict
# ---------------------------------------------------------------------------

class PredictIn(BaseModel):
    ticker: str
    n_days: int = 5


class PredictOut(PredictIn):
    success: bool
    model_path: str
    forecast: dict[str, float]


@app.post("/predict", response_model=PredictOut, summary="Forecast volatility")
def predict_volatility(request: PredictIn) -> PredictOut:
    """
    Load the most recently saved GARCH model for *ticker* and return an
    annualised volatility forecast for the next *n_days* trading days.

    Example body::

        {
            "ticker": "RELIANCE.NS",
            "n_days": 5
        }

    The response ``forecast`` dict maps horizon labels (``"h.1"`` … ``"h.N"``)
    to annualised volatility values (in %).
    """
    try:
        # Find the most recent model file for this ticker
        pattern = os.path.join(
            _MODELS_DIR,
            f"{request.ticker.replace('/', '-')}_*.pkl",
        )
        candidates = sorted(glob.glob(pattern))

        if not candidates:
            raise FileNotFoundError(
                f"No saved model found for ticker '{request.ticker}'. "
                "Call /fit first."
            )

        latest_model_path = candidates[-1]

        # Load and forecast
        model = _get_model(ticker=request.ticker)
        model.load(latest_model_path)
        vol_series = model.predict_volatility(horizon=request.n_days)

        forecast = {
            label: round(float(value), 6)
            for label, value in vol_series.items()
        }

        return PredictOut(
            **request.model_dump(),
            success=True,
            model_path=latest_model_path,
            forecast=forecast,
        )

    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
