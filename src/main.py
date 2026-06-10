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

import os
import sqlite3
from datetime import datetime
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field, field_validator

# Resolve paths relative to this file so the server works from any cwd
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_DB_PATH = os.path.join(_ROOT, "database", "stock_data.db")
_MODELS_DIR = os.path.join(_ROOT, "models")
_FIT_API_KEY = os.getenv("FIT_API_KEY")
_DEFAULT_LOOKBACK_DAYS = 365 * 5

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


def _require_fit_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Protect the expensive training endpoint when FIT_API_KEY is configured."""
    if _FIT_API_KEY and x_api_key != _FIT_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


def _fit_garch(
    ticker: str,
    start_date: str,
    end_date: str,
    n_observations: int,
    p: int,
    q: int,
    persist: bool,
) -> tuple[Any, str | None]:
    """Fetch data, train a GARCH model, and optionally persist data/model files."""
    api = _get_api()
    df = api.get_daily_data(ticker=ticker, start=start_date, end=end_date)

    repo = None
    if persist:
        repo = _get_repo()
        repo.insert_table(table_name=ticker, records=df, if_exists="replace")
    else:
        repo = _get_in_memory_repo(ticker=ticker, records=df)

    model = _get_model(ticker=ticker, repo=repo)
    model.wrangle_data(n_observations=n_observations)
    model.fit(p=p, q=q)

    model_path = None
    if persist:
        from model import GarchModel  # noqa: PLC0415
        model_path = GarchModel.build_model_path(
            ticker=ticker,
            models_dir=_MODELS_DIR,
        )
        model.dump(model_path)

    return model, model_path


def _get_in_memory_repo(ticker: str, records: Any):
    """Build a temporary repository for stateless model training."""
    from data import SQLRepository  # noqa: PLC0415

    conn = sqlite3.connect(":memory:")
    repo = SQLRepository(connection=conn)
    repo.insert_table(table_name=ticker, records=records, if_exists="replace")
    return repo


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
    ticker: str = Field(..., min_length=1, max_length=20)
    start_date: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$")
    end_date: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$")
    n_observations: int = Field(default=2500, ge=30, le=5000)
    p: int = Field(default=1, ge=1, le=5)
    q: int = Field(default=1, ge=1, le=5)
    persist: bool = False

    @field_validator("ticker")
    @classmethod
    def validate_ticker(cls, value: str) -> str:
        cleaned = value.strip().upper()
        allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-_^")
        if any(char not in allowed for char in cleaned):
            raise ValueError("Ticker contains unsupported characters.")
        return cleaned

    @field_validator("end_date")
    @classmethod
    def validate_date_order(cls, value: str, info):
        start_date = info.data.get("start_date")
        if start_date and value <= start_date:
            raise ValueError("end_date must be after start_date.")
        return value


class FitOut(FitIn):
    success: bool
    message: str
    model_path: str | None
    aic: float
    bic: float


@app.post("/fit", response_model=FitOut, summary="Fit a GARCH model")
def fit_model(
    request: FitIn,
    _: None = Depends(_require_fit_api_key),
) -> FitOut:
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
        model, model_path = _fit_garch(
            ticker=request.ticker,
            start_date=request.start_date,
            end_date=request.end_date,
            n_observations=request.n_observations,
            p=request.p,
            q=request.q,
            persist=request.persist,
        )

        return FitOut(
            **request.model_dump(),
            success=True,
            message=(
                f"Model fitted for {request.ticker}."
                if not model_path
                else f"Model fitted and saved for {request.ticker}."
            ),
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
    ticker: str = Field(..., min_length=1, max_length=20)
    n_days: int = Field(default=5, ge=1, le=30)
    start_date: str | None = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    end_date: str | None = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    n_observations: int = Field(default=1250, ge=30, le=5000)
    p: int = Field(default=1, ge=1, le=5)
    q: int = Field(default=1, ge=1, le=5)

    @field_validator("ticker")
    @classmethod
    def validate_ticker(cls, value: str) -> str:
        cleaned = value.strip().upper()
        allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-_^")
        if any(char not in allowed for char in cleaned):
            raise ValueError("Ticker contains unsupported characters.")
        return cleaned

    @field_validator("end_date")
    @classmethod
    def validate_date_order(cls, value: str | None, info):
        start_date = info.data.get("start_date")
        if value and start_date and value <= start_date:
            raise ValueError("end_date must be after start_date.")
        return value


class PredictOut(PredictIn):
    success: bool
    model_path: str | None
    model_source: str
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
        latest_model_path = _latest_model_path(request.ticker)

        if latest_model_path:
            model = _get_model(ticker=request.ticker)
            model.load(latest_model_path)
            model_source = "saved_model"
        else:
            end_date = request.end_date or datetime.utcnow().date().isoformat()
            start_date = request.start_date or _default_start_date(end_date)
            model, latest_model_path = _fit_garch(
                ticker=request.ticker,
                start_date=start_date,
                end_date=end_date,
                n_observations=request.n_observations,
                p=request.p,
                q=request.q,
                persist=False,
            )
            model_source = "trained_on_demand"

        vol_series = model.predict_volatility(horizon=request.n_days)

        forecast = {
            label: round(float(value), 6)
            for label, value in vol_series.items()
        }

        return PredictOut(
            **request.model_dump(),
            success=True,
            model_path=latest_model_path,
            model_source=model_source,
            forecast=forecast,
        )

    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _latest_model_path(ticker: str) -> str | None:
    import glob

    pattern = os.path.join(_MODELS_DIR, f"{ticker.replace('/', '-')}_*.pkl")
    candidates = sorted(glob.glob(pattern))
    return candidates[-1] if candidates else None


def _default_start_date(end_date: str) -> str:
    from datetime import date, timedelta

    end = date.fromisoformat(end_date)
    return (end - timedelta(days=_DEFAULT_LOOKBACK_DAYS)).isoformat()
