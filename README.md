# Machine Learning-Based Stock Market Volatility Forecasting
### Bombay Stock Exchange (BSE) — GARCH Model + FastAPI Deployment

A end-to-end project covering: data collection → volatility calculation → GARCH modelling → API deployment.  
Data source: **Yahoo Finance** (`yfinance`). Exchange focus: **BSE / NSE (India)**.

---

## Project Structure

```
stock-market-volatility-forecasting/
│
├── notebooks/
│   ├── 01_getting_data.ipynb        # Fetch & clean BSE data from Yahoo Finance
│   ├── 02_data_model.ipynb          # TDD + SQLite ETL (builds data.py)
│   ├── 03_garch.ipynb               # Volatility calc + GARCH model (builds model.py)
│   └── 04_model_deployment.ipynb    # FastAPI server + /fit + /predict endpoints
│
├── src/
│   ├── data.py                      # YFinanceAPI, SQLRepository, get_stock_data()
│   ├── model.py                     # GarchModel: fit, predict, dump, load
│   └── main.py                      # FastAPI app — /hello, /fit, /predict
│
├── database/                        # SQLite database files (auto-created)
├── models/                          # Saved GARCH model checkpoints (.pkl)
├── tests/
│   ├── test_data.py                 # Unit tests for data.py
│   └── test_model.py                # Unit tests for model.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Workflow

```
Google Colab         Replit               GitHub
─────────────        ──────────────────   ──────────────
Work on .ipynb  ───► Push to GitHub  ───► Stores everything
notebooks                │
                         ▼
                    Pull latest changes
                    Work on .py files
                    Run FastAPI server
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the FastAPI server

```bash
cd src
uvicorn main:app --reload --workers 1 --host localhost --port 8008
```

Interactive docs: http://localhost:8008/docs

### 3. Run tests

```bash
pytest tests/ -v
```

---

## API Endpoints

| Method | Path       | Description                                  |
|--------|-----------|----------------------------------------------|
| GET    | `/hello`   | Health check                                 |
| POST   | `/fit`     | Fetch data, fit GARCH model, save to disk    |
| POST   | `/predict` | Load saved model, return volatility forecast |

### Example: Fit a model

```bash
curl -X POST http://localhost:8008/fit \
  -H "Content-Type: application/json" \
  -d '{
    "ticker": "RELIANCE.NS",
    "start_date": "2018-01-01",
    "end_date": "2024-12-31",
    "n_observations": 1500,
    "p": 1,
    "q": 1
  }'
```

### Example: Predict volatility

```bash
curl -X POST http://localhost:8008/predict \
  -H "Content-Type: application/json" \
  -d '{
    "ticker": "RELIANCE.NS",
    "n_days": 5
  }'
```

---

## Key BSE/NSE Tickers (Yahoo Finance)

| Symbol        | Name                    |
|---------------|-------------------------|
| `^BSESN`      | BSE SENSEX              |
| `^NSEI`       | NSE NIFTY 50            |
| `RELIANCE.NS` | Reliance Industries     |
| `TCS.NS`      | Tata Consultancy Svcs   |
| `INFY.NS`     | Infosys                 |
| `HDFCBANK.NS` | HDFC Bank               |
| `WIPRO.NS`    | Wipro                   |
| `ITC.NS`      | ITC Ltd                 |
| `SBIN.NS`     | State Bank of India     |
| `TATAMOTORS.NS` | Tata Motors           |

Append `.BO` instead of `.NS` to get BSE-listed prices where both exist.

---

## Notebooks

| Notebook | Purpose | Key Output |
|----------|---------|------------|
| `01_getting_data.ipynb` | Download, clean, explore BSE data | Clean DataFrame with returns |
| `02_data_model.ipynb` | TDD for data layer, SQLite storage | `data.py` (ETL classes) |
| `03_garch.ipynb` | Volatility computation + GARCH fit | `model.py`, saved `.pkl` |
| `04_model_deployment.ipynb` | FastAPI walkthrough, test endpoints | `main.py`, live API |

---

## Stack

- **Data**: `yfinance`, `pandas`, `numpy`
- **Database**: `SQLite` + `SQLAlchemy`
- **Model**: `arch` (GARCH), `joblib` (serialisation)
- **API**: `FastAPI`, `uvicorn`, `pydantic`
- **Viz**: `matplotlib`, `plotly`
- **Tests**: `pytest`
