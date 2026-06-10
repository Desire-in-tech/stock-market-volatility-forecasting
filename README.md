# Machine Learning-Based Stock Market Volatility Forecasting

FastAPI service for forecasting stock-market volatility with a GARCH model using Yahoo Finance data.

The API can run without a pre-saved model or committed database. If no saved model exists, `/predict` fetches recent market data, trains a temporary in-memory GARCH model, and returns a forecast.

## Project Structure

```text
stock-market-volatility-forecasting/
├── src/
│   ├── data.py
│   ├── model.py
│   └── main.py
├── database/
├── models/
├── notebooks/
├── tests/
├── .python-version
├── render.yaml
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

## Local Setup

Create and activate a virtual environment, then install dependencies:

```bash
pip install -r requirements-dev.txt
```

Run the API locally from the project root:

```bash
uvicorn main:app --app-dir src --reload --host 127.0.0.1 --port 8008
```

Open the API docs:

```text
http://127.0.0.1:8008/docs
```

Run tests:

```bash
pytest tests -v
```

## Render Deployment

This repository includes `render.yaml` for Render Blueprint deployments.

Render settings used by this project:

```yaml
buildCommand: pip install -r requirements.txt
startCommand: uvicorn main:app --app-dir src --host 0.0.0.0 --port $PORT
```

Python is pinned with:

```text
.python-version
3.11.11
```

The `FIT_API_KEY` environment variable is declared in `render.yaml` with `sync: false`. Set its value in Render if you want to protect `/fit`. When `FIT_API_KEY` is set, requests to `/fit` must include:

```text
X-API-Key: your-secret-key
```

## API Endpoints

| Method | Path       | Description |
|--------|------------|-------------|
| GET    | `/hello`   | Health check |
| POST   | `/fit`     | Train a GARCH model. Saves files only when `persist` is `true`. |
| POST   | `/predict` | Forecast volatility. Uses saved model if present, otherwise trains in memory. |

### Health Check

```bash
curl https://your-render-service.onrender.com/hello
```

### Predict Without a Saved Model

```bash
curl -X POST https://your-render-service.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "ticker": "RELIANCE.NS",
    "n_days": 5
  }'
```

If no saved model exists, the response includes:

```json
{
  "model_source": "trained_on_demand",
  "model_path": null
}
```

### Fit a Model

By default, `/fit` trains without saving model/database files:

```bash
curl -X POST https://your-render-service.onrender.com/fit \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-secret-key" \
  -d '{
    "ticker": "RELIANCE.NS",
    "start_date": "2018-01-01",
    "end_date": "2024-12-31",
    "n_observations": 1500,
    "p": 1,
    "q": 1,
    "persist": false
  }'
```

Set `"persist": true` only if your Render service has persistent storage configured. Otherwise saved SQLite databases and `.pkl` model files can disappear after a restart or redeploy.

## Persistence Notes

The app is deployment-ready without committed database or model files:

- `database/*.db` is ignored by Git.
- `models/*.pkl` is ignored by Git.
- `/predict` can train on demand when no saved model exists.

For production, prefer Render Postgres or another managed database instead of SQLite, and store trained model artifacts in persistent object storage. A Render persistent disk is acceptable for a demo service.

## Dependency Files

`requirements.txt` is for Render/runtime dependencies only.

`requirements-dev.txt` includes notebooks, plotting, and testing tools for local development.

## `__pycache__`

`__pycache__` folders are automatically created by Python when files are imported or compiled. They contain cached bytecode, not source code. They are already ignored by `.gitignore`, so they should not appear in GitHub.

You can delete `src/__pycache__/` locally if you want a cleaner file tree. Python will recreate it the next time the app or tests run.

## Common Yahoo Finance Symbols

| Symbol          | Name |
|-----------------|------|
| `^BSESN`        | BSE SENSEX |
| `^NSEI`         | NSE NIFTY 50 |
| `RELIANCE.NS`   | Reliance Industries |
| `TCS.NS`        | Tata Consultancy Services |
| `INFY.NS`       | Infosys |
| `HDFCBANK.NS`   | HDFC Bank |
| `WIPRO.NS`      | Wipro |
| `ITC.NS`        | ITC Ltd |
| `SBIN.NS`       | State Bank of India |
| `TATAMOTORS.NS` | Tata Motors |
