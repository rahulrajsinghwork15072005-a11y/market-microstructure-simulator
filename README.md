# Market Microstructure Simulator

An agent-based **limit-order-book** simulator. Zero-intelligence + Hawkes order
flow (limit orders, market orders, cancellations) runs through a strict
price-time-priority matching engine, producing emergent **stylized facts**:
fat-tailed returns, volatility clustering, a fluctuating bid-ask spread, and
measurable **price impact**.

See [PLAN.md](PLAN.md) for the full design and math, and
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the system architecture (the LOB,
the order-flow generators, the discrete-event kernel, and the Hawkes calibration).

## What's implemented

| Layer | Status |
|-------|--------|
| Core (`app/core`) | price-time-priority LOB, Poisson + Hawkes flow (with **maximum-likelihood calibration** of the branching ratio), Kyle/square-root impact, microstructure metrics, discrete-event simulator, scenarios |
| API (`app/main.py`) | FastAPI REST + OpenAPI docs |
| Dashboard (`app/static`) | Plotly dashboard (mid tape, spread, book depth, return histogram, |return| ACF, impact) |
| Tests (`tests/`) | 62 unit + API tests (matching, reproducibility, stylized facts, multi-agent, Hawkes MLE parameter recovery) |

The Python price-time-priority engine here is the reference (and only)
implementation. Optionally binding the workspace's `limit-order-book` C++ engine
via pybind11 is a **roadmap** item (see [PLAN.md](PLAN.md) and
[docs/STYLIZED_FACTS.md](docs/STYLIZED_FACTS.md)) — it is not yet implemented.

## Run it

```powershell
# from market-microstructure-simulator/backend/
& ..\.venv\Scripts\python.exe -m pip install -r requirements.txt
& ..\.venv\Scripts\python.exe -m pytest -q
& ..\.venv\Scripts\python.exe scripts\bench.py        # engine + Hawkes-calibration throughput
& ..\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8003
```

- Dashboard: http://127.0.0.1:8003/
- Swagger docs: http://127.0.0.1:8003/docs

## Key endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/api/scenarios` | preset scenarios (balanced, thin_liquidity, toxic_flow) |
| POST | `/api/simulate` | run a custom `SimConfig` |
| GET | `/api/demo/run` | mid/spread/depth series |
| GET | `/api/demo/stylized-facts` | kurtosis, return ACF, |return| ACF |
| GET | `/api/demo/impact` | Kyle lambda + square-root impact curve |
| GET | `/api/demo/book` | final book depth snapshot |
| GET | `/api/demo/multi-agent` | multi-agent sim (market-maker / informed / noise) |
| GET | `/api/demo/hawkes-calibration` | simulate a Hawkes flow + MLE-recover (μ, α, β) |
| POST | `/api/hawkes/fit` | MLE fit of a Hawkes process to supplied event times |

## Live data (optional)

Calibrate simulator parameters from a real ticker's recent volatility (offline
fallback returns synthetic defaults):

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/api/calibrate?ticker=AAPL` | suggested mid / volatility from real data |

## React frontend (Vite)

```powershell
cd frontend
npm install
npm run dev        # http://localhost:5183 (dev proxy /api -> :8003)
npm run build      # production build to dist/
```

## Docker

```powershell
docker compose up --build
# frontend: http://localhost:8083  |  API: http://localhost:8003
```

## Production & operations

- Observability: `GET /api/version`, `GET /api/ready`, `GET /metrics` (Prometheus text format).
- Middleware: CORS, GZip, request-id + security headers, structured request logs, optional rate limiting.
- Env config: `CORS_ORIGINS`, `LOG_LEVEL`, `RATE_LIMIT_PER_SEC`.

## Market-making agent & metrics

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/api/agent/quote` | Avellaneda–Stoikov inventory-skewed bid/ask |

`/api/demo/stylized-facts` now also returns `hurst` and `realized_volatility`.

## Known limitations & roadmap

**Limitations**
- Research simulator, not a venue; single-asset; the matching core is Python
  (no C++ binding yet).
- Live calibration via yfinance is optional and falls back to synthetic defaults.

**Roadmap**
- pybind11 binding to the workspace `limit-order-book` C++ engine.
- L2 calibration of arrival/cancel rates and spread/depth. (Maximum-likelihood
  calibration of the **Hawkes branching ratio** is implemented — `fit_hawkes` and
  `GET /api/demo/hawkes-calibration`; plus a multi-agent simulation with exact
  per-agent PnL — `GET /api/demo/multi-agent`.)
  See [docs/STYLIZED_FACTS.md](docs/STYLIZED_FACTS.md).
