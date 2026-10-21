"""FastAPI application: microstructure simulator REST API + dashboard."""
from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from . import demo as demo_service
from .config import settings
from .core.agents import AvellanedaStoikovMaker
from .core.flow.hawkes import fit_hawkes
from .core.impact import kyle_lambda
from .core.metrics import stylized_facts
from .core.multiagent import MultiAgentConfig, run_multi_agent
from .core.simulator import MarketSimulator, SimConfig
from .data.calibrate import calibrate as calibrate_ticker
from .observability import (
    ObservabilityMiddleware,
    RateLimiter,
    RateLimitMiddleware,
    metrics,
    uptime_seconds,
)
from .schemas import HawkesFitRequest, SimConfigRequest

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

_STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(
    title="Market Microstructure Simulator",
    version=settings.version,
    description="Agent-based limit-order-book simulation with emergent stylized facts.",
)

app.add_middleware(GZipMiddleware, minimum_size=512)
if settings.rate_limit_per_sec > 0:
    app.add_middleware(RateLimitMiddleware, limiter=RateLimiter(settings.rate_limit_per_sec))
app.add_middleware(ObservabilityMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(status_code=500, content={"error": "internal_error", "detail": str(exc)})


if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
def dashboard() -> FileResponse:
    index = _STATIC_DIR / "index.html"
    if not index.exists():
        raise HTTPException(status_code=404, detail="dashboard not found")
    return FileResponse(str(index))


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/version")
def version() -> dict:
    return {
        "name": settings.app_name,
        "version": settings.version,
        "environment": settings.environment,
    }


@app.get("/api/ready")
def ready() -> dict:
    return {"ready": True, "uptime_seconds": round(uptime_seconds(), 1)}


@app.get("/metrics", include_in_schema=False)
def prometheus_metrics() -> PlainTextResponse:
    return PlainTextResponse(metrics.render(), media_type="text/plain; version=0.0.4")


@app.get("/api/scenarios")
def scenarios() -> dict:
    return {"scenarios": demo_service.list_scenarios()}


@app.post("/api/simulate")
def simulate(req: SimConfigRequest) -> dict:
    config = SimConfig(**req.model_dump())
    res = MarketSimulator(config).run()
    mids = np.asarray(res.mids, dtype=float)
    dp = np.diff(mids)
    v = np.asarray(res.signed_mkt_vol[1:], dtype=float)
    return {
        "n_events": len(res.mids),
        "n_trades": res.n_trades,
        "final_mid": res.mids[-1] if res.mids else None,
        "stylized_facts": stylized_facts(res.mids),
        "kyle_lambda": kyle_lambda(dp, v) if dp.size >= 2 else 0.0,
        "snapshot": res.snapshot,
    }


@app.get("/api/demo/run")
def demo_run(scenario: str = "balanced") -> dict:
    try:
        return demo_service.demo_run(scenario)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/demo/stylized-facts")
def demo_stylized_facts(scenario: str = "balanced") -> dict:
    try:
        return demo_service.demo_stylized_facts(scenario)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/demo/impact")
def demo_impact(scenario: str = "balanced") -> dict:
    try:
        return demo_service.demo_impact(scenario)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/demo/book")
def demo_book(scenario: str = "balanced") -> dict:
    try:
        return demo_service.demo_book(scenario)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/demo/multi-agent")
def demo_multi_agent(
    n_steps: int = 600,
    info_drift: float = 0.0,
    info_vol: float = 0.5,
    seed: int = 7,
) -> dict:
    """Run the multi-agent simulation (market-maker + informed + noise traders)."""
    n_steps = max(50, min(n_steps, 5000))
    result = run_multi_agent(
        MultiAgentConfig(
            seed=seed, n_steps=n_steps, info_drift=info_drift, info_vol=info_vol
        )
    )
    return {
        "n_steps": n_steps,
        "times": result.times,
        "mids": result.mids,
        "spreads": result.spreads,
        "fair_values": result.fair_values,
        "agents": result.agents,
        "n_trades": result.n_trades,
        "snapshot": result.snapshot,
    }


@app.get("/api/calibrate")
def calibrate(ticker: str = "AAPL", allow_network: bool = True) -> dict:
    return calibrate_ticker(ticker, allow_network=allow_network)


@app.get("/api/demo/hawkes-calibration")
def demo_hawkes_calibration(
    mu: float = 1.0,
    alpha: float = 0.8,
    beta: float = 1.5,
    horizon: float = 1500.0,
    seed: int = 7,
) -> dict:
    """Simulate a Hawkes flow with known params and MLE-recover them."""
    if alpha >= beta:
        raise HTTPException(status_code=422, detail="require alpha < beta (branching ratio < 1)")
    horizon = max(50.0, min(horizon, 10_000.0))
    try:
        return demo_service.demo_hawkes_calibration(
            mu=mu, alpha=alpha, beta=beta, horizon=horizon, seed=seed
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/hawkes/fit")
def hawkes_fit(req: HawkesFitRequest) -> dict:
    """Maximum-likelihood fit of an exponential Hawkes process to event times."""
    try:
        return fit_hawkes(req.times, horizon=req.horizon)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/api/agent/quote")
def agent_quote(
    mid: float = 1000.0,
    inventory: float = 0.0,
    gamma: float = 0.1,
    sigma: float = 1.0,
    k: float = 1.5,
    horizon: float = 1.0,
) -> dict:
    try:
        maker = AvellanedaStoikovMaker(gamma=gamma, sigma=sigma, k=k, horizon=horizon)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return asdict(maker.quote(mid, inventory))
