"""REST API exposing risk signals to downstream consumers."""

from fastapi import FastAPI, Query

from ripple import __version__
from ripple.bus import get_client
from ripple.config import STREAM_SIGNALS, STREAM_STRESS
from ripple.schemas import RiskSignal, StressResult
from ripple.stress_test.portfolio import load_portfolio

app = FastAPI(title="Ripple API", version=__version__)


@app.get("/health")
def health() -> dict:
    try:
        redis_ok = get_client().ping()
    except Exception:
        redis_ok = False
    return {"status": "ok", "redis": redis_ok, "version": __version__}


@app.get("/signals", response_model=list[RiskSignal])
def latest_signals(
    limit: int = Query(50, ge=1, le=10_000),
    entity: str | None = None,
    min_impact: float = Query(1.0, ge=1.0, le=10.0),
) -> list[RiskSignal]:
    """Most recent signals first, optionally filtered by entity and minimum impact score."""
    filtered = entity is not None or min_impact > 1.0
    entries = get_client().xrevrange(STREAM_SIGNALS, count=10_000 if filtered else limit)
    signals = [RiskSignal.model_validate_json(fields["data"]) for _, fields in entries]
    if entity:
        signals = [s for s in signals if s.entity.upper() == entity.upper()]
    signals = [s for s in signals if s.impact_score >= min_impact]
    return signals[:limit]


@app.get("/stats")
def stats() -> dict:
    """Engine counters (received / filtered / signals) and the replay's simulated clock."""
    client = get_client()
    counters = {k: int(v) for k, v in client.hgetall("ripple:engine_stats").items()}
    return {"engine": counters, "replay_clock": client.get("ripple:replay_clock")}


@app.get("/stress", response_model=list[StressResult])
def stress_results(limit: int = Query(20, ge=1, le=1000)) -> list[StressResult]:
    """Module B: most recent stress tests first."""
    entries = get_client().xrevrange(STREAM_STRESS, count=limit)
    return [StressResult.model_validate_json(fields["data"]) for _, fields in entries]


@app.get("/portfolio")
def portfolio() -> list[dict]:
    """The synthetic wholesale portfolio that Module B stress-tests."""
    return load_portfolio().astype(object).where(lambda d: d.notna(), None).to_dict(orient="records")
