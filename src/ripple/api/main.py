"""REST API exposing risk signals to downstream consumers."""

from fastapi import FastAPI, Query

from ripple import __version__
from ripple.bus import get_client
from ripple.config import STREAM_SIGNALS
from ripple.schemas import RiskSignal

app = FastAPI(title="Ripple API", version=__version__)


@app.get("/health")
def health() -> dict:
    try:
        redis_ok = get_client().ping()
    except Exception:
        redis_ok = False
    return {"status": "ok", "redis": redis_ok, "version": __version__}


@app.get("/signals", response_model=list[RiskSignal])
def latest_signals(limit: int = Query(50, ge=1, le=1000), entity: str | None = None) -> list[RiskSignal]:
    entries = get_client().xrevrange(STREAM_SIGNALS, count=limit if entity is None else 1000)
    signals = [RiskSignal.model_validate_json(fields["data"]) for _, fields in entries]
    if entity:
        signals = [s for s in signals if s.entity.upper() == entity.upper()][:limit]
    return signals
