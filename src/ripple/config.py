"""Central configuration, read from environment variables (see .env.example)."""

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.getenv("DATA_DIR", ROOT_DIR / "data"))
MODELS_DIR = Path(os.getenv("MODELS_DIR", ROOT_DIR / "models"))

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Redis Streams used as the pub/sub backbone between services.
STREAM_RAW_TEXT = "ripple:raw_text"
STREAM_SIGNALS = "ripple:signals"

# Ingestion: "replay" (offline, default, reproducible) or "live" (GDELT + optional keyed APIs).
INGEST_MODE = os.getenv("INGEST_MODE", "replay")
REPLAY_SPEED = float(os.getenv("REPLAY_SPEED", "60"))  # simulated seconds per real second

# Optional API keys; the system must run fully without them.
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "")
ALPHAVANTAGE_KEY = os.getenv("ALPHAVANTAGE_KEY", "")

# Module B: a signal at or above this impact score triggers a stress test.
STRESS_IMPACT_THRESHOLD = float(os.getenv("STRESS_IMPACT_THRESHOLD", "7"))

SIGNALS_JSONL = DATA_DIR / "processed" / "signals.jsonl"
