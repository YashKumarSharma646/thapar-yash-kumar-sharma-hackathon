"""Baseline impact score (1-10): an explainable blend of event severity, sentiment strength
and abnormal attention ("buzz"). Every component is reported on the evidence card.

Used when no calibrated model is available; CalibratedImpact (below) replaces it once fitted.
"""

import json
import math
from collections import defaultdict, deque
from datetime import datetime, timedelta
from pathlib import Path

from ripple.config import MODELS_DIR
from ripple.schemas import EventType, SourceType

EVENT_SEVERITY = {
    EventType.CREDIT_EVENT: 0.9,
    EventType.GEOPOLITICAL: 0.8,
    EventType.MACROECONOMIC: 0.7,
    EventType.REGULATORY_LEGAL: 0.6,
    EventType.OPERATIONAL: 0.6,
    EventType.EARNINGS: 0.6,
    EventType.MERGER_ACQUISITION: 0.5,
    EventType.PRODUCT_LAUNCH: 0.35,
    EventType.OTHER: 0.2,
}
# Social posts are unverified until corroborated, so they carry less weight than news.
SOURCE_CREDIBILITY = {SourceType.NEWS: 1.0, SourceType.GDELT: 0.95, SourceType.SOCIAL: 0.8}
WEIGHTS = {"severity": 0.35, "sentiment": 0.35, "buzz": 0.30}

WINDOW = timedelta(hours=24)
BASELINE = timedelta(days=7)


class BuzzTracker:
    """Mentions per entity in the last 24h relative to that entity's trailing 7-day daily average."""

    def __init__(self):
        self._mentions: dict[str, deque[datetime]] = defaultdict(deque)

    def observe(self, entity: str, ts: datetime) -> float:
        q = self._mentions[entity]
        q.append(ts)
        while q and q[0] < ts - WINDOW - BASELINE:
            q.popleft()
        recent = sum(1 for t in q if t >= ts - WINDOW)
        baseline_daily = (len(q) - recent) / BASELINE.days
        ratio = recent / (baseline_daily + 1.0)
        return min(1.0, math.log2(1.0 + ratio) / 4.0)  # ratio ~15x saturates at 1.0


def impact_score(
    event_type: EventType, event_confidence: float, sentiment: float, buzz: float, source: SourceType
) -> tuple[float, dict[str, float]]:
    components = {
        "severity": EVENT_SEVERITY[event_type] * event_confidence,
        "sentiment": abs(sentiment),
        "buzz": buzz,
    }
    raw = sum(WEIGHTS[k] * v for k, v in components.items()) * SOURCE_CREDIBILITY[source]
    score = round(1.0 + 9.0 * min(1.0, raw), 2)
    breakdown = {k: round(v, 3) for k, v in components.items()} | {
        "source_credibility": SOURCE_CREDIBILITY[source],
        "raw": round(raw, 3),
    }
    return score, breakdown


# --- Calibrated impact (Day 5) -------------------------------------------------------------------
# A logistic model of P(material move | signal), fitted on historical signals against realised
# abnormal returns (scripts/calibrate_impact.py). Impact is a linear map of the log-odds onto 1-10,
# so every feature's contribution is additive and shown on the evidence card.

IMPACT_MODEL = MODELS_DIR / "impact_calibration.json"


def impact_features(
    event_type: EventType, sentiment: float, buzz: float, source: SourceType, severity: float | None
) -> dict[str, float]:
    # Without a model severity (rules backend), fall back to the event-class prior.
    sev = EVENT_SEVERITY[event_type] if severity is None else severity
    social = float(source == SourceType.SOCIAL)
    features = {
        "severity": sev,
        "negative": max(0.0, -sentiment),
        "positive": max(0.0, sentiment),
        "severity_x_sentiment": sev * abs(sentiment),
        "buzz": buzz,
        "social": social,
        # Tweet volume in the source data reflects collection more than attention, so social posts get
        # their own buzz/severity/sentiment slopes instead of sharing the news ones.
        "social_x_buzz": social * buzz,
        "social_x_severity": social * sev,
        "social_x_negative": social * max(0.0, -sentiment),
    }
    features |= {f"event:{e.value}": float(event_type == e) for e in EventType if e != EventType.OTHER}
    return features


# Evidence-card grouping of feature contributions.
_GROUP = {
    "severity": "severity", "social_x_severity": "severity", "severity_x_sentiment": "severity",
    "negative": "sentiment", "positive": "sentiment", "social_x_negative": "sentiment",
    "buzz": "buzz", "social_x_buzz": "buzz",
}


class CalibratedImpact:
    def __init__(self, params: dict):
        self.intercept = params["intercept"]
        self.weights = params["weights"]
        self.lo, self.hi = params["logit_lo"], params["logit_hi"]

    @classmethod
    def load(cls, path: Path = IMPACT_MODEL) -> "CalibratedImpact | None":
        return cls(json.loads(path.read_text())) if path.exists() else None

    def __call__(
        self, event_type: EventType, sentiment: float, buzz: float, source: SourceType, severity: float | None
    ) -> tuple[float, dict[str, float]]:
        x = impact_features(event_type, sentiment, buzz, source, severity)
        contrib = {k: self.weights.get(k, 0.0) * v for k, v in x.items()}
        groups: dict[str, float] = defaultdict(float)
        for k, v in contrib.items():
            groups[_GROUP.get(k, "event" if k.startswith("event:") else "source")] += v
        logit = self.intercept + sum(contrib.values())
        score = round(1.0 + 9.0 * min(1.0, max(0.0, (logit - self.lo) / (self.hi - self.lo))), 2)
        breakdown = {
            "p_material_move": round(1.0 / (1.0 + math.exp(-logit)), 4),
            "severity": round(x["severity"], 3),
            "buzz": round(buzz, 3),
            "source_credibility": SOURCE_CREDIBILITY[source],
            # log-odds contributions, grouped for the evidence card
        } | {f"contrib_{g}": round(v, 3) for g, v in groups.items()}
        return score, breakdown
