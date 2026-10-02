"""Baseline impact score (1-10): an explainable blend of event severity, sentiment strength
and abnormal attention ("buzz"). Every component is reported on the evidence card.

Day 5 calibrates this against realised abnormal returns.
"""

import math
from collections import defaultdict, deque
from datetime import datetime, timedelta

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
