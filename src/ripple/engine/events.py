"""Baseline event classifier: weighted keyword rules.

Transparent and fast; it also produces the matched phrases shown on the evidence card.
Day 4 replaces it with a distilled transformer, keeping the same interface.
"""

import re
from dataclasses import dataclass

from ripple.schemas import EventType

RULES: dict[EventType, list[tuple[str, float]]] = {
    EventType.CREDIT_EVENT: [
        (r"default(ed|s)?|bankrupt\w*|chapter 11|insolven\w*|downgrade[ds]?|junk|credit rating|debt crisis", 2.0),
        (r"moody'?s|fitch|s&p global ratings|liquidity crunch|missed (a )?payment|restructur\w*", 1.5),
    ],
    EventType.GEOPOLITICAL: [
        (r"tariffs?|trade war|trade tensions?|sanctions?|embargo|military|\bwar\b|missile|nuclear", 2.0),
        (r"north korea|iran|russia|brexit|geopolitic\w*|retaliat\w*|diplomat\w*", 1.5),
    ],
    EventType.MACROECONOMIC: [
        (r"federal reserve|\bfed\b|interest rates?|rate (hike|cut)|inflation|\bcpi\b|\bgdp\b|recession", 2.0),
        (r"unemployment|jobs report|payrolls|treasury yields?|bond yields?|lira|yuan|currency|emerging markets?", 1.5),
        (r"oil prices?|opec|economy|economic slowdown|stimulus", 1.0),
    ],
    EventType.MERGER_ACQUISITION: [
        (r"acquir\w*|acquisition|merger|merge[ds]?|takeover|buyout|to buy\b|bid for|deal to buy", 2.0),
        (r"\bstake\b|spin[- ]?off|divest\w*|\bdeal\b", 1.0),
    ],
    EventType.PRODUCT_LAUNCH: [
        (r"launch\w*|unveil\w*|introduc\w*|new (product|iphone|model|service|feature)|rolls? out|release[ds]?", 1.5),
    ],
    EventType.EARNINGS: [
        (r"earnings|\beps\b|revenue|profit|quarterly results|\bq[1-4]\b|guidance|beats?|miss(es|ed)?|outlook", 1.5),
        (r"user growth|sales (rose|fell|grew)|forecast|margins?", 1.0),
    ],
    EventType.REGULATORY_LEGAL: [
        (r"lawsuit|\bsue[ds]?\b|\bsec\b|\bftc\b|\bdoj\b|antitrust|probe|investigat\w*|\bfined?\b|penalty|regulat\w*", 2.0),
        (r"court|settle\w*|congress|senate|hearing|gdpr|privacy law", 1.0),
    ],
    EventType.OPERATIONAL: [
        (r"breach|hack(ed|ers?)?|outage|recall\w*|strike|layoffs?|job cuts|scandal|data leak|cyber\w*", 2.0),
        (r"fire|accident|shutdown|supply chain|disruption", 1.0),
    ],
}

_COMPILED = {
    event: [(re.compile(pattern, re.IGNORECASE), weight) for pattern, weight in rules]
    for event, rules in RULES.items()
}


@dataclass
class EventResult:
    event_type: EventType
    confidence: float
    phrases: list[str]


def classify_event(text: str) -> EventResult:
    scores: dict[EventType, float] = {}
    phrases: dict[EventType, list[str]] = {}
    for event, rules in _COMPILED.items():
        for pattern, weight in rules:
            hits = [m.group(0) for m in pattern.finditer(text)]
            if hits:
                scores[event] = scores.get(event, 0.0) + weight * len(hits)
                phrases.setdefault(event, []).extend(hits)

    if not scores:
        return EventResult(EventType.OTHER, 0.3, [])

    best = max(scores, key=scores.get)
    total = sum(scores.values())
    # Confidence grows with evidence strength and with the margin over competing classes.
    strength = min(1.0, scores[best] / 4.0)
    share = scores[best] / total
    confidence = round(0.35 + 0.6 * strength * share, 3)
    return EventResult(best, confidence, list(dict.fromkeys(p.lower() for p in phrases[best])))
