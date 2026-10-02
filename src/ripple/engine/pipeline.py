"""RiskEngine: RawDocument batch -> RiskSignal list. Pure logic, no I/O, so it is unit-testable."""

from collections import Counter
from datetime import datetime, timedelta
from typing import Protocol

from ripple.engine.entities import MARKET, EntityLinker
from ripple.engine.events import classify_event
from ripple.engine.impact import BuzzTracker, impact_score
from ripple.engine.relevance import keep_social
from ripple.schemas import EventType, Evidence, RawDocument, RiskSignal, SourceType

# Market-wide themes still matter downstream (Module B) even when no company is named.
MARKET_EVENTS = {EventType.GEOPOLITICAL, EventType.MACROECONOMIC, EventType.CREDIT_EVENT}
REPLAY_RESET_GAP = timedelta(days=1)


class Scorer(Protocol):
    def score(self, texts: list[str]) -> list[float]: ...


class RiskEngine:
    def __init__(self, sentiment: Scorer, linker: EntityLinker | None = None):
        self.sentiment = sentiment
        self.linker = linker or EntityLinker()
        self.reset()

    def reset(self) -> None:
        self.buzz = BuzzTracker()
        self.stats: Counter[str] = Counter()
        self._clock: datetime | None = None

    def process(self, docs: list[RawDocument]) -> list[RiskSignal]:
        candidates = []
        for doc in docs:
            # A replay restart sends the clock back in time: start a fresh session so stale
            # attention history doesn't distort buzz scores.
            if self._clock is not None and doc.published_at < self._clock - REPLAY_RESET_GAP:
                self.reset()
            self._clock = max(self._clock or doc.published_at, doc.published_at)
            self.stats["received"] += 1
            if doc.source == SourceType.SOCIAL and not keep_social(doc.text):
                self.stats["filtered_noise"] += 1
                continue
            event = classify_event(doc.text)
            tickers, mentions = self.linker.link(doc.text, doc.tickers_hint)
            if not tickers and event.event_type in MARKET_EVENTS:
                tickers = [MARKET]
            if not tickers:
                self.stats["no_entity"] += 1
                continue
            candidates.append((doc, event, tickers, mentions))

        sentiments = self.sentiment.score([doc.text for doc, *_ in candidates])

        signals = []
        for (doc, event, tickers, mentions), sentiment in zip(candidates, sentiments):
            for ticker in tickers:
                buzz = self.buzz.observe(ticker, doc.published_at)
                impact, breakdown = impact_score(event.event_type, event.confidence, sentiment, buzz, doc.source)
                signals.append(
                    RiskSignal(
                        doc_id=doc.doc_id,
                        published_at=doc.published_at,
                        source=doc.source,
                        entity=ticker,
                        sentiment_score=sentiment,
                        event_type=event.event_type,
                        event_confidence=event.confidence,
                        impact_score=impact,
                        headline=doc.text[:280],
                        evidence=Evidence(
                            key_phrases=mentions + event.phrases,
                            impact_breakdown=breakdown,
                            source_urls=[doc.url] if doc.url else [],
                        ),
                    )
                )
        self.stats["signals"] += len(signals)
        return signals
