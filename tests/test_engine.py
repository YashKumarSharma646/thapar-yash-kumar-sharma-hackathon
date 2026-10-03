from datetime import datetime, timedelta, timezone

import pytest

from ripple.engine.analyzers import RuleAnalyzer
from ripple.engine.entities import MARKET, EntityLinker
from ripple.engine.events import classify_event
from ripple.engine.impact import BuzzTracker, impact_score
from ripple.engine.pipeline import RiskEngine
from ripple.engine.relevance import keep_social
from ripple.schemas import EventType, RawDocument, SourceType

T0 = datetime(2018, 7, 26, 14, 0, tzinfo=timezone.utc)


class FixedScorer:
    def __init__(self, value: float = -0.8):
        self.value = value

    def score(self, texts: list[str]) -> list[float]:
        return [self.value] * len(texts)


@pytest.fixture(scope="module")
def linker() -> EntityLinker:
    return EntityLinker()


def doc(text: str, source: SourceType = SourceType.NEWS, hints: list[str] | None = None, ts=T0) -> RawDocument:
    return RawDocument(source=source, source_name="test", published_at=ts, text=text, tickers_hint=hints or [])


def test_linker_uses_text_not_dataset_tags(linker):
    tickers, mentions = linker.link("Facebook's Dismal Q2 Hits Tech Rally; Instagram growth slows")
    assert tickers == ["FB"]
    assert mentions == ["Facebook"]


def test_linker_cashtags_and_hints(linker):
    tickers, _ = linker.link("Loading up on $AAPL and $MSFT today", hints=["AMZN"])
    assert tickers == ["AAPL", "MSFT", "AMZN"]


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Facebook reports slower-than-expected revenue growth in Q2 earnings", EventType.EARNINGS),
        ("U.S. slaps new tariffs on $200 billion of Chinese goods as trade war escalates", EventType.GEOPOLITICAL),
        ("Turkish lira plunges as emerging markets sell off on inflation fears", EventType.MACROECONOMIC),
        ("Moody's downgrade pushes issuer toward default", EventType.CREDIT_EVENT),
        ("Turkey's sovereign debt downgraded to junk by Fitch", EventType.CREDIT_EVENT),
        ("Goldman downgrades Parker Hannifin, Rockwell Automation", EventType.OTHER),
        ("Disney completes acquisition of 21st Century Fox assets", EventType.MERGER_ACQUISITION),
        ("FTC probe into Facebook privacy practices widens", EventType.REGULATORY_LEGAL),
        ("A nice sunny day", EventType.OTHER),
    ],
)
def test_event_classifier(text, expected):
    assert classify_event(text).event_type == expected


def test_noise_filter():
    assert not keep_social("I posted a new video to Facebook https://t.co/abc")
    assert not keep_social("facebook funny af lol")
    assert keep_social("Facebook wipes $130 billion in market cap after revenue miss")


def test_buzz_rises_with_abnormal_attention():
    tracker = BuzzTracker()
    for day in range(7, 0, -1):  # quiet baseline: 2 mentions/day
        for _ in range(2):
            tracker.observe("FB", T0 - timedelta(days=day))
    quiet = tracker.observe("FB", T0)
    for _ in range(40):
        spike = tracker.observe("FB", T0)
    assert spike > quiet
    assert spike == pytest.approx(1.0, abs=0.05)


def test_impact_score_bounds_and_ordering():
    low, _ = impact_score(EventType.PRODUCT_LAUNCH, 0.5, 0.1, 0.0, SourceType.SOCIAL)
    high, breakdown = impact_score(EventType.CREDIT_EVENT, 0.95, -0.95, 1.0, SourceType.NEWS)
    assert 1.0 <= low < high <= 10.0
    assert set(breakdown) >= {"severity", "sentiment", "buzz", "source_credibility"}


def test_engine_resets_when_replay_restarts(linker):
    engine = RiskEngine(RuleAnalyzer(FixedScorer()), linker)
    engine.process([doc("Facebook revenue falls", ts=T0 + timedelta(days=d)) for d in range(5)])
    assert engine.stats["received"] == 5
    engine.process([doc("Facebook revenue falls", ts=T0)])  # clock jumps back 4 days
    assert engine.stats["received"] == 1


def test_pipeline_end_to_end(linker):
    engine = RiskEngine(RuleAnalyzer(FixedScorer(-0.8)), linker)
    signals = engine.process(
        [
            doc("Facebook wipes $130 billion in market cap after Q2 revenue miss"),
            doc("I posted a new video to Facebook", source=SourceType.SOCIAL, hints=["FB"]),
            doc("Trade war fears grow as new tariffs announced"),
            doc("Lovely weather in London"),
        ]
    )
    assert [(s.entity, s.event_type) for s in signals] == [("FB", EventType.EARNINGS), (MARKET, EventType.GEOPOLITICAL)]
    assert engine.stats["filtered_noise"] == 1 and engine.stats["no_entity"] == 1
    fb = signals[0]
    assert fb.sentiment_score == -0.8 and 1 <= fb.impact_score <= 10
    assert "Facebook" in fb.evidence.key_phrases
