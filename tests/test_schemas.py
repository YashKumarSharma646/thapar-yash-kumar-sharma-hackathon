from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from ripple.schemas import EventType, RiskSignal, SourceType


def make_signal(**overrides) -> RiskSignal:
    fields = dict(
        doc_id="d1",
        published_at=datetime(2023, 3, 10, tzinfo=timezone.utc),
        source=SourceType.NEWS,
        entity="SIVB",
        sentiment_score=-0.9,
        event_type=EventType.CREDIT_EVENT,
        event_confidence=0.95,
        impact_score=9.2,
        headline="SVB Financial shares halted after failed capital raise",
    )
    return RiskSignal(**(fields | overrides))


def test_signal_roundtrips_through_json():
    signal = make_signal()
    assert RiskSignal.model_validate_json(signal.model_dump_json()) == signal


@pytest.mark.parametrize("field,value", [("sentiment_score", 1.5), ("impact_score", 0), ("impact_score", 11)])
def test_signal_rejects_out_of_range_scores(field, value):
    with pytest.raises(ValidationError):
        make_signal(**{field: value})
