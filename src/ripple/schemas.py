"""Data contracts shared by every service.

RawDocument  -> what ingestors publish (one piece of text from one source).
RiskSignal   -> what the engine publishes (structured, machine-readable risk signal).
"""

from datetime import datetime, timezone
from enum import Enum
from uuid import uuid4

from pydantic import BaseModel, Field


class SourceType(str, Enum):
    NEWS = "news"
    SOCIAL = "social"
    GDELT = "gdelt"


class EventType(str, Enum):
    GEOPOLITICAL = "Geopolitical"
    MACROECONOMIC = "Macroeconomic"
    CREDIT_EVENT = "Credit Event"
    MERGER_ACQUISITION = "Merger/Acquisition"
    PRODUCT_LAUNCH = "Product Launch"
    EARNINGS = "Earnings"
    REGULATORY_LEGAL = "Regulatory/Legal"
    OPERATIONAL = "Operational"
    OTHER = "Other"


class RawDocument(BaseModel):
    doc_id: str = Field(default_factory=lambda: uuid4().hex)
    source: SourceType
    source_name: str  # e.g. "kaggle_news", "kaggle_tweets", "gdelt"
    published_at: datetime
    text: str
    url: str | None = None
    tickers_hint: list[str] = Field(default_factory=list)  # tickers already tagged by the source, if any


class Evidence(BaseModel):
    """Why the engine produced this signal: shown as an evidence card in the dashboard."""

    key_phrases: list[str] = Field(default_factory=list)
    impact_breakdown: dict[str, float] = Field(default_factory=dict)
    source_urls: list[str] = Field(default_factory=list)
    model: str = ""  # NLP backend that produced the sentiment and event


class RiskSignal(BaseModel):
    signal_id: str = Field(default_factory=lambda: uuid4().hex)
    doc_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    published_at: datetime
    source: SourceType
    entity: str  # ticker or event/region name
    sentiment_score: float = Field(ge=-1.0, le=1.0)
    event_type: EventType
    event_confidence: float = Field(ge=0.0, le=1.0)
    impact_score: float = Field(ge=1.0, le=10.0)
    headline: str
    evidence: Evidence = Field(default_factory=Evidence)
