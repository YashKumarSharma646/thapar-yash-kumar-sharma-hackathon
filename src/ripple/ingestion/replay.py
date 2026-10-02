"""Replay ingestor: streams the committed historical sample as if it were arriving live.

Both sources (news + social) are merged on their original timestamps and released on an
accelerated clock: REPLAY_SPEED simulated seconds pass per real second.
"""

import logging
import time
from collections.abc import Iterator
from pathlib import Path

import pandas as pd

from ripple.config import DATA_DIR
from ripple.schemas import RawDocument, SourceType

log = logging.getLogger("ripple.ingestion")

SAMPLE_FILES = [DATA_DIR / "sample" / "news_2018.csv", DATA_DIR / "sample" / "tweets_2018.csv"]


def load_documents(files: list[Path] = SAMPLE_FILES) -> list[RawDocument]:
    df = pd.concat([pd.read_csv(f, keep_default_na=False) for f in files])
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True)
    df = df.sort_values("published_at", kind="stable")
    return [
        RawDocument(
            source=SourceType(r.source),
            source_name=r.source_name,
            published_at=r.published_at.to_pydatetime(),
            text=r.text,
            url=r.url or None,
            tickers_hint=[t for t in str(r.tickers_hint).split("|") if t],
        )
        for r in df.itertuples()
    ]


def replay(docs: list[RawDocument], speed: float) -> Iterator[RawDocument]:
    """Yield documents paced so that simulated time advances `speed` x faster than wall time."""
    if not docs:
        return
    sim_start, wall_start = docs[0].published_at, time.monotonic()
    for doc in docs:
        due = (doc.published_at - sim_start).total_seconds() / speed
        delay = due - (time.monotonic() - wall_start)
        if delay > 0:
            time.sleep(delay)
        yield doc
