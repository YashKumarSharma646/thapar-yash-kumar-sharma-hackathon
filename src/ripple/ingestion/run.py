"""Ingestor service entrypoint."""

import logging

from ripple.bus import get_client, publish
from ripple.config import INGEST_MODE, REPLAY_SPEED, STREAM_RAW_TEXT, STREAM_SIGNALS, STREAM_STRESS
from ripple.ingestion.replay import load_documents, replay

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("ripple.ingestion")


def main() -> None:
    if INGEST_MODE != "replay":
        raise SystemExit(f"INGEST_MODE={INGEST_MODE!r} not supported yet (only 'replay')")

    client = get_client()
    docs = load_documents()
    log.info("Replaying %d documents (%s -> %s) at %.0fx", len(docs), docs[0].published_at, docs[-1].published_at, REPLAY_SPEED)
    # New replay session: clear the previous run's signals so consumers start from a clean slate.
    client.delete("ripple:replay_clock", STREAM_SIGNALS, STREAM_STRESS, "ripple:engine_stats")
    for i, doc in enumerate(replay(docs, REPLAY_SPEED), 1):
        publish(client, STREAM_RAW_TEXT, doc)
        client.set("ripple:replay_clock", doc.published_at.isoformat())
        if i % 500 == 0:
            log.info("published %d/%d (sim time %s)", i, len(docs), doc.published_at)
    log.info("Replay finished")


if __name__ == "__main__":
    main()
