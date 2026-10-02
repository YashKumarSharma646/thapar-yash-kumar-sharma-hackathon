"""Engine service: consume raw documents, publish risk signals to Redis and a JSONL file."""

import logging
import os
import socket

from ripple.bus import consume_batches, get_client, publish_many
from ripple.config import SIGNALS_JSONL, STREAM_RAW_TEXT, STREAM_SIGNALS
from ripple.engine.pipeline import RiskEngine
from ripple.engine.sentiment import SentimentModel
from ripple.schemas import RawDocument

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("ripple.engine")


def main() -> None:
    log.info("Loading models...")
    engine = RiskEngine(SentimentModel())
    client = get_client()
    SIGNALS_JSONL.parent.mkdir(parents=True, exist_ok=True)
    consumer = os.getenv("HOSTNAME", socket.gethostname())
    log.info("Engine ready, consuming %s", STREAM_RAW_TEXT)

    with SIGNALS_JSONL.open("a", encoding="utf-8") as sink:
        for batch in consume_batches(client, STREAM_RAW_TEXT, "engine", consumer, RawDocument):
            signals = engine.process(batch)
            if signals:
                publish_many(client, STREAM_SIGNALS, signals)
                sink.writelines(s.model_dump_json() + "\n" for s in signals)
                sink.flush()
            client.hset("ripple:engine_stats", mapping=dict(engine.stats))
            log.info("batch=%d signals=%d totals=%s", len(batch), len(signals), dict(engine.stats))


if __name__ == "__main__":
    main()
