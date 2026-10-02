"""Run the engine over the replay sample without Redis and write signals to a JSONL file.

Used for evaluation and calibration; the live system uses ripple.engine.worker instead.

Usage: python scripts/run_offline.py [--out data/processed/signals_offline.jsonl]
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ripple.engine.pipeline import RiskEngine  # noqa: E402
from ripple.engine.sentiment import SentimentModel  # noqa: E402
from ripple.ingestion.replay import load_documents  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "processed" / "signals_offline.jsonl")
    parser.add_argument("--batch", type=int, default=64)
    args = parser.parse_args()

    docs = load_documents()
    engine = RiskEngine(SentimentModel())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    with args.out.open("w", encoding="utf-8") as f:
        for i in range(0, len(docs), args.batch):
            for s in engine.process(docs[i : i + args.batch]):
                f.write(s.model_dump_json() + "\n")
    elapsed = time.perf_counter() - start
    print(f"{len(docs):,} docs in {elapsed:.0f}s ({len(docs) / elapsed:.0f} docs/s) -> {args.out}")
    print(dict(engine.stats))


if __name__ == "__main__":
    main()
