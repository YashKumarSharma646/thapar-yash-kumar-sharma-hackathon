"""Run the engine over historical news + tweets OUTSIDE the replay window and attach realised moves.

Output: data/processed/calibration_signals.parquet, one row per signal with the engine's features
(event, sentiment, model severity, buzz, source) and z, the 2-day abnormal move of the entity's stock.
The replay window (15 Jul - 31 Aug 2018) is excluded here and used as the test set.

Usage: python scripts/build_calibration_set.py [--since 2014-01-01]
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from build_sample import END, MACRO_THEMES, RAW, START, alias_pattern, companies, load_news, spread_time  # noqa: E402
from ripple.calibration.returns import abnormal_z, attach_moves, load_prices  # noqa: E402
from ripple.engine.analyzers import load_analyzer  # noqa: E402
from ripple.engine.pipeline import RiskEngine  # noqa: E402
from ripple.schemas import RawDocument, SourceType  # noqa: E402

OUT = ROOT / "data" / "processed" / "calibration_signals.parquet"


def historical_docs(since: str) -> pd.DataFrame:
    cos = companies()
    news = load_news()[["headline", "date"]]
    news["date"] = news.date.dt.normalize()
    news = news.rename(columns={"headline": "text"}).drop_duplicates(subset="text")
    keep = news.text.str.contains(alias_pattern(cos)) | news.text.str.contains(MACRO_THEMES, case=False, regex=True)
    news = news[keep].assign(source="news", hint="")

    t = pd.read_csv(RAW / "stock_tweets" / "full_dataset-release.csv", usecols=["TWEET", "STOCK", "DATE"],
                    encoding_errors="replace", lineterminator="\n", low_memory=False)
    t["date"] = pd.to_datetime(t.DATE, format="%d/%m/%Y", errors="coerce")
    name_to_ticker = dict(zip(cos.tweet_name, cos.ticker))
    t = t[t.STOCK.isin(name_to_ticker)].dropna(subset=["TWEET"])
    tweets = pd.DataFrame({"text": t.TWEET.astype(str).str.replace(r"\s+", " ", regex=True).str.strip(),
                           "date": t.date, "source": "social", "hint": t.STOCK.map(name_to_ticker)})

    docs = pd.concat([news, tweets]).dropna(subset=["date"])
    docs = docs[(docs.date >= since) & ~docs.date.between(START, f"{END} 23:59")]
    docs["published_at"] = [spread_time(d, x) for d, x in zip(docs.date, docs.text)]
    return docs.sort_values("published_at")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--since", default="2014-01-01")
    args = parser.parse_args()

    docs = historical_docs(args.since)
    print(f"{len(docs):,} historical docs ({docs.source.value_counts().to_dict()})")
    engine = RiskEngine(load_analyzer())
    rows, start = [], time.perf_counter()
    for i in range(0, len(docs), 256):
        batch = [
            RawDocument(source=SourceType(r.source), source_name="calibration", published_at=r.published_at,
                        text=r.text, tickers_hint=[r.hint] if r.hint else [])
            for r in docs.iloc[i : i + 256].itertuples()
        ]
        for s in engine.process(batch):
            b = s.evidence.impact_breakdown
            rows.append({"published_at": s.published_at, "entity": s.entity, "source": s.source.value,
                         "event_type": s.event_type.value, "event_confidence": s.event_confidence,
                         "sentiment": s.sentiment_score, "model_severity": b.get("model_severity"),
                         "buzz": b["buzz"], "baseline_impact": s.impact_score, "headline": s.headline})
        if i % 25600 == 0:
            print(f"  {i:,}/{len(docs):,} ({time.perf_counter() - start:.0f}s)")
    signals = attach_moves(pd.DataFrame(rows), abnormal_z(load_prices())).dropna(subset=["z"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    signals.to_parquet(OUT, index=False)
    print(f"{OUT.relative_to(ROOT)}: {len(signals):,} signals with realised moves, "
          f"material (|z|>=2): {(signals.z.abs() >= 2).mean():.1%}")


if __name__ == "__main__":
    main()
