"""Build the committed replay sample (data/sample/) from the raw Kaggle datasets.

Window: 15 Jul - 31 Aug 2018, the period where both sources overlap densely and which contains
the Facebook Q2 earnings crash (25-26 Jul), US-China tariff escalation and the Turkish lira crisis.

- News:   headlines that mention a universe company or a macro/geopolitical theme, de-duplicated
          (the raw data lists the same headline under many unrelated tickers).
- Tweets: tweets about universe companies. Finance-relevant tweets are sampled uniformly (so
          relative volume spikes, the "buzz" signal, are preserved) plus a small slice of spam/chatter
          so the engine's noise filter has something realistic to remove.

Usage: python scripts/build_sample.py [--relevant-frac 0.25] [--noise-frac 0.005]
"""

import argparse
import hashlib
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ripple.engine.relevance import keep_social  # noqa: E402

RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "sample"
START, END = "2018-07-15", "2018-08-31"

MACRO_THEMES = (
    r"tariff|trade war|trade tension|sanction|Turkey|Turkish|lira|emerging market|Federal Reserve|\bFed\b"
    r"|interest rate|rate hike|inflation|recession|Treasury yield|bond yield|oil price|OPEC|Brexit|Iran"
    r"|North Korea|China|yuan|currency crisis|default|bankruptcy|downgrade"
)


def companies() -> pd.DataFrame:
    return pd.read_csv(ROOT / "data" / "reference" / "companies.csv")


def alias_pattern(cos: pd.DataFrame) -> re.Pattern:
    aliases = sorted({a for row in cos.aliases for a in row.split("|")}, key=len, reverse=True)
    return re.compile(r"\b(?:" + "|".join(re.escape(a) for a in aliases) + r")\b")


def spread_time(day: pd.Timestamp, key: str) -> pd.Timestamp:
    """Date-only rows get a deterministic pseudo-random time in 07:00-21:00 so replay is smooth."""
    minutes = int(hashlib.md5(key.encode()).hexdigest(), 16) % (14 * 60)
    return day + pd.Timedelta(hours=7, minutes=minutes)


def build_news(cos: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for name in ["raw_analyst_ratings.csv", "raw_partner_headlines.csv"]:
        df = pd.read_csv(RAW / "stock_news" / name, usecols=["headline", "url", "publisher", "date"])
        df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.tz_localize(None)
        frames.append(df)
    news = pd.concat(frames).dropna(subset=["headline", "date"])
    news = news[news.date.between(START, f"{END} 23:59")]
    news["headline"] = news.headline.str.strip()
    news = news.sort_values("date").drop_duplicates(subset=["headline"])

    keep = news.headline.str.contains(alias_pattern(cos)) | news.headline.str.contains(
        MACRO_THEMES, case=False, regex=True
    )
    news = news[keep].copy()
    midnight = news.date == news.date.dt.normalize()
    news.loc[midnight, "date"] = [spread_time(d, h) for d, h in zip(news.date[midnight], news.headline[midnight])]
    return pd.DataFrame(
        {
            "source": "news",
            "source_name": "kaggle_news:" + news.publisher.fillna("unknown"),
            "published_at": news.date,
            "text": news.headline,
            "url": news.url,
            "tickers_hint": "",
        }
    )


def build_tweets(cos: pd.DataFrame, relevant_frac: float, noise_frac: float) -> pd.DataFrame:
    t = pd.read_csv(
        RAW / "stock_tweets" / "full_dataset-release.csv",
        usecols=["TWEET", "STOCK", "DATE"],
        encoding_errors="replace",
        lineterminator="\n",
        low_memory=False,
    )
    t["DATE"] = pd.to_datetime(t["DATE"], format="%d/%m/%Y", errors="coerce")
    name_to_ticker = dict(zip(cos.tweet_name, cos.ticker))
    t = t[t.DATE.between(START, END) & t.STOCK.isin(name_to_ticker)].dropna(subset=["TWEET"])
    t["TWEET"] = t.TWEET.astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    relevant = t.TWEET.map(keep_social)
    print(f"tweets in window: {len(t):,}, finance-relevant: {relevant.sum():,} ({relevant.mean():.1%})")
    t = pd.concat(
        [t[relevant].sample(frac=relevant_frac, random_state=42), t[~relevant].sample(frac=noise_frac, random_state=42)]
    )
    return pd.DataFrame(
        {
            "source": "social",
            "source_name": "kaggle_tweets",
            "published_at": [spread_time(d, tw) for d, tw in zip(t.DATE, t.TWEET)],
            "text": t.TWEET,
            "url": "",
            "tickers_hint": t.STOCK.map(name_to_ticker),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--relevant-frac", type=float, default=0.25)
    parser.add_argument("--noise-frac", type=float, default=0.005)
    args = parser.parse_args()

    cos = companies()
    OUT.mkdir(parents=True, exist_ok=True)
    for name, df in [("news_2018.csv", build_news(cos)), ("tweets_2018.csv", build_tweets(cos, args.relevant_frac, args.noise_frac))]:
        df = df.sort_values("published_at")
        df.to_csv(OUT / name, index=False)
        size_mb = (OUT / name).stat().st_size / 1e6
        print(f"{name}: {len(df):,} rows, {size_mb:.1f} MB, {df.published_at.min()} -> {df.published_at.max()}")


if __name__ == "__main__":
    main()
