"""Build the text corpus that the LLM teacher labels on Colab (notebooks/04_label_and_distill.ipynb).

- train: news headlines (2009-2020) and finance-relevant tweets (2017-2018) from OUTSIDE the replay
         window. News is stratified by the rule classifier's guess so rare classes (credit, geopolitical,
         macro) are not drowned out by generic market chatter.
- eval:  every document of the replay sample (15 Jul - 31 Aug 2018) that passes the noise filter, i.e.
         exactly what the demo processes. Never used for training.

Output: data/labeling/corpus.csv (committed, so the notebook can fetch it from GitHub).
Usage:  python scripts/build_label_corpus.py [--news 11000] [--tweets 5000]
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from build_sample import END, MACRO_THEMES, RAW, START, alias_pattern, companies  # noqa: E402
from ripple.engine.events import classify_event  # noqa: E402
from ripple.engine.relevance import keep_social  # noqa: E402

OUT = ROOT / "data" / "labeling" / "corpus.csv"
SEED = 42


def norm(text: str) -> str:
    return re.sub(r"\W+", " ", text.lower()).strip()


def stratified(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Up to an equal share per rule class; leftover budget goes to the larger classes."""
    groups = {k: g.sample(frac=1, random_state=SEED) for k, g in df.groupby("rule_event")}
    share, picked = n // len(groups), []
    for g in groups.values():
        picked.append(g.iloc[:share])
    rest = pd.concat([g.iloc[share:] for g in groups.values()])
    picked.append(rest.sample(n=min(len(rest), n - sum(map(len, picked))), random_state=SEED))
    return pd.concat(picked)


def train_news(cos: pd.DataFrame, n: int) -> pd.DataFrame:
    frames = [
        pd.read_csv(RAW / "stock_news" / name, usecols=["headline", "date"])
        for name in ["raw_analyst_ratings.csv", "raw_partner_headlines.csv"]
    ]
    news = pd.concat(frames).dropna()
    news["date"] = pd.to_datetime(news.date, errors="coerce", utc=True).dt.tz_localize(None)
    news = news[~news.date.between(START, f"{END} 23:59")]
    news["text"] = news.headline.str.strip()
    news = news.drop_duplicates(subset="text")
    keep = news.text.str.contains(alias_pattern(cos)) | news.text.str.contains(MACRO_THEMES, case=False, regex=True)
    news = news[keep & news.text.str.len().between(25, 300)]
    news["rule_event"] = news.text.map(lambda t: classify_event(t).event_type.value)
    print(f"candidate news headlines: {len(news):,}\n{news.rule_event.value_counts().to_string()}")
    return stratified(news, n).assign(source="news")


def train_tweets(cos: pd.DataFrame, n: int) -> pd.DataFrame:
    t = pd.read_csv(
        RAW / "stock_tweets" / "full_dataset-release.csv",
        usecols=["TWEET", "STOCK", "DATE"],
        encoding_errors="replace",
        lineterminator="\n",
        low_memory=False,
    )
    t["DATE"] = pd.to_datetime(t.DATE, format="%d/%m/%Y", errors="coerce")
    t = t[t.STOCK.isin(set(cos.tweet_name)) & ~t.DATE.between(START, END)].dropna(subset=["TWEET"])
    t["text"] = t.TWEET.astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    t = t.drop_duplicates(subset="text")
    t = t[t.text.map(keep_social)]
    print(f"candidate relevant tweets: {len(t):,}")
    t = t.sample(n=min(n, len(t)), random_state=SEED)
    return t.assign(source="social", rule_event=t.text.map(lambda x: classify_event(x).event_type.value))


def eval_replay() -> pd.DataFrame:
    sample = ROOT / "data" / "sample"
    df = pd.concat([pd.read_csv(sample / "news_2018.csv"), pd.read_csv(sample / "tweets_2018.csv")])
    df = df[(df.source == "news") | df.text.map(keep_social)].drop_duplicates(subset="text")
    return df.assign(rule_event=df.text.map(lambda x: classify_event(x).event_type.value))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--news", type=int, default=11000)
    parser.add_argument("--tweets", type=int, default=5000)
    args = parser.parse_args()

    cos = companies()
    ev = eval_replay().assign(split="eval")
    tr = pd.concat([train_news(cos, args.news), train_tweets(cos, args.tweets)]).assign(split="train")
    tr = tr[~tr.text.map(norm).isin(set(ev.text.map(norm)))]  # no eval leakage via near-duplicates

    corpus = pd.concat([tr, ev])[["split", "source", "text", "rule_event"]].sample(frac=1, random_state=SEED)
    corpus.insert(0, "id", range(len(corpus)))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    corpus.to_csv(OUT, index=False)
    print(f"\n{OUT.relative_to(ROOT)}: {len(corpus):,} rows, {OUT.stat().st_size / 1e6:.1f} MB")
    print(corpus.groupby(["split", "source"]).size().to_string())
    print("\nrule-event mix (train):\n" + corpus[corpus.split == "train"].rule_event.value_counts().to_string())


if __name__ == "__main__":
    main()
