"""Download the public Kaggle datasets into data/raw/.

Requires a Kaggle API token at ~/.kaggle/kaggle.json (or KAGGLE_USERNAME / KAGGLE_KEY env vars)
and `pip install kaggle`. The app itself never needs this: it runs from the committed
samples in data/sample/. This script only fetches the full sets for training and calibration.

Usage:  python scripts/download_data.py [dataset_name ...]
"""

import sys
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

DATASETS = {
    # Sentiment labels (Financial PhraseBank derivative): evaluation / fine-tuning
    "financial_news_sentiment": "ankurzing/sentiment-analysis-for-financial-news",
    # Stock tweets with returns: social source for replay + impact calibration
    "stock_tweets": "thedevastator/tweet-sentiment-s-impact-on-stock-returns",
    # Dated, ticker-tagged news headlines: news source for replay + impact calibration
    "stock_news": "miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests",
}


def main(names: list[str]) -> int:
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    failed = []
    for name in names or DATASETS:
        slug = DATASETS[name]
        target = RAW_DIR / name
        print(f"-> {name}: {slug}")
        try:
            api.dataset_download_files(slug, path=target, unzip=True, quiet=False)
        except Exception as e:  # keep going so one bad slug doesn't block the rest
            print(f"   FAILED: {e}")
            failed.append(name)

    if failed:
        print(f"\nFailed: {', '.join(failed)}")
        return 1
    print(f"\nAll datasets downloaded to {RAW_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
