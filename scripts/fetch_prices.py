"""Download daily adjusted closes for the tracked companies, SPY and the region ETFs (Yahoo Finance via yfinance).

Output: data/processed/prices.csv (not committed: Yahoo data, regenerable). Used by impact
calibration (Day 5) and the stress-test shock library (Module B).
Usage: python scripts/fetch_prices.py [--start 2009-01-01] [--end 2021-01-01]
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ripple.calibration.returns import PRICES_CSV, YAHOO_SYMBOL  # noqa: E402
from ripple.engine.regions import REGION_ETF  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2009-01-01")
    parser.add_argument("--end", default="2021-01-01")
    args = parser.parse_args()

    # Companies, the US market (SPY) and one ETF per region entity (TUR, FXI, VGK, EPI, ILF).
    tickers = pd.read_csv(ROOT / "data" / "reference" / "companies.csv").ticker.tolist() + ["SPY"] + list(REGION_ETF.values())
    symbols = [YAHOO_SYMBOL.get(t, t) for t in tickers]
    close = yf.download(symbols, start=args.start, end=args.end, auto_adjust=True, progress=False)["Close"]
    close = close.rename(columns={v: k for k, v in YAHOO_SYMBOL.items()})[tickers]
    missing = [t for t in tickers if close[t].isna().all()]
    assert not missing, f"no prices for {missing}"
    PRICES_CSV.parent.mkdir(parents=True, exist_ok=True)
    close.round(4).to_csv(PRICES_CSV)
    print(f"{PRICES_CSV.relative_to(ROOT)}: {close.shape[0]} days x {close.shape[1]} tickers, "
          f"{close.index.min():%Y-%m-%d} -> {close.index.max():%Y-%m-%d}")
    print("first valid date per ticker (late listings):", close.apply(pd.Series.first_valid_index).dt.strftime("%Y-%m").loc[lambda s: s > args.start[:7]].to_dict())


if __name__ == "__main__":
    main()
