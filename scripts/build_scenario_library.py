"""Build the stress-scenario library from real market episodes (Yahoo Finance via yfinance).

Each market-wide scenario replays the factor moves observed over a historical window:
  equity  : % change of SPY (global) and regional ETFs (FXI China, TUR Turkey, EEM emerging markets)
  rates   : change in the US 10y yield (^TNX), bp
  fx      : % change in the foreign currency's USD value (TRY, CNY, EUR, INR)
  spreads : credit-spread change inferred from bond-ETF returns net of the rate move:
            dspread ~= -return / duration - drate  (LQD ~8.5y for IG, HYG ~3.7y for HY, EMB ~7y for EM)
Fields that cannot be measured from free data are marked as assumptions in `provenance`.

Output: data/reference/scenario_library.json (committed; the stress tester never needs network access).
Usage: python scripts/build_scenario_library.py
"""

import json
from pathlib import Path

import sys

import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ripple.calibration.returns import abnormal_z, load_prices  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "reference" / "scenario_library.json"

SYMBOLS = ["SPY", "FXI", "TUR", "EEM", "^TNX", "LQD", "HYG", "EMB", "TRY=X", "CNY=X", "EURUSD=X", "INR=X"]
DURATION = {"LQD": 8.5, "HYG": 3.7, "EMB": 7.0}

EPISODES = {
    "trade_war": {
        "title": "US-China trade-war escalation (May 2019)",
        "window": ("2019-05-03", "2019-05-13"),
        "event_types": ["Geopolitical"],
        "narrative": "US raises tariffs on $200bn of Chinese goods to 25%; China retaliates. "
                     "Global equities sell off, China-exposed assets and the yuan weaken, flight to Treasuries.",
        "affected_region": "China",
        "region_spread_bp": 120,  # assumption: China-exposed corporate spreads (no free CDS data)
    },
    "em_currency_crisis": {
        "title": "Turkish lira crisis (Aug 2018)",
        "window": ("2018-08-01", "2018-08-13"),
        "event_types": ["Credit Event"],
        "narrative": "Lira collapses amid US sanctions and doubts over central-bank independence; "
                     "Turkish sovereign and bank credit reprices, contagion to emerging markets.",
        "affected_region": "Turkey",
        "region_spread_bp": 250,  # assumption: Turkey 5y CDS widened by roughly this much in Aug 2018
    },
    "rates_shock": {
        "title": "Taper tantrum (May-Jun 2013)",
        "window": ("2013-05-02", "2013-06-24"),
        "event_types": ["Macroeconomic"],
        "narrative": "Fed signals QE tapering; US 10y yield jumps ~1pp, credit and EM assets reprice.",
        "affected_region": None,
        "region_spread_bp": 0,
    },
}


def main() -> None:
    first = min(e["window"][0] for e in EPISODES.values())
    px = yf.download(SYMBOLS, start=first, end="2019-12-31", auto_adjust=True, progress=False)["Close"].ffill()
    library = {}
    for key, ep in EPISODES.items():
        start, end = (px.index[px.index.get_indexer([d], method="bfill")[0]] for d in ep["window"])
        a, b = px.loc[start], px.loc[end]
        ret = lambda s: float(b[s] / a[s] - 1)  # noqa: E731
        d10y_bp = float((b["^TNX"] - a["^TNX"]) * 100)
        spread = {name: round(-ret(etf) / DURATION[etf] * 1e4 - d10y_bp, 1) for name, etf in [("IG", "LQD"), ("HY", "HYG"), ("EM", "EMB")]}
        region = ep["affected_region"]
        library[key] = {
            "title": ep["title"],
            "narrative": ep["narrative"],
            "event_types": ep["event_types"],
            "window": [f"{start:%Y-%m-%d}", f"{end:%Y-%m-%d}"],
            "affected_region": region,
            "shocks": {
                "equity": {"global": round(ret("SPY"), 4), "region:China": round(ret("FXI"), 4),
                           "region:Turkey": round(ret("TUR"), 4), "region:EM": round(ret("EEM"), 4)},
                "rates_bp": {"USD": round(d10y_bp, 1)},
                # quoted as USD per 1 unit of foreign currency: TRY=X is TRY per USD, so invert
                "fx": {"TRY": round(float(a["TRY=X"] / b["TRY=X"] - 1), 4), "CNY": round(float(a["CNY=X"] / b["CNY=X"] - 1), 4),
                       "INR": round(float(a["INR=X"] / b["INR=X"] - 1), 4), "EUR": round(ret("EURUSD=X"), 4)},
                "spreads_bp": spread | ({f"region:{region}": ep["region_spread_bp"]} if region else {}),
            },
            "provenance": {
                "equity, rates_bp, fx": f"measured: Yahoo Finance closes {start:%Y-%m-%d} -> {end:%Y-%m-%d}",
                "spreads_bp IG/HY/EM": "derived: bond-ETF return / duration net of the 10y move (LQD, HYG, EMB)",
                **({f"spreads_bp region:{region}": "assumption (no free CDS history)"} if region else {}),
                "rates_bp EUR/other": "assumption: non-USD curves move 60% of USD (engine)",
            },
        }
        print(f"{key:20} {ep['window']}  SPY {ret('SPY'):+.1%}  10y {d10y_bp:+.0f}bp  "
              f"TRY {library[key]['shocks']['fx']['TRY']:+.1%}  CNY {library[key]['shocks']['fx']['CNY']:+.1%}  spreads {spread}")

    # Company-specific shocks are sized in units of each stock's own 2-day abnormal-return volatility
    # (2017-2018, from data/processed/prices.csv written by scripts/fetch_prices.py).
    prices = load_prices()
    r2 = prices.shift(-1) / prices.shift(1) - 1
    ar = r2.drop(columns="SPY").sub(r2["SPY"], axis=0).loc["2017":"2018"]
    library["equity_vol_2d"] = {t: round(float(v), 4) for t, v in ar.std().items()}
    print("2-day abnormal vol:", {k: library["equity_vol_2d"][k] for k in ["FB", "NFLX", "KO", "JPM"]})
    OUT.write_text(json.dumps(library, indent=2))
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
