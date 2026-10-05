"""Synthetic wholesale-banking book: loans, corporate and sovereign bonds, rate swaps, FX forwards, CDS
and equity swaps. Seeded and fully reproducible; no real client data.

Why synthetic: the brief's suggested transaction datasets are retail/card data, which do not describe
a wholesale book. Borrowers are the 30 companies the engine tracks (so a news signal hits real exposures)
plus synthetic corporates in the regions the scenarios stress (Turkey, China, Europe, India, LatAm).

Generate: python -m ripple.stress_test.portfolio  ->  data/reference/portfolio.csv
"""

import math

import numpy as np
import pandas as pd

from ripple.config import DATA_DIR

PORTFOLIO_CSV = DATA_DIR / "reference" / "portfolio.csv"
SEED = 7

# Long-run average 1y corporate default rates by rating grade (S&P-style), used as baseline PDs.
PD_BY_RATING = {"AAA": 0.0001, "AA": 0.0002, "A": 0.0006, "BBB": 0.002, "BB": 0.009, "B": 0.04, "CCC": 0.20}
# Typical credit spreads (bp) by grade: baseline yields, and the denominator of the PD stress.
SPREAD_BY_RATING = {"AAA": 40, "AA": 60, "A": 90, "BBB": 150, "BB": 300, "B": 500, "CCC": 1000}
EM_REGIONS = {"Turkey", "China", "India", "LatAm"}
RISK_FREE = {"USD": 0.029, "EUR": 0.005, "TRY": 0.18, "CNY": 0.035, "INR": 0.075}

TRACKED_RATINGS = {
    "AAPL": "AA", "MSFT": "AAA", "GOOGL": "AA", "AMZN": "AA", "FB": "A", "NFLX": "BB", "INTC": "A", "CSCO": "AA",
    "ORCL": "A", "IBM": "A", "ADBE": "A", "DIS": "A", "CMCSA": "A", "VZ": "BBB", "NKE": "AA", "SBUX": "BBB",
    "MCD": "BBB", "F": "BBB", "WMT": "AA", "COST": "A", "PEP": "A", "KO": "AA", "JPM": "A", "C": "BBB",
    "V": "AA", "MA": "A", "PYPL": "A", "BA": "A", "XOM": "AA", "PFE": "AA",
}
# name, region, sector, rating, loan currency
SYNTHETIC = [
    ("Anatolia Steel AS", "Turkey", "Materials", "BB", "TRY"),
    ("Bosphorus Bank AS", "Turkey", "Financials", "BB", "USD"),
    ("Ege Textiles AS", "Turkey", "Consumer Discretionary", "B", "TRY"),
    ("Marmara Energy AS", "Turkey", "Utilities", "BB", "USD"),
    ("Shenzhen Components Ltd", "China", "Technology", "BBB", "CNY"),
    ("Yangtze Logistics Co", "China", "Industrials", "BBB", "USD"),
    ("Pearl River Property Ltd", "China", "Real Estate", "BB", "USD"),
    ("Huabei Auto Parts Co", "China", "Consumer Discretionary", "BB", "CNY"),
    ("Rhine Chemicals GmbH", "Europe", "Materials", "BBB", "EUR"),
    ("Iberia Telecom SA", "Europe", "Communication Services", "BBB", "EUR"),
    ("Nordic Shipping ASA", "Europe", "Industrials", "BB", "USD"),
    ("Lombardy Machinery SpA", "Europe", "Industrials", "BBB", "EUR"),
    ("Gujarat Infra Ltd", "India", "Industrials", "BB", "INR"),
    ("Deccan Pharma Ltd", "India", "Health Care", "BBB", "USD"),
    ("Bengal Power Ltd", "India", "Utilities", "BB", "INR"),
    ("Petro Andes SA", "LatAm", "Energy", "BB", "USD"),
    ("Pampas Agro SA", "LatAm", "Consumer Staples", "B", "USD"),
    ("Rio Metals SA", "LatAm", "Materials", "BB", "USD"),
    ("Midwest Freight Inc", "US", "Industrials", "BB", "USD"),
    ("Gulf Coast Refining Inc", "US", "Energy", "BB", "USD"),
    ("Sunbelt Retail Inc", "US", "Consumer Discretionary", "B", "USD"),
]


def bond_risk(y: float, maturity: float) -> tuple[float, float]:
    """Modified duration and convexity of a par bond with annual coupon = yield."""
    times = np.arange(1, math.ceil(maturity) + 1)
    cf = np.full(len(times), y)
    cf[-1] += 1.0
    disc = (1 + y) ** -times
    price = (cf * disc).sum()
    macaulay = (times * cf * disc).sum() / price
    convexity = (times * (times + 1) * cf * disc).sum() / (price * (1 + y) ** 2)
    return macaulay / (1 + y), convexity


def generate(seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cos = pd.read_csv(DATA_DIR / "reference" / "companies.csv")
    names = [(r.name, "US", r.sector, TRACKED_RATINGS[r.ticker], "USD", r.ticker) for r in cos.itertuples()]
    names += [(n, reg, sec, rat, ccy, None) for n, reg, sec, rat, ccy in SYNTHETIC]
    rows = []

    def add(**kw):
        rows.append({"position_id": f"P{len(rows) + 1:03d}", "ticker": None, "pd": None, "lgd": None,
                     "duration": 0.0, "convexity": 0.0, "dv01_usd": 0.0, "direction": 1, "beta": 0.0} | kw)

    for name, region, sector, rating, ccy, ticker in names:
        ead = float(np.round(rng.lognormal(np.log(150 if ticker else 90), 0.5), 1)) * 1e6
        add(asset_class="Loan", counterparty=name, ticker=ticker, sector=sector, region=region, rating=rating,
            currency=ccy, notional_usd=ead, market_value_usd=ead, maturity_y=int(rng.integers(1, 8)),
            pd=PD_BY_RATING[rating], lgd=0.35 if rating in ("BB", "B") else 0.45)

    bond_issuers = [n for n in names if n[5] in {"AAPL", "AMZN", "FB", "NFLX", "F", "VZ", "C", "BA", "IBM", "SBUX"}]
    bond_issuers += [n for n in names if n[1] in EM_REGIONS | {"Europe"}][::2]
    for name, region, sector, rating, _, ticker in bond_issuers:
        maturity = int(rng.integers(3, 11))
        y = RISK_FREE["USD"] + SPREAD_BY_RATING[rating] / 1e4
        d, c = bond_risk(y, maturity)
        mv = float(np.round(rng.uniform(40, 160), 1)) * 1e6
        add(asset_class="Corporate bond", counterparty=name, ticker=ticker, sector=sector, region=region,
            rating=rating, currency="USD", notional_usd=mv, market_value_usd=mv, maturity_y=maturity,
            duration=round(d, 2), convexity=round(c, 1))

    for name, region, rating, ccy, mv, maturity in [
        ("US Treasury 2028", "US", "AAA", "USD", 800e6, 10), ("German Bund 2028", "Europe", "AAA", "EUR", 300e6, 10),
        ("Republic of Turkey 2028 (USD)", "Turkey", "BB", "USD", 150e6, 10),
        ("China Government Bond 2027", "China", "A", "CNY", 200e6, 9), ("India Government Bond 2028", "India", "BBB", "INR", 150e6, 10),
    ]:
        sovereign_spread = SPREAD_BY_RATING[rating] if ccy == "USD" and region != "US" else 0
        d, c = bond_risk(RISK_FREE[ccy] + sovereign_spread / 1e4, maturity)
        add(asset_class="Sovereign bond", counterparty=name, sector="Sovereign", region=region, rating=rating,
            currency=ccy, notional_usd=mv, market_value_usd=mv, maturity_y=maturity, duration=round(d, 2), convexity=round(c, 1))

    # Rate swaps hedge part of the bond book's duration. DV01 = P&L per +1bp parallel move (pay-fixed gains).
    for name, ccy, notional, maturity, direction in [("USD IRS pay fixed 10y", "USD", 1_000e6, 10, 1),
                                                     ("EUR IRS receive fixed 5y", "EUR", 400e6, 5, -1)]:
        d, _ = bond_risk(RISK_FREE[ccy] + 0.001, maturity)
        add(asset_class="Interest rate swap", counterparty=name, sector="Rates", region="US" if ccy == "USD" else "Europe",
            rating="AA", currency=ccy, notional_usd=notional, market_value_usd=0.0, maturity_y=maturity,
            dv01_usd=round(notional * d * 1e-4), direction=direction)

    # FX forwards: direction +1 = long the foreign currency. Short TRY hedges part of the TRY loans.
    for name, ccy, notional, direction in [("Short TRY forward 6m", "TRY", 150e6, -1), ("Long CNY forward 3m (client)", "CNY", 120e6, 1),
                                           ("Short INR forward 6m", "INR", 60e6, -1)]:
        add(asset_class="FX forward", counterparty=name, sector="FX", region={"TRY": "Turkey", "CNY": "China", "INR": "India"}[ccy],
            rating="A", currency=ccy, notional_usd=notional, market_value_usd=0.0, maturity_y=1, direction=direction)

    # CDS: direction +1 = protection bought (gains when the reference entity's spread widens).
    by_name = {n[0]: n for n in names}
    for ref, direction in [("Bosphorus Bank AS", 1), ("Pearl River Property Ltd", 1), ("Ford Motor Co.", 1),
                           ("Netflix Inc.", 1), ("International Business Machines", -1), ("Walmart Inc.", -1),
                           ("Rhine Chemicals GmbH", -1), ("Facebook Inc.", -1)]:
        name, region, sector, rating, _, ticker = by_name[ref]
        notional = 50e6
        add(asset_class="CDS", counterparty=f"CDS 5y on {name}", ticker=ticker, sector=sector, region=region,
            rating=rating, currency="USD", notional_usd=notional, market_value_usd=0.0, maturity_y=5,
            dv01_usd=round(notional * 4.5 * 1e-4), direction=direction)

    # Equity swaps (trading book), with an index hedge.
    for ticker, notional, beta in [("FB", 60e6, 1.2), ("AAPL", 50e6, 1.1), ("AMZN", 50e6, 1.3), ("NFLX", 30e6, 1.5),
                                   ("MSFT", 40e6, 1.1), ("GOOGL", 40e6, 1.1), ("BA", 30e6, 1.2), ("JPM", 40e6, 1.1),
                                   ("MARKET", -150e6, 1.0)]:
        name = "S&P 500 future (hedge)" if ticker == "MARKET" else cos.set_index("ticker").name[ticker]
        sector = "Index" if ticker == "MARKET" else cos.set_index("ticker").sector[ticker]
        add(asset_class="Equity swap", counterparty=name, ticker=ticker, sector=sector, region="US", rating="NR",
            currency="USD", notional_usd=abs(notional), market_value_usd=0.0, maturity_y=1,
            direction=1 if notional > 0 else -1, beta=beta)

    book = pd.DataFrame(rows)
    book[["notional_usd", "market_value_usd"]] = book[["notional_usd", "market_value_usd"]].round(0)
    return book


def load_portfolio() -> pd.DataFrame:
    return pd.read_csv(PORTFOLIO_CSV, keep_default_na=False, na_values={"ticker": [""], "pd": [""], "lgd": [""]})


if __name__ == "__main__":
    book = generate()
    PORTFOLIO_CSV.parent.mkdir(parents=True, exist_ok=True)
    book.to_csv(PORTFOLIO_CSV, index=False)
    summary = book.groupby("asset_class").agg(positions=("position_id", "count"), notional_bn=("notional_usd", "sum"))
    summary["notional_bn"] = (summary.notional_bn / 1e9).round(2)
    print(f"{PORTFOLIO_CSV}: {len(book)} positions\n{summary.to_string()}")
    print("by region (loans+bonds, $bn):", (book[book.market_value_usd > 0].groupby("region").market_value_usd.sum() / 1e9).round(2).to_dict())
