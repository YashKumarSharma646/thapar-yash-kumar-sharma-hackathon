"""Entity linking: map free text to tickers in the tracked universe.

Headlines are linked from their text (aliases and $cashtags), never from the dataset's ticker
column, which records the page a headline was listed on rather than its subject.
"""

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ripple.config import DATA_DIR

from ripple.engine.regions import MARKET  # noqa: F401  (re-exported: pseudo-entity for market-wide news)


@dataclass(frozen=True)
class Company:
    ticker: str
    name: str
    sector: str


class EntityLinker:
    def __init__(self, companies_csv: Path = DATA_DIR / "reference" / "companies.csv"):
        df = pd.read_csv(companies_csv)
        self.companies = {r.ticker: Company(r.ticker, r.name, r.sector) for r in df.itertuples()}
        alias_to_ticker = {alias.lower(): r.ticker for r in df.itertuples() for alias in r.aliases.split("|")}
        self._alias_to_ticker = alias_to_ticker
        aliases = sorted(alias_to_ticker, key=len, reverse=True)
        self._alias_re = re.compile(r"\b(?:" + "|".join(re.escape(a) for a in aliases) + r")\b", re.IGNORECASE)
        self._cashtag_re = re.compile(r"\$([A-Z]{1,5})\b")

    def link(self, text: str, hints: list[str] | None = None) -> tuple[list[str], list[str]]:
        """Return (tickers, matched surface forms), ordered by first mention."""
        found: dict[str, str] = {}
        for m in self._alias_re.finditer(text):
            found.setdefault(self._alias_to_ticker[m.group(0).lower()], m.group(0))
        for m in self._cashtag_re.finditer(text):
            if m.group(1) in self.companies:
                found.setdefault(m.group(1), m.group(0))
        for ticker in hints or []:
            if ticker in self.companies:
                found.setdefault(ticker, ticker)
        return list(found), list(found.values())
