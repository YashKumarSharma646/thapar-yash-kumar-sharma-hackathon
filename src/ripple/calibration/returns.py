"""Abnormal price moves around news, the ground truth for impact calibration.

For a signal about ticker i published on day t (moved to the next trading day if t is not one):
    R_i  = close(t+1) / close(t-1) - 1        2-day window: covers news released intraday or after hours
    AR   = R_i - R_SPY                         market-adjusted abnormal return (MARKET signals: R_SPY itself;
                                               region signals: the region's ETF, e.g. TUR, minus SPY)
    z    = AR / sigma                          sigma: std of the 2-day AR over the previous 120 trading days
|z| is "how unusual was the move for this stock". |z| >= 2 is treated as a material move.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from ripple.config import DATA_DIR
from ripple.engine.regions import REGION_ETF

PRICES_CSV = DATA_DIR / "processed" / "prices.csv"
YAHOO_SYMBOL = {"FB": "META"}  # renamed tickers: our id -> current Yahoo symbol
MARKET_PROXY = "SPY"
VOL_WINDOW = 120
MATERIAL_Z = 2.0


def load_prices(path: Path = PRICES_CSV) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0, parse_dates=True).sort_index()


def abnormal_z(prices: pd.DataFrame) -> pd.DataFrame:
    """Wide frame (trading day x entity: tickers, regions, 'MARKET') of 2-day abnormal-move z-scores."""
    r2 = prices.shift(-1) / prices.shift(1) - 1
    ar = r2.drop(columns=MARKET_PROXY).sub(r2[MARKET_PROXY], axis=0)
    ar = ar.rename(columns={etf: entity for entity, etf in REGION_ETF.items()})
    ar["MARKET"] = r2[MARKET_PROXY]
    # Volatility from strictly earlier days (shift(2): the t-1 -> t+1 window must not overlap itself).
    sigma = ar.rolling(VOL_WINDOW, min_periods=VOL_WINDOW // 2).std().shift(2)
    return ar / sigma


def attach_moves(signals: pd.DataFrame, z: pd.DataFrame) -> pd.DataFrame:
    """Add event_day and z (abnormal move) to a frame with published_at and entity columns."""
    days = z.index
    published = pd.to_datetime(signals.published_at, utc=True).dt.tz_localize(None).dt.normalize()
    pos = np.searchsorted(days.values, published.values)  # first trading day on/after publication
    ok = pos < len(days)
    event_day = pd.Series(pd.NaT, index=signals.index, dtype="datetime64[ns]")
    event_day[ok] = days[pos[ok]]
    long = z.stack().rename("z")
    keys = pd.MultiIndex.from_arrays([event_day, signals.entity])
    out = signals.assign(event_day=event_day.values, z=long.reindex(keys).values)
    return out
