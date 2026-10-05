"""Revalue the portfolio under a scenario and compute capital impact.

Pricing (sensitivity-based, the standard first-order approach for a stress screen):
  Bonds          dV = MV * (-D * dy + 0.5 * C * dy^2),  dy = rate move + credit-spread move
  Loans          dEL = EAD * LGD * (PD_stressed - PD);  PD_stressed = PD * (1 + dspread / spread_rating)
  Rate swaps     dV = direction * DV01 * drate_bp
  CDS            dV = direction * spread DV01 * dspread_bp   (direction +1 = protection bought)
  FX forwards    dV = direction * notional * dfx
  Equity swaps   dV = direction * notional * (beta * global equity + name-specific move)
  FX translation non-USD loans and bonds lose/gain value with their currency
Capital: Basel IRB (corporate) risk weights on loans, bonds and CDS sold, using baseline vs stressed PDs.
CET1 capital is set at 13% of baseline RWA (assumption); losses are taken pre-tax.
"""

import math
from statistics import NormalDist

import pandas as pd

from ripple.stress_test.portfolio import EM_REGIONS, PD_BY_RATING, SPREAD_BY_RATING
from ripple.stress_test.scenarios import Scenario

NON_USD_RATE_BETA = 0.6  # assumption: non-USD curves move 60% of the USD move
CET1_RATIO_BASE = 0.13
PD_CAP = 0.95
_N = NormalDist()


def irb_capital(pd_: float, lgd: float, maturity: float) -> float:
    """Basel II/III IRB capital requirement K per unit of EAD (corporate exposures)."""
    p = min(max(pd_, 0.0003), 0.9999)
    w = (1 - math.exp(-50 * p)) / (1 - math.exp(-50))
    r = 0.12 * w + 0.24 * (1 - w)
    b = (0.11852 - 0.05478 * math.log(p)) ** 2
    m = min(max(maturity, 1.0), 5.0)
    k = lgd * _N.cdf((_N.inv_cdf(p) + math.sqrt(r) * _N.inv_cdf(0.999)) / math.sqrt(1 - r)) - p * lgd
    return max(0.0, k * (1 + (m - 2.5) * b) / (1 - 1.5 * b))


def _ig(rating: str) -> bool:
    return rating in ("AAA", "AA", "A", "BBB")


def spread_move_bp(pos: pd.Series, s: Scenario) -> float:
    if pos.asset_class == "Sovereign bond" and pos.region in ("US", "Europe"):
        return 0.0  # core sovereigns: rates only
    bp = s.spreads_bp.get("IG" if _ig(pos.rating) else "HY", 0.0)
    if pos.region in EM_REGIONS:
        bp += s.spreads_bp.get("EM", 0.0)
    bp += s.spreads_bp.get(f"region:{pos.region}", 0.0)
    if isinstance(pos.ticker, str):
        bp += s.spreads_bp.get(pos.ticker, 0.0)
    return bp


def rate_move_bp(currency: str, s: Scenario) -> float:
    usd = s.rates_bp.get("USD", 0.0)
    return s.rates_bp.get(currency, usd if currency == "USD" else NON_USD_RATE_BETA * usd)


def equity_move(pos: pd.Series, s: Scenario) -> float:
    move = pos.beta * s.equity.get("global", 0.0)
    if pos.ticker != "MARKET":
        move += s.equity.get(pos.ticker, 0.0) + s.equity.get(f"region:{pos.region}", 0.0)
    return move


def revalue(book: pd.DataFrame, s: Scenario) -> pd.DataFrame:
    rows = []
    for pos in book.itertuples(index=False):
        p = pd.Series(pos._asdict())
        dspread = spread_move_bp(p, s)
        drate = rate_move_bp(p.currency, s)
        fx = s.fx.get(p.currency, 0.0) if p.currency != "USD" else 0.0
        mtm = el_before = el_after = 0.0
        pd_after = p.pd if p.asset_class == "Loan" else None
        if p.asset_class in ("Corporate bond", "Sovereign bond"):
            dy = (drate + dspread) / 1e4
            mtm = p.market_value_usd * (-p.duration * dy + 0.5 * p.convexity * dy * dy)
        elif p.asset_class == "Loan":
            pd_after = min(PD_CAP, p.pd * (1 + max(0.0, dspread) / SPREAD_BY_RATING[p.rating]))
            el_before = p.notional_usd * p.lgd * p.pd
            el_after = p.notional_usd * p.lgd * pd_after
        elif p.asset_class == "Interest rate swap":
            mtm = p.direction * p.dv01_usd * drate
        elif p.asset_class == "CDS":
            mtm = p.direction * p.dv01_usd * dspread
        elif p.asset_class == "FX forward":
            mtm = p.direction * p.notional_usd * fx
            fx = 0.0  # already the position's whole exposure
        elif p.asset_class == "Equity swap":
            mtm = p.direction * p.notional_usd * equity_move(p, s)
        fx_pnl = (p.market_value_usd + mtm) * fx if p.market_value_usd else 0.0
        rows.append({"position_id": p.position_id, "asset_class": p.asset_class, "counterparty": p.counterparty,
                     "region": p.region, "rating": p.rating, "spread_move_bp": dspread, "rate_move_bp": drate,
                     "mtm_pnl": mtm, "fx_pnl": fx_pnl, "provision": -(el_after - el_before),
                     "pd_before": p.pd if p.asset_class == "Loan" else None, "pd_after": pd_after})
    out = pd.DataFrame(rows)
    out["total_pnl"] = out.mtm_pnl + out.fx_pnl + out.provision
    return out


def credit_rwa(book: pd.DataFrame, pd_override: dict[str, float] | None = None) -> float:
    rwa = 0.0
    for p in book.itertuples(index=False):
        if p.asset_class == "Loan":
            pd_, lgd, ead = (pd_override or {}).get(p.position_id, p.pd), p.lgd, p.notional_usd
        elif p.asset_class == "Corporate bond" or (p.asset_class == "CDS" and p.direction < 0):
            pd_, lgd, ead = PD_BY_RATING[p.rating], 0.45, p.notional_usd
        else:
            continue
        rwa += 12.5 * irb_capital(pd_, lgd, p.maturity_y) * ead
    return rwa


def run_stress(book: pd.DataFrame, s: Scenario) -> dict:
    pos = revalue(book, s)
    loans = book.asset_class == "Loan"
    el_before = float((book.notional_usd * book.lgd * book.pd)[loans].sum())
    nav_before = float(book.market_value_usd.sum()) - el_before
    pnl = float(pos.total_pnl.sum())

    rwa_before = credit_rwa(book)
    stressed_pd = dict(zip(pos.position_id[pos.pd_after.notna()], pos.pd_after.dropna()))
    rwa_after = credit_rwa(book, stressed_pd)
    cet1 = CET1_RATIO_BASE * rwa_before

    by_class = pos.groupby("asset_class").total_pnl.sum()
    top = pos.reindex(pos.total_pnl.abs().sort_values(ascending=False).index).head(10)
    return {
        "value_before": nav_before,
        "value_after": nav_before + pnl,
        "pnl_total": pnl,
        "pnl_by_asset_class": {k: float(v) for k, v in by_class.items()},
        "pnl_by_region": {k: float(v) for k, v in pos.groupby("region").total_pnl.sum().items()},
        "expected_loss_before": el_before,
        "expected_loss_after": el_before - float(pos.provision.sum()),
        "rwa_before": rwa_before,
        "rwa_after": rwa_after,
        "cet1_ratio_before": CET1_RATIO_BASE,
        "cet1_ratio_after": (cet1 + pnl) / rwa_after,
        "top_positions": [
            {"position_id": r.position_id, "counterparty": r.counterparty, "asset_class": r.asset_class,
             "region": r.region, "pnl": float(r.total_pnl)} for r in top.itertuples() if abs(r.total_pnl) > 0
        ],
    }
