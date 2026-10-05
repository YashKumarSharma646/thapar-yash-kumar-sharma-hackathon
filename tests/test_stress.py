from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from ripple.schemas import EventType, RiskSignal, SourceType
from ripple.stress_test.engine import irb_capital, revalue, run_stress
from ripple.stress_test.portfolio import generate
from ripple.engine.regions import detect_region
from ripple.stress_test.scenarios import Scenario, build_scenario, load_library, severity_multiplier
from ripple.stress_test.worker import Trigger

T0 = datetime(2018, 8, 10, 14, tzinfo=timezone.utc)


@pytest.fixture(scope="module")
def book() -> pd.DataFrame:
    return generate()


def signal(entity="MARKET", event=EventType.GEOPOLITICAL, impact=9.0, sentiment=-0.9, headline="Trade war", ts=T0):
    return RiskSignal(doc_id="t", published_at=ts, source=SourceType.NEWS, entity=entity, sentiment_score=sentiment,
                      event_type=event, event_confidence=0.9, impact_score=impact, headline=headline)


def pnl(book, scenario, asset_class):
    out = revalue(book, scenario)
    return out[out.asset_class == asset_class].total_pnl.sum()


def test_portfolio_is_reproducible_and_mixed(book):
    assert book.equals(generate())
    assert {"Loan", "Corporate bond", "Sovereign bond", "Interest rate swap", "FX forward", "CDS", "Equity swap"} <= set(book.asset_class)


def test_irb_capital_rises_with_pd():
    ks = [irb_capital(p, 0.45, 2.5) for p in (0.0003, 0.002, 0.01, 0.05)]
    assert ks == sorted(ks) and 0 < ks[0] < ks[-1] < 0.45


def test_rate_rise_hurts_bonds_helps_pay_fixed_swap(book):
    s = Scenario(name="t", title="t", narrative="", multiplier=1.0, rates_bp={"USD": 100})
    assert pnl(book, s, "Sovereign bond") < 0 and pnl(book, s, "Corporate bond") < 0
    swap = revalue(book, s).set_index("counterparty").total_pnl
    assert swap["USD IRS pay fixed 10y"] > 0 and swap["EUR IRS receive fixed 5y"] < 0


def test_spread_widening_loss_and_cds_hedges(book):
    s = Scenario(name="t", title="t", narrative="", multiplier=1.0, spreads_bp={"region:Turkey": 300})
    out = revalue(book, s).set_index(["counterparty", "asset_class"])
    assert out.total_pnl["CDS 5y on Bosphorus Bank AS", "CDS"] > 0  # protection bought
    assert out.provision["Anatolia Steel AS", "Loan"] < 0  # higher expected loss on the loan
    assert out.loc[out.region == "US", "total_pnl"].abs().sum() == 0  # untouched outside Turkey


def test_currency_fall_hits_local_loans_and_short_forward_hedges(book):
    s = Scenario(name="t", title="t", narrative="", multiplier=1.0, fx={"TRY": -0.25})
    out = revalue(book, s).set_index(["counterparty", "asset_class"])
    assert out.fx_pnl["Anatolia Steel AS", "Loan"] < 0
    assert out.total_pnl["Short TRY forward 6m", "FX forward"] == pytest.approx(150e6 * 0.25)


def test_market_and_company_scenarios():
    lib = load_library()
    turkey = build_scenario(signal(event=EventType.CREDIT_EVENT, impact=10, headline="Turkish lira collapses"), lib)
    assert turkey.name == "em_currency_crisis" and turkey.multiplier == 1.0 and turkey.fx["TRY"] < -0.2
    europe = build_scenario(signal(headline="EU retaliates with tariffs on US goods"), lib)
    assert europe.name == "trade_war" and europe.affected_region == "Europe" and "region:Europe" in europe.spreads_bp
    tariffs_on_turkey = build_scenario(signal("TURKEY", EventType.GEOPOLITICAL, 9, -0.9, "Trump doubles tariffs on Turkish steel"), lib)
    assert tariffs_on_turkey.name == "em_currency_crisis" and tariffs_on_turkey.affected_region == "Turkey"
    rupee = build_scenario(signal("INDIA", EventType.CREDIT_EVENT, 10, -0.9, "Rupee crashes"), lib)
    assert rupee.fx["INR"] < -0.2 and "TRY" not in rupee.fx  # the crisis currency move follows the target region
    fb = build_scenario(signal("FB", EventType.EARNINGS, 10, -0.8, "Facebook misses"), lib)
    assert fb.name == "company_event" and fb.equity["FB"] < 0 and fb.spreads_bp["FB"] > 0 and not fb.rates_bp
    good = build_scenario(signal("MSFT", EventType.EARNINGS, 10, 0.9, "Microsoft beats"), lib)
    assert good.equity["MSFT"] > 0 and "MSFT" not in good.spreads_bp
    assert severity_multiplier(7) == 0.4 and severity_multiplier(10) == 1.0 and detect_region("Lira slides") == "Turkey"


def test_stress_reduces_value_and_capital(book):
    r = run_stress(book, build_scenario(signal(event=EventType.MACROECONOMIC, impact=10, headline="Fed hikes"), load_library()))
    assert r["value_after"] < r["value_before"] and r["cet1_ratio_after"] < r["cet1_ratio_before"]
    assert r["pnl_total"] == pytest.approx(sum(r["pnl_by_asset_class"].values()))


def test_trigger_threshold_and_cooldown():
    lib, trig = load_library(), Trigger(threshold=7.0, cooldown=timedelta(hours=12))
    s = signal(headline="China tariffs")
    sc = build_scenario(s, lib)
    assert trig.should_run(s, sc)
    assert not trig.should_run(signal(headline="China tariffs", ts=T0 + timedelta(hours=2)), sc)
    assert trig.should_run(signal(headline="China tariffs", ts=T0 + timedelta(hours=13)), sc)
    assert not trig.should_run(signal(impact=6.9, ts=T0 + timedelta(days=3)), sc)
    # inside the cooldown only a materially stronger signal (+1 impact over the last run at 9.0) re-runs
    assert not trig.should_run(signal(impact=9.8, headline="China tariffs", ts=T0 + timedelta(hours=14)), sc)
    assert trig.should_run(signal(impact=10.0, headline="China tariffs", ts=T0 + timedelta(hours=14)), sc)


def test_company_headline_never_triggers_market_scenario():
    sc = build_scenario(signal("NFLX", EventType.MACROECONOMIC, 9, -0.6, "Netflix says forex benefit was smaller"), load_library())
    assert sc.name == "company_event" and not sc.rates_bp and sc.equity["NFLX"] < 0
