"""Signal -> stress scenario.

A scenario is a set of factor shocks: equity returns, rate moves, FX moves, credit-spread moves.
- Market-wide events (Geopolitical, Macroeconomic, sovereign/EM Credit Event) replay a historical
  episode from data/reference/scenario_library.json, re-targeted to the region named in the headline.
- Company events shock that issuer's equity and credit, sized in units of its own volatility.
- Severity scales with the signal: multiplier = clip((impact - 5) / 5, 0.4, 1.0), so impact 10 replays
  the full historical episode and the alert threshold (7) replays 40% of it.
"""

import json
from dataclasses import dataclass, field

from ripple.config import DATA_DIR
from ripple.engine.regions import REGION_NAME, detect_region, is_company
from ripple.schemas import EventType, RiskSignal

LIBRARY_JSON = DATA_DIR / "reference" / "scenario_library.json"
EM_REGIONS = {"Turkey", "China", "India", "LatAm"}

# Company-event templates: equity shock in units of the stock's 2-day abnormal-return vol, and the
# issuer's credit-spread widening (bp) when the news is negative. Assumptions, stated on the dashboard.
COMPANY_SHOCKS = {
    EventType.CREDIT_EVENT: (4.0, 400),
    EventType.EARNINGS: (3.0, 60),
    EventType.REGULATORY_LEGAL: (2.0, 50),
    EventType.OPERATIONAL: (2.0, 60),
    EventType.MERGER_ACQUISITION: (1.5, 40),
    EventType.GEOPOLITICAL: (1.0, 25),
    EventType.MACROECONOMIC: (1.0, 25),
    EventType.PRODUCT_LAUNCH: (1.0, 10),
    EventType.OTHER: (1.0, 20),
}
MARKET_TEMPLATE = {
    EventType.GEOPOLITICAL: "trade_war",
    EventType.MACROECONOMIC: "rates_shock",
    EventType.CREDIT_EVENT: "em_currency_crisis",
}


@dataclass
class Scenario:
    name: str
    title: str
    narrative: str
    multiplier: float
    affected_region: str | None = None
    equity: dict[str, float] = field(default_factory=dict)  # "global", "region:<R>", "<TICKER>"
    rates_bp: dict[str, float] = field(default_factory=dict)  # by currency
    fx: dict[str, float] = field(default_factory=dict)  # % change of the currency's USD value
    spreads_bp: dict[str, float] = field(default_factory=dict)  # "IG", "HY", "EM", "region:<R>", "<TICKER>"
    assumptions: list[str] = field(default_factory=list)


def load_library() -> dict:
    return json.loads(LIBRARY_JSON.read_text())


def severity_multiplier(impact: float) -> float:
    return round(min(1.0, max(0.4, (impact - 5.0) / 5.0)), 2)


def build_scenario(signal: RiskSignal, library: dict | None = None) -> Scenario:
    library = library or load_library()
    m = severity_multiplier(signal.impact_score)
    region = REGION_NAME.get(signal.entity) or detect_region(signal.headline)
    template = MARKET_TEMPLATE.get(signal.event_type)
    # A Credit Event about a named company is idiosyncratic; only a sovereign/EM one replays the crisis.
    if signal.event_type == EventType.CREDIT_EVENT and is_company(signal.entity) and region not in EM_REGIONS:
        template = None

    if template:
        ep = library[template]
        shocks = ep["shocks"]
        target = region or ep["affected_region"]
        scenario = Scenario(
            name=template, title=ep["title"], narrative=ep["narrative"], multiplier=m, affected_region=target,
            equity={k: v * m for k, v in shocks["equity"].items() if k == "global" or k == f"region:{target}"
                    or (k == "region:EM" and target in EM_REGIONS)},
            rates_bp={k: v * m for k, v in shocks["rates_bp"].items()},
            fx={k: v * m for k, v in shocks["fx"].items()},
            spreads_bp={k: v * m for k, v in shocks["spreads_bp"].items() if not k.startswith("region:")},
            assumptions=[f"Replays {ep['title']} ({ep['window'][0]} to {ep['window'][1]}) x {m:.0%}"],
        )
        region_bp = next((v for k, v in shocks["spreads_bp"].items() if k.startswith("region:")), 0)
        if target and region_bp:
            scenario.spreads_bp[f"region:{target}"] = region_bp * m
            if target != ep["affected_region"]:
                scenario.assumptions.append(f"Episode re-targeted from {ep['affected_region']} to {target} (named in the headline)")
    else:
        scenario = Scenario(
            name="company_event", title=f"{signal.event_type.value} shock: {signal.entity}",
            narrative="Issuer-specific repricing of the named company's equity and credit.", multiplier=m,
        )

    if is_company(signal.entity):
        k_sigma, spread_bp = COMPANY_SHOCKS[signal.event_type]
        vol = library.get("equity_vol_2d", {}).get(signal.entity, 0.02)
        # Direction follows the news: clearly positive news lifts the stock; anything else is stressed down.
        direction = 1.0 if signal.sentiment_score > 0.2 else -1.0
        scenario.equity[signal.entity] = direction * k_sigma * vol * m
        if direction < 0:
            scenario.spreads_bp[signal.entity] = spread_bp * m
        scenario.assumptions.append(
            f"{signal.entity}: equity {direction * k_sigma * m:+.1f} sigma (2-day vol {vol:.1%}), "
            f"spread {'+' + str(round(spread_bp * m)) + 'bp' if direction < 0 else 'unchanged (positive news)'}"
        )
    return scenario
