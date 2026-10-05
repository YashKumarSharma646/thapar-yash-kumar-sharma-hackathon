"""Region entities for market-wide news.

Macro, geopolitical and credit news with no company named becomes a region entity when it names a
region (e.g. "Turkish lira plunges" -> TURKEY), otherwise MARKET. Each region gets its own attention
baseline (buzz) and is calibrated against its own equity market (region ETF vs SPY).
"""

import re

MARKET = "MARKET"  # pseudo-entity for market-wide news that names no region

# entity id -> (display name, keyword pattern, Yahoo ETF used as the region's market)
REGIONS = {
    "TURKEY": ("Turkey", r"turk\w*|lira|erdogan|ankara|istanbul", "TUR"),
    "CHINA": ("China", r"china|chinese|beijing|yuan|renminbi|xi jinping|huawei", "FXI"),
    "EUROPE": ("Europe", r"europe\w*|\beu\b|brexit|\beuro\b|eurozone|germany|german|italy|italian|france|french|\becb\b", "VGK"),
    "INDIA": ("India", r"india\w*|rupee|modi|mumbai", "EPI"),
    "LATAM": ("LatAm", r"brazil\w*|argentin\w*|mexic\w*|\bpeso|venezuel\w*|latin america", "ILF"),
}
_COMPILED = {entity: re.compile(pattern, re.IGNORECASE) for entity, (_, pattern, _) in REGIONS.items()}
REGION_ETF = {entity: etf for entity, (_, _, etf) in REGIONS.items()}
REGION_NAME = {entity: name for entity, (name, _, _) in REGIONS.items()}
NON_COMPANY = {MARKET} | set(REGIONS)


def detect_region_entity(text: str) -> str | None:
    for entity, pattern in _COMPILED.items():
        if pattern.search(text):
            return entity
    return None


def detect_region(text: str) -> str | None:
    """Display name of the first region named in the text (e.g. "Turkey")."""
    entity = detect_region_entity(text)
    return REGION_NAME[entity] if entity else None


def is_company(entity: str) -> bool:
    return entity not in NON_COMPANY
