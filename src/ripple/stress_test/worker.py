"""Module B service: subscribe to risk signals, stress-test the portfolio on high-impact events.

A signal with impact >= STRESS_IMPACT_THRESHOLD triggers a stress test, unless the same scenario for the
same target already ran within the cooldown (simulated time), so one news burst yields one test.
Results go to the Redis stream ripple:stress_results and data/processed/stress_results.jsonl.
"""

import logging
import os
import socket
from datetime import datetime, timedelta

from ripple.bus import consume_batches, get_client, publish
from ripple.engine.regions import is_company
from ripple.config import DATA_DIR, STREAM_SIGNALS, STREAM_STRESS, STRESS_IMPACT_THRESHOLD
from ripple.schemas import RiskSignal, StressResult
from ripple.stress_test.engine import run_stress
from ripple.stress_test.portfolio import load_portfolio
from ripple.stress_test.scenarios import Scenario, build_scenario, load_library

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("ripple.stress_test")

COOLDOWN = timedelta(hours=12)
RESULTS_JSONL = DATA_DIR / "processed" / "stress_results.jsonl"


def stress_result(signal: RiskSignal, scenario: Scenario, book, library=None) -> StressResult:
    shocks = {
        "equity": scenario.equity, "rates_bp": scenario.rates_bp, "fx": scenario.fx, "spreads_bp": scenario.spreads_bp,
    }
    return StressResult(
        trigger=signal, scenario=scenario.name, scenario_title=scenario.title, narrative=scenario.narrative,
        severity_multiplier=scenario.multiplier, affected_region=scenario.affected_region,
        shocks={k: {kk: round(vv, 5) for kk, vv in v.items()} for k, v in shocks.items()},
        assumptions=scenario.assumptions, **run_stress(book, scenario),
    )


class Trigger:
    """Decides which signals start a stress test (threshold + per-target cooldown in simulated time)."""

    def __init__(self, threshold: float = STRESS_IMPACT_THRESHOLD, cooldown: timedelta = COOLDOWN):
        self.threshold, self.cooldown = threshold, cooldown
        self.last: dict[tuple, datetime] = {}

    def key(self, signal: RiskSignal, scenario: Scenario) -> tuple:
        return (scenario.name, signal.entity if is_company(signal.entity) else scenario.affected_region)

    def should_run(self, signal: RiskSignal, scenario: Scenario) -> bool:
        if signal.impact_score < self.threshold:
            return False
        k, t = self.key(signal, scenario), signal.published_at
        if any(t < seen - timedelta(days=1) for seen in self.last.values()):
            self.last.clear()  # replay restarted
        if k in self.last and t - self.last[k] < self.cooldown:
            return False
        self.last[k] = t
        return True


def main() -> None:
    book, library = load_portfolio(), load_library()
    client = get_client()
    trigger = Trigger()
    RESULTS_JSONL.parent.mkdir(parents=True, exist_ok=True)
    consumer = os.getenv("HOSTNAME", socket.gethostname())
    log.info("Stress tester ready: %d positions, threshold %.1f", len(book), trigger.threshold)
    with RESULTS_JSONL.open("a", encoding="utf-8") as sink:
        for batch in consume_batches(client, STREAM_SIGNALS, "stress_test", consumer, RiskSignal):
            for signal in batch:
                if signal.impact_score < trigger.threshold:
                    continue
                scenario = build_scenario(signal, library)
                if not trigger.should_run(signal, scenario):
                    continue
                result = stress_result(signal, scenario, book)
                publish(client, STREAM_STRESS, result, maxlen=5_000)
                sink.write(result.model_dump_json() + "\n")
                sink.flush()
                log.info("%s %s impact %.1f -> %s: P&L %+.1fm, CET1 %.2f%% -> %.2f%%", signal.entity,
                         signal.event_type.value, signal.impact_score, scenario.title, result.pnl_total / 1e6,
                         100 * result.cet1_ratio_before, 100 * result.cet1_ratio_after)


if __name__ == "__main__":
    main()
