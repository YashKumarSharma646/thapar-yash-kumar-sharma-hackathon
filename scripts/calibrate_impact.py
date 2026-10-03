"""Fit the calibrated impact model and test it on the held-out replay window.

1. Select the regularisation on a time split of the historical set (fit < 2019, validate 2019-2020).
2. Refit on all historical signals; map log-odds onto 1-10.
3. Test on the replay window (15 Jul - 31 Aug 2018, never used in fitting): does impact rank
   realised abnormal moves better than the Day 2 hand-set formula?

Output: models/impact_calibration.json (parameters + metrics, read by the engine).
Usage: python scripts/build_calibration_set.py && python scripts/calibrate_impact.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ripple.calibration.returns import MATERIAL_Z, abnormal_z, attach_moves, load_prices  # noqa: E402
from ripple.engine.analyzers import load_analyzer  # noqa: E402
from ripple.engine.impact import IMPACT_MODEL, CalibratedImpact, impact_features  # noqa: E402
from ripple.engine.pipeline import RiskEngine  # noqa: E402
from ripple.ingestion.replay import load_documents  # noqa: E402
from ripple.schemas import EventType, SourceType  # noqa: E402

CALIBRATION = ROOT / "data" / "processed" / "calibration_signals.parquet"
ALERT = 7.0  # impact at or above this triggers a stress test (STRESS_IMPACT_THRESHOLD)
ALERT_TOP_PCT = 3.0


def features(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            impact_features(EventType(e), s, b, SourceType(src), None if pd.isna(sev) else sev)
            for e, s, b, src, sev in zip(df.event_type, df.sentiment, df.buzz, df.source, df.model_severity)
        ],
        index=df.index,
    )


def replay_test_set() -> pd.DataFrame:
    engine = RiskEngine(load_analyzer(), impact=None)  # baseline impact, plus the raw features
    rows = []
    for i in range(0, len(docs := load_documents()), 256):
        for s in engine.process(docs[i : i + 256]):
            b = s.evidence.impact_breakdown
            rows.append({"published_at": s.published_at, "entity": s.entity, "source": s.source.value,
                         "event_type": s.event_type.value, "sentiment": s.sentiment_score, "buzz": b["buzz"],
                         "model_severity": b.get("model_severity"), "baseline_impact": s.impact_score,
                         "headline": s.headline})
    return attach_moves(pd.DataFrame(rows), abnormal_z(load_prices())).dropna(subset=["z"])


def ranking_metrics(score: pd.Series, z: pd.Series, entity_day: pd.Series) -> dict:
    material = z.abs() >= MATERIAL_Z
    # Many signals share one stock-day; also score at that level (the strongest signal of the day).
    day = pd.DataFrame({"score": score, "m": material, "k": entity_day}).groupby("k").agg(score=("score", "max"), m=("m", "max"))
    alerts = score >= ALERT
    return {
        "auc_signal": roc_auc_score(material, score),
        "auc_stock_day": roc_auc_score(day.m, day.score),
        "spearman_abs_z": spearmanr(score, z.abs()).statistic,
        "alerts_share": alerts.mean(),
        "alert_precision": material[alerts].mean() if alerts.any() else float("nan"),
        "base_rate": material.mean(),
    }


def main() -> None:
    hist = pd.read_parquet(CALIBRATION)
    hist["y"] = hist.z.abs() >= MATERIAL_Z
    X = features(hist)
    early = pd.to_datetime(hist.published_at, utc=True) < pd.Timestamp("2019-01-01", tz="UTC")
    print(f"historical signals: {len(hist):,} (fit {early.sum():,}, validate {(~early).sum():,}), "
          f"material rate {hist.y.mean():.1%}")

    best = None
    for C in [0.01, 0.1, 1.0, 10.0]:
        m = LogisticRegression(C=C, max_iter=2000).fit(X[early], hist.y[early])
        auc = roc_auc_score(hist.y[~early], m.decision_function(X[~early]))
        print(f"  C={C:<5} validation AUC {auc:.3f}")
        best = max(best or (auc, C), (auc, C))
    print(f"  baseline formula validation AUC {roc_auc_score(hist.y[~early], hist.baseline_impact[~early]):.3f}")

    model = LogisticRegression(C=best[1], max_iter=2000).fit(X, hist.y)
    logits = model.decision_function(X)
    params = {
        "intercept": float(model.intercept_[0]),
        "weights": dict(zip(X.columns, map(float, model.coef_[0]))),
        # Impact 1 at the 10th percentile of historical log-odds; scaled so that impact >= 7 (the
        # stress-test trigger) means "top 3% of historical signals by predicted move probability".
        "logit_lo": float(lo := np.percentile(logits, 10)),
        "logit_hi": float(lo + 1.5 * (np.percentile(logits, 100 - ALERT_TOP_PCT) - lo)),
    }
    calibrated = CalibratedImpact(params)

    test = replay_test_set()
    test["impact"] = [
        calibrated(EventType(e), s, b, SourceType(src), sev)[0]
        for e, s, b, src, sev in zip(test.event_type, test.sentiment, test.buzz, test.source, test.model_severity)
    ]
    key = test.entity + "|" + test.event_day.astype(str)
    metrics = {
        "fit": {"signals": int(len(hist)), "C": best[1], "validation_auc": round(best[0], 4)},
        "test_replay_window": {
            "signals": int(len(test)),
            "baseline": ranking_metrics(test.baseline_impact, test.z, key),
            "calibrated": ranking_metrics(test.impact, test.z, key),
        },
    }
    metrics = json.loads(json.dumps(metrics), parse_float=lambda x: round(float(x), 4))
    IMPACT_MODEL.write_text(json.dumps(params | {"metrics": metrics}, indent=2))

    print("\nweights (log-odds per unit):")
    for k, v in sorted(params["weights"].items(), key=lambda kv: -abs(kv[1])):
        print(f"  {k:<28} {v:+.3f}")
    print("\ntest on replay window:\n" + pd.DataFrame(metrics["test_replay_window"]).drop("signals", axis=1).to_string())
    pd.set_option("display.width", 220)
    for col in ["baseline_impact", "impact"]:
        print(f"\ntop 10 by {col}:")
        print(test.sort_values(col, ascending=False).head(10)[["event_day", "entity", "event_type", "sentiment", col, "z", "headline"]]
              .assign(headline=lambda d: d.headline.str[:60], event_day=lambda d: d.event_day.dt.strftime("%m-%d")).round(2).to_string(index=False))
    print(f"\nwrote {IMPACT_MODEL.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
