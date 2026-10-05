# Ripple

**AI/NLP Risk Engine for real-time financial risk signals, with event-driven portfolio stress testing.**

S&P Global × Crisil Campus Hackathon 2026 · Case Study submission · Yash Kumar Sharma, Thapar Institute of Engineering and Technology

> 🎥 Demo video: _link coming soon_ · 📊 Deck: [`docs/presentation.pdf`](docs/presentation.pdf)

## What it does

1. **Ingests** text from two sources (financial news headlines and stock-related tweets), replayed from public datasets on an accelerated clock, so the demo is reproducible offline.
2. **Analyzes** each item with NLP and emits a structured `RiskSignal`: sentiment (-1 to 1), event class, impact score (1-10) and an evidence card explaining the scores.
3. **Publishes** signals over Redis Streams, a REST API and a JSONL file.
4. **Stress-tests** a synthetic wholesale banking portfolio (loans, bonds, derivatives) whenever a high-impact event is detected (Module B), and shows value, P&L, expected loss and CET1 before and after.

Architecture: see [`docs/architecture.md`](docs/architecture.md).

## Quickstart

**Requirements:** Docker Desktop (with Docker Compose v2) and Git.

```bash
git clone https://github.com/YashKumarSharma646/thapar-yash-kumar-sharma-hackathon.git
cd thapar-yash-kumar-sharma-hackathon
docker compose up --build
```

- Dashboard: http://localhost:8501
- API docs: http://localhost:8000/docs
- Output files on the host: `data/processed/signals.jsonl`, `data/processed/stress_results.jsonl`

No API keys are needed and nothing is downloaded at run time: the NLP model ships in the repo (`models/ripple-nlp/`, 34 MB int8 ONNX).

**What you'll see:** the ingestor replays 15 Jul – 31 Aug 2018 news headlines and tweets (8,556 documents) on an accelerated clock (~10 minutes end to end). The dashboard updates every 3 seconds; watch Facebook (FB) around 25–26 Jul 2018, its record Q2 earnings crash.

| Service | Port | Role |
|---|---|---|
| `ingestor` | – | Replays the two sources into Redis Streams |
| `engine` | – | NLP risk engine: raw text → `RiskSignal` |
| `stress_test` | – | Module B: subscribes to signals, stress-tests the portfolio on impact ≥ 7 |
| `api` | 8000 | REST: `/signals`, `/stress`, `/portfolio`, `/stats`, `/health` |
| `dashboard` | 8501 | Streamlit: risk signals, stress tests, what-if scenarios |
| `redis` | 6379 | Message bus (Redis Streams) |

To replay again: `docker compose restart ingestor`. To change speed, set `REPLAY_SPEED` in `.env.example` (simulated seconds per real second).

## Local development (without Docker)

Requires Python 3.11+ and a running Redis.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt -r requirements-engine.txt
pip install -r requirements-finbert.txt   # optional: baseline backend + export test
pytest                          # unit tests (no model download needed)
python scripts/run_offline.py   # run the engine over the sample without Redis
```

Rebuilding the sample from the raw Kaggle datasets (optional): `python scripts/download_data.py` then `python scripts/build_sample.py`.

## NLP model (Day 4: LLM-teacher distillation)

The engine's sentiment, event type and severity come from one small multi-task model, distilled from an open LLM:

1. **Teacher:** `Qwen/Qwen2.5-7B-Instruct` labelled 20.8k headlines and tweets (event, sentiment, severity 1–5) on a Colab L4 GPU.
2. **Student:** `BAAI/bge-small-en-v1.5` (33M params) fine-tuned with three heads, exported to int8 ONNX. It runs at ~140–200 docs/s on a laptop CPU with no PyTorch.
3. **Evaluation:** on the held-out replay window (4,941 docs, never trained on), agreement with the teacher:

| | Day 2 baseline | Distilled student |
|---|---|---|
| Event type, macro-F1 | 0.57 (keyword rules) | **0.75** |
| Sentiment, macro-F1 | 0.53 (FinBERT) | **0.77** |
| Severity, Spearman ρ | – | **0.75** |

Reproduce: `python scripts/build_label_corpus.py`, then run [`notebooks/04_label_and_distill.ipynb`](notebooks/04_label_and_distill.ipynb) on Colab (L4). Teacher labels: `data/labeling/teacher_labels.csv`. The baseline backend stays available via `NLP_BACKEND=rules`.

## Impact calibration (Day 5)

Impact (1–10) is calibrated against real price reactions, not hand-set weights. A logistic model estimates
**P(material move)**, where a material move is a 2-day market-adjusted return of at least 2σ for that stock (Yahoo Finance prices, SPY as the market).
Inputs are the model's severity, sentiment, event type, buzz and source; every input's log-odds contribution is shown on the evidence card.
Impact ≥ 7, the stress-test trigger, means the top 3% of historical signals by predicted move probability.

Fitted on 129k historical signals (2014–2020, replay window excluded), compared with the hand-set Day 2 formula
(both use the same buzz, with the warm-up fix, and region entities):

| | Hand-set formula | Calibrated |
|---|---|---|
| Validation 2019–2020 (28k signals): AUC | 0.661 | 0.655 |
| Replay window (7.4k signals): alert precision, impact ≥ 7 followed by a ≥ 2σ move | 32% | **47%** (base rate 11%) |
| Replay window: signal AUC / stock-day AUC | 0.657 / 0.645 | 0.654 / 0.615 |
| Replay window: Spearman with abs(abnormal move) | 0.144 | 0.154 |

On ranking the two are on par; the calibrated model is kept because impact then has a stated meaning (a probability,
and a threshold defined as the top 3% of history) and its alerts are more precise. The strongest single driver is buzz.
The Facebook Q2-miss headline (26 Jul 2018, −8.7σ) and the Turkish lira crisis (10 Aug 2018, Turkey ETF −7.3σ) are
among the replay's top alerts.
Reproduce: `python scripts/fetch_prices.py && python scripts/build_calibration_set.py && python scripts/calibrate_impact.py`.

## Module B: event-driven portfolio stress testing

The `stress_test` service subscribes to the engine's signals. A signal with impact ≥ 7 triggers a stress test of a
synthetic wholesale book, unless the same scenario for the same target ran in the previous 12 hours of simulated time.

**Portfolio** (`data/reference/portfolio.csv`, seeded generator in `src/ripple/stress_test/portfolio.py`): 97 positions,
$12.6bn notional: loans to 51 borrowers (the 30 tracked companies plus synthetic corporates in Turkey, China, Europe,
India and LatAm), corporate and sovereign bonds, interest-rate swaps, FX forwards, CDS and equity swaps.

**Scenarios** (`data/reference/scenario_library.json`, built by `scripts/build_scenario_library.py`):

| Signal | Scenario | Shocks (full severity) |
|---|---|---|
| Geopolitical | US–China trade-war escalation, May 2019 (measured) | SPY −4.5%, 10y −13bp, CNY −1.3%, HY +48bp; named region +120bp (assumed) |
| Credit Event (sovereign / EM) | Turkish lira crisis, Aug 2018 (measured) | TRY −28%, EM spreads +43bp; Turkey +250bp (assumed) |
| Macroeconomic | Taper tantrum, May–Jun 2013 (measured) | 10y +92bp, HY +86bp, EM +115bp, TRY −7.5% |
| Company events | Issuer-specific | Equity −k·σ of that stock (k = 1–4 by event type), issuer spread +10–400bp |

Shocks scale with impact: ×40% at the threshold (7), the full historical episode at 10. A headline naming a region
re-targets the regional shock (e.g. an EU tariff story stresses European exposures).

**Revaluation and capital** (`src/ripple/stress_test/engine.py`): bonds by duration and convexity; swaps by DV01; CDS by
spread DV01; FX forwards and equity swaps by delta; FX translation of non-USD loans and bonds; loan expected loss
(PD × LGD × EAD) with PDs stressed in proportion to the spread move; Basel IRB credit RWA; CET1 ratio before and after
(CET1 capital set at 13% of baseline RWA, losses pre-tax). Simplifications are listed on the dashboard.

The dashboard's **What-if** tab runs any event type, entity and impact through the same scenarios and pricing.

## Repository structure

```
src/ripple/        Python package (ingestion, engine, api, stress_test, rebalancer, dashboard)
data/              Replay sample, portfolio, scenario library, labelling corpus (committed); raw + processed (generated)
docs/              Architecture and presentation
notebooks/         Colab notebooks for model training and calibration
scripts/           Data download and utility scripts
models/            Distilled NLP model (int8 ONNX) and impact calibration (committed)
tests/             Unit tests
```

## Data sources

All data is public or synthetic. See [`data/README.md`](data/README.md) for sources and citations.

## License

[MIT](LICENSE)
