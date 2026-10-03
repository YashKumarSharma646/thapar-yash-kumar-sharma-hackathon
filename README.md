# Ripple

**AI/NLP Risk Engine for real-time financial risk signals, with event-driven portfolio stress testing.**

S&P Global × Crisil Campus Hackathon 2026 · Case Study submission · Yash Kumar Sharma, Thapar Institute of Engineering and Technology

> 🎥 Demo video: _link coming soon_ · 📊 Deck: [`docs/presentation.pdf`](docs/presentation.pdf)

## What it does

1. **Ingests** text from two sources (financial news and stock-related social posts), replayed from public datasets or pulled live from GDELT.
2. **Analyzes** each item with NLP and emits a structured `RiskSignal`: sentiment (-1 to 1), event class, impact score (1-10) and an evidence card explaining the scores.
3. **Publishes** signals over Redis Streams, a REST API and a JSONL file.
4. **Stress-tests** a synthetic wholesale banking portfolio (loans, bonds, derivatives) when a high-impact event is detected (Module B).

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
- Signals file: `data/processed/signals.jsonl` (written on the host)

No API keys are needed and nothing is downloaded at run time: the NLP model ships in the repo (`models/ripple-nlp/`, 34 MB int8 ONNX).

**What you'll see:** the ingestor replays 15 Jul – 31 Aug 2018 news headlines and tweets (7,109 documents) on an accelerated clock (~10 minutes end to end). The dashboard updates every 3 seconds; watch Facebook (FB) around 25–26 Jul 2018, its record Q2 earnings crash.

| Service | Port | Role |
|---|---|---|
| `ingestor` | – | Replays the two sources into Redis Streams |
| `engine` | – | NLP risk engine: raw text → `RiskSignal` |
| `api` | 8000 | REST: `/signals`, `/stats`, `/health` |
| `dashboard` | 8501 | Streamlit live view |
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

## Repository structure

```
src/ripple/        Python package (ingestion, engine, api, stress_test, rebalancer, dashboard)
data/              Sample data (committed), raw + processed (generated); sources in data/README.md
docs/              Architecture and presentation
notebooks/         Colab notebooks for model training and calibration
scripts/           Data download and utility scripts
models/            Exported model artifacts (generated)
tests/             Unit tests
```

## Data sources

All data is public or synthetic. See [`data/README.md`](data/README.md) for sources and citations.

## License

[MIT](LICENSE)
