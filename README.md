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

No API keys or downloads are needed. The default replay mode runs on the samples in `data/sample/`.

## Local development (without Docker)

Requires Python 3.11+ and a running Redis.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
pytest
```

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
