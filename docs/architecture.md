# Architecture

```mermaid
flowchart LR
    subgraph Sources
        N[News: Kaggle replay / GDELT live]
        S[Social: Kaggle stock tweets replay]
    end
    N --> I[Ingestor]
    S --> I
    I -- RawDocument --> R1[(Redis Stream<br/>raw_text)]
    R1 --> E[NLP Risk Engine<br/>sentiment · event class · impact<br/>entity linking · evidence]
    E -- RiskSignal --> R2[(Redis Stream<br/>signals)]
    E --> F[signals.jsonl]
    R2 --> API[FastAPI]
    R2 --> B[Module B<br/>Stress Testing]
    R2 -.stretch.-> A[Module A<br/>Index Rebalancer]
    API --> D[Streamlit Dashboard]
    B --> D
    A -.-> D
```

## Services (docker-compose)

| Service | Command | Status |
|---|---|---|
| redis | `redis:7-alpine` | ✅ |
| api | `uvicorn ripple.api.main:app` | ✅ skeleton |
| dashboard | `streamlit run src/ripple/dashboard/app.py` | ✅ skeleton |
| ingestor | `python -m ripple.ingestion.run` | Day 2–3 |
| engine | `python -m ripple.engine.worker` | Day 2 |
| stress_test | `python -m ripple.stress_test.worker` | Day 6–7 |

## Data contracts

See [`src/ripple/schemas.py`](../src/ripple/schemas.py): `RawDocument` (ingestor → engine) and `RiskSignal` (engine → consumers).
