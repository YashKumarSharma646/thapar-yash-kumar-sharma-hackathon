# syntax=docker/dockerfile:1
# Targets:
#   base   -> lightweight services (api, dashboard, ingestor)
#   engine -> adds CPU PyTorch + Transformers and bakes the FinBERT weights into the image,
#             so the engine starts offline and nothing is downloaded at run time.
# Dependencies are installed before source is copied, so code edits don't invalidate heavy layers.
# A pip cache mount keeps finished downloads across build retries.

FROM python:3.11-slim AS deps
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    HF_HOME=/opt/hf \
    PIP_DEFAULT_TIMEOUT=300 \
    PIP_RETRIES=10
WORKDIR /app
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt


FROM deps AS engine-deps
COPY requirements-engine.txt .
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements-engine.txt
# Retries + long timeout: the weights are ~440 MB and slow links time out (downloads resume).
RUN for i in 1 2 3 4 5; do \
      HF_HUB_DOWNLOAD_TIMEOUT=120 python -c "from huggingface_hub import snapshot_download; \
        snapshot_download('ProsusAI/finbert', allow_patterns=['*.json', '*.txt', 'pytorch_model.bin'])" && break; \
      echo "FinBERT download attempt $i failed, retrying..."; sleep 5; \
    done && \
    python -c "from transformers import AutoModelForSequenceClassification as M; M.from_pretrained('ProsusAI/finbert')"
ENV HF_HUB_OFFLINE=1


FROM deps AS base
COPY .streamlit/ .streamlit/
COPY src/ src/
COPY data/ data/
COPY models/ models/


FROM engine-deps AS engine
COPY src/ src/
COPY data/ data/
COPY models/ models/
