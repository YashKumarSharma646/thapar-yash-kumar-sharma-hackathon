# syntax=docker/dockerfile:1
# Targets:
#   base   -> lightweight services (api, dashboard, ingestor)
#   engine -> adds ONNX Runtime for the distilled NLP model committed in models/ripple-nlp/
#             (no PyTorch, nothing downloaded at run time).
# Dependencies are installed before source is copied, so code edits don't invalidate heavy layers.
# A pip cache mount keeps finished downloads across build retries.

FROM python:3.11-slim AS deps
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    PIP_DEFAULT_TIMEOUT=300 \
    PIP_RETRIES=10
WORKDIR /app
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt


FROM deps AS engine-deps
COPY requirements-engine.txt .
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements-engine.txt


FROM deps AS base
COPY .streamlit/ .streamlit/
COPY src/ src/
COPY data/ data/
COPY models/ models/


FROM engine-deps AS engine
COPY src/ src/
COPY data/ data/
COPY models/ models/
