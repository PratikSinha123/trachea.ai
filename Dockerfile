FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000 \
    TRACHEA_OUTPUT=/app/processed_data

WORKDIR /app

COPY requirements.txt .
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
        libgomp1 \
    && python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt \
    && apt-get purge -y --auto-remove build-essential git \
    && rm -rf /var/lib/apt/lists/*

COPY server ./server
COPY segmentation ./segmentation
COPY reconstruction ./reconstruction
COPY visualization ./visualization
COPY training ./training
COPY frontend ./frontend

RUN mkdir -p /app/processed_data /data \
    && useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app /data

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${PORT}/api/scans" >/dev/null || exit 1

CMD ["sh", "-c", "uvicorn server.app:app --host 0.0.0.0 --port ${PORT}"]
