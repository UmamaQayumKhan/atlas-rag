# Multi-stage so the runtime image carries no build toolchain.
FROM python:3.12-slim AS builder

WORKDIR /build
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1

COPY requirements.txt .
# Streamlit is a dev/UI dependency and has no business in the API image.
RUN grep -v "^streamlit" requirements.txt > requirements.api.txt \
 && pip install --prefix=/install -r requirements.api.txt


FROM python:3.12-slim AS runtime

# Run unprivileged. A container that can write to its own source tree is an
# avoidable finding in any security review.
RUN useradd --create-home --uid 10001 atlas
WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    PATH="/usr/local/bin:$PATH"

COPY --from=builder /install /usr/local
COPY src/ ./src/
COPY eval/ ./eval/
COPY data/ ./data/

RUN mkdir -p /app/.atlas_index && chown -R atlas:atlas /app
USER atlas

EXPOSE 8000

# Hits the real readiness endpoint, so an unindexed container reports degraded
# instead of passing a port check and serving empty answers.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,sys,json; \
b=json.load(urllib.request.urlopen('http://localhost:8000/healthz',timeout=4)); \
sys.exit(0 if b.get('status')=='ok' else 1)"

CMD ["uvicorn", "atlas.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
