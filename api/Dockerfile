# ============================================================================
# Sentinel Platform API - Multi-stage Docker Build
# ============================================================================
#
# Build: docker build -t sentinel-api:latest -f api/Dockerfile .
# Run:   docker run -p 8000:8000 sentinel-api:latest
#
# Security features:
# - Multi-stage build to minimize image size
# - Non-root user for runtime
# - No shell in final image (optional hardening)
# - Health check configured
# ============================================================================

# ----------------------------------------------------------------------------
# Stage 1: Builder
# ----------------------------------------------------------------------------
FROM python:3.11-slim as builder

# Set build-time environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy and install requirements
WORKDIR /build
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# ----------------------------------------------------------------------------
# Stage 2: Runtime
# ----------------------------------------------------------------------------
FROM python:3.11-slim as runtime

# Metadata labels
LABEL maintainer="Sentinel Platform Team" \
      version="1.0.0" \
      description="Sentinel Intrusion Detection API" \
      org.opencontainers.image.source="https://github.com/sentinel-platform/sentinel"

# Runtime environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    # Application settings
    APP_ENV=production \
    APP_HOST=0.0.0.0 \
    APP_PORT=8000 \
    APP_WORKERS=4 \
    # Paths
    APP_HOME=/app \
    MODELS_PATH=/app/models \
    # Python
    PATH="/opt/venv/bin:$PATH"

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd --gid 1000 sentinel && \
    useradd --uid 1000 --gid sentinel --shell /bin/bash --create-home sentinel

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Set working directory
WORKDIR ${APP_HOME}

# Copy application code
# Note: Order matters for Docker layer caching
COPY --chown=sentinel:sentinel ml/ ./ml/
COPY --chown=sentinel:sentinel api/ ./api/
COPY --chown=sentinel:sentinel data/ ./data/

# Create models directory (models will be mounted or copied separately)
RUN mkdir -p ${MODELS_PATH} && \
    chown -R sentinel:sentinel ${MODELS_PATH}

# Switch to non-root user
USER sentinel

# Expose API port
EXPOSE 8000

# Health check
# Checks /health endpoint every 30 seconds, times out after 10 seconds
# Considers unhealthy after 3 consecutive failures
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl --fail http://localhost:${APP_PORT}/health || exit 1

# Default command - run the API server
# Using exec form for proper signal handling
CMD ["sh", "-c", "uvicorn api.main:app --host ${APP_HOST} --port ${APP_PORT} --workers ${APP_WORKERS}"]

# ----------------------------------------------------------------------------
# Alternative: Production with Gunicorn
# ----------------------------------------------------------------------------
# For production deployments with more control over workers and timeouts:
#
# CMD ["gunicorn", "api.main:app", \
#      "--bind", "0.0.0.0:8000", \
#      "--workers", "4", \
#      "--worker-class", "uvicorn.workers.UvicornWorker", \
#      "--timeout", "120", \
#      "--keep-alive", "5", \
#      "--max-requests", "10000", \
#      "--max-requests-jitter", "1000", \
#      "--access-logfile", "-", \
#      "--error-logfile", "-"]
