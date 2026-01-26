"""
Sentinel Security API
=====================
Cloud-Native Intrusion Detection API powered by Machine Learning.

Endpoints:
- POST /v1/predict - Analyze single security event
- POST /v1/analyze/batch - Analyze batch of events
- GET /health - Health check
- GET /ready - Readiness check
- GET /metrics - Prometheus metrics
"""

import time
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime

import structlog
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from api.middleware.logging import setup_logging
from api.models.schemas import (
    BatchPredictionRequest,
    BatchPredictionResult,
    HealthResponse,
    PredictionResult,
    ReadyResponse,
    SecurityEvent,
)
from api.security.auth import TokenData, verify_token
from ml.features.pipeline import FeatureExtractor
from ml.models.anomaly_detector import AnomalyDetector
from ml.models.classifier import AttackClassifier
from ml.models.fusion import FusionLayer

# ============================================================================
# Logging Setup
# ============================================================================
setup_logging()
logger = structlog.get_logger()

# ============================================================================
# Prometheus Metrics
# ============================================================================
REQUEST_COUNT = Counter(
    "sentinel_api_requests_total",
    "Total API requests",
    ["endpoint", "method", "status"],
)

REQUEST_LATENCY = Histogram(
    "sentinel_api_request_latency_seconds",
    "Request latency in seconds",
    ["endpoint"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)

PREDICTION_SCORE = Histogram(
    "sentinel_prediction_threat_score",
    "Distribution of threat scores",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

ATTACK_TYPE_COUNT = Counter(
    "sentinel_attack_type_total",
    "Predictions by attack type",
    ["attack_type"],
)

# ============================================================================
# Rate Limiter
# ============================================================================
limiter = Limiter(key_func=get_remote_address)

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan - load models on startup."""
    logger.info("Starting Sentinel API...")

    # Initialize components and store in app.state
    app.state.feature_extractor = FeatureExtractor()
    app.state.anomaly_detector = AnomalyDetector()
    app.state.attack_classifier = AttackClassifier()
    app.state.fusion_layer = FusionLayer()

    # Load trained models (if available)
    try:
        app.state.anomaly_detector.load()
        app.state.attack_classifier.load()
        logger.info("Models loaded successfully")
    except FileNotFoundError:
        logger.warning("No trained models found - using untrained models")

    yield

    logger.info("Shutting down Sentinel API...")


# ============================================================================
# FastAPI Application
# ============================================================================
app = FastAPI(
    title="Sentinel Security API",
    description="Cloud-Native Intrusion Detection API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add rate limiter
app.state.limiter = limiter


# Rate limit error handler
@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded) -> Response:
    return Response(
        content='{"detail": "Rate limit exceeded"}',
        status_code=429,
        media_type="application/json",
    )


# ============================================================================
# Middleware
# ============================================================================
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """Record request metrics."""
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time

    # Record metrics
    REQUEST_LATENCY.labels(endpoint=request.url.path).observe(duration)
    REQUEST_COUNT.labels(
        endpoint=request.url.path,
        method=request.method,
        status=response.status_code,
    ).inc()

    return response


# ============================================================================
# Health & Readiness Endpoints
# ============================================================================
@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health() -> HealthResponse:
    """Basic health check - is the service running?"""
    return HealthResponse(status="healthy", timestamp=datetime.utcnow())


@app.get("/ready", response_model=ReadyResponse, tags=["Health"])
async def ready(request: Request) -> ReadyResponse:
    """Readiness check - is the service ready to receive traffic?"""
    feature_extractor = getattr(request.app.state, "feature_extractor", None)
    anomaly_detector = getattr(request.app.state, "anomaly_detector", None)
    attack_classifier = getattr(request.app.state, "attack_classifier", None)
    checks = {
        "feature_extractor": feature_extractor is not None,
        "anomaly_detector": anomaly_detector is not None and anomaly_detector.is_loaded,
        "attack_classifier": attack_classifier is not None and attack_classifier.is_loaded,
    }

    all_ready = all(checks.values())

    if not all_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"status": "not_ready", "checks": checks},
        )

    return ReadyResponse(status="ready", checks=checks)


@app.get("/metrics", tags=["Observability"])
async def metrics() -> Response:
    """Prometheus metrics endpoint."""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


# ============================================================================
# Prediction Endpoints
# ============================================================================
@app.post("/v1/predict", response_model=PredictionResult, tags=["Prediction"])
@limiter.limit("100/minute")
async def predict(
    request: Request,
    event: SecurityEvent,
    token_data: TokenData = Depends(verify_token),
) -> PredictionResult:
    """
    Analyze a single security event and return threat assessment.

    This endpoint:
    1. Extracts features from the event
    2. Runs anomaly detection
    3. Classifies attack type
    4. Combines results with fusion layer
    """
    event_id = str(uuid.uuid4())

    logger.info(
        "prediction_request",
        event_id=event_id,
        event_type=event.source.type,
        user=event.actor.user,
        source_ip=event.actor.source_ip,
    )

    try:
        # 1. Extract features
        features = request.app.state.feature_extractor.transform(event.model_dump())

        # 2. Run models
        anomaly_result = request.app.state.anomaly_detector.predict(features)
        classification_result = request.app.state.attack_classifier.predict(features)

        # 3. Combine with fusion layer
        fusion_result = request.app.state.fusion_layer.combine(
            anomaly_result=anomaly_result,
            classification_result=classification_result,
            event=event.model_dump(),
        )

        # 4. Build response
        result = PredictionResult(
            event_id=event_id,
            threat_score=fusion_result["threat_score"],
            severity=fusion_result["severity"],
            attack_type=fusion_result["attack_type"],
            confidence=fusion_result["confidence"],
            is_anomaly=fusion_result["is_anomaly"],
            explanation=fusion_result["explanation"],
            recommended_actions=fusion_result["recommended_actions"],
            timestamp=datetime.utcnow(),
        )

        # 5. Record metrics
        PREDICTION_SCORE.observe(result.threat_score)
        ATTACK_TYPE_COUNT.labels(attack_type=result.attack_type).inc()

        logger.info(
            "prediction_complete",
            event_id=event_id,
            threat_score=result.threat_score,
            severity=result.severity,
            attack_type=result.attack_type,
        )

        return result

    except Exception as e:
        logger.error("prediction_failed", event_id=event_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction failed",
        )


@app.post("/v1/analyze/batch", response_model=BatchPredictionResult, tags=["Prediction"])
@limiter.limit("10/minute")
async def analyze_batch(
    request: Request,
    batch: BatchPredictionRequest,
    token_data: TokenData = Depends(verify_token),
) -> BatchPredictionResult:
    """
    Analyze a batch of security events (max 100).

    Useful for bulk processing of historical logs.
    """
    if len(batch.events) > 100:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Maximum 100 events per batch",
        )

    logger.info("batch_prediction_request", count=len(batch.events))

    results = []
    for event in batch.events:
        try:
            result = await predict(request, event, token_data)
            results.append(result)
        except HTTPException:
            # Skip failed predictions in batch mode
            logger.warning("batch_prediction_item_failed", event=event.model_dump())

    return BatchPredictionResult(
        total=len(batch.events),
        processed=len(results),
        results=results,
    )


# ============================================================================
# Error Handlers
# ============================================================================
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> Response:
    """Global exception handler."""
    logger.error("unhandled_exception", error=str(exc), path=request.url.path)
    return Response(
        content='{"detail": "Internal server error"}',
        status_code=500,
        media_type="application/json",
    )


# ============================================================================
# Main Entry Point
# ============================================================================
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
