"""
Sentinel Security API
=====================
Cloud-Native Intrusion Detection API powered by Machine Learning.

Two detection tracks are served side by side over different data modalities.
They share no feature space and no taxonomy, so they have separate models,
separate schemas and separate endpoints:

- **events** — log records, 17 features, ``normal|brute_force|…``
- **flows** — CICFlowMeter network flows, 78 features,
  ``BENIGN|Botnet|BruteForce|DDoS|DoS|Scan|WebAttack``

The tracks also degrade independently. If the flow models are absent, the flow
endpoints answer 503 while the event endpoints keep serving, and vice versa.

Endpoints:
- POST /v1/predict - Analyze single security event
- POST /v1/analyze/batch - Analyze batch of events
- POST /v1/predict/flow - Analyze single network flow
- POST /v1/analyze/flow/batch - Analyze batch of network flows
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
from api.models.flow_schemas import (
    BatchFlowPredictionResult,
    BatchFlowRequest,
    FlowPredictionResult,
    NetworkFlow,
)
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
from ml.flow.serving import CascadeInferenceService, FeatureContractError
from ml.models.anomaly_detector import AnomalyDetector
from ml.models.classifier import AttackClassifier
from ml.models.fusion import FusionLayer

# Value of the `track` metric label, one per data modality.
TRACK_EVENT = "event"
TRACK_FLOW = "flow"

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

# Both collectors carry a `track` label rather than being duplicated per track:
# a second set of collectors would mean every dashboard panel and alert rule has
# to be written twice and kept in step. Existing queries wrap these in `sum(…)`,
# which aggregates the new label away, so they keep working unchanged.
PREDICTION_SCORE = Histogram(
    "sentinel_prediction_threat_score",
    "Distribution of threat scores",
    ["track"],
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

ATTACK_TYPE_COUNT = Counter(
    "sentinel_attack_type_total",
    "Predictions by attack type",
    ["attack_type", "track"],
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
    app.state.flow_service = CascadeInferenceService()

    # Load trained models (if available)
    try:
        app.state.anomaly_detector.load()
        app.state.attack_classifier.load()
        logger.info("Models loaded successfully")
    except FileNotFoundError:
        logger.warning("No trained models found - using untrained models")

    # The flow track loads independently, and its failures must not take the
    # event track down with it.
    try:
        app.state.flow_service.load()
        logger.info("flow_models_loaded")
    except FileNotFoundError:
        logger.warning("flow_models_not_found", hint="flow endpoints will answer 503")
    except FeatureContractError as exc:
        # A stale artifact, not a missing one. Logged at error level because it
        # needs a retrain — but it still must not prevent the event endpoints
        # from serving, so startup continues with the flow track degraded.
        logger.error("flow_models_rejected", error=str(exc))

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
    """
    Readiness check - is the service ready to receive traffic?

    The event track gates readiness; the flow track is reported but does not.
    That asymmetry is deliberate and load-bearing: making a missing flow model
    fail the probe would have Kubernetes pull the pod out of service and take
    the event endpoints down with it, for a track that is optional. So a
    degraded flow track shows up as ``flow_detector: false`` in the checks and
    as a 503 on the flow routes only.
    """
    feature_extractor = getattr(request.app.state, "feature_extractor", None)
    anomaly_detector = getattr(request.app.state, "anomaly_detector", None)
    attack_classifier = getattr(request.app.state, "attack_classifier", None)

    # These decide the status code.
    event_checks = {
        "feature_extractor": feature_extractor is not None,
        "anomaly_detector": anomaly_detector is not None and anomaly_detector.is_loaded,
        "attack_classifier": attack_classifier is not None and attack_classifier.is_loaded,
    }

    # These are informational. Both stages come from the same cascade artifact,
    # so they move together — they are reported separately because that is what
    # an operator reads to tell which half of the cascade is missing when the
    # loader ever gains the ability to load them apart.
    flow_ready = _flow_service_ready(request)
    flow_checks = {
        "flow_detector": flow_ready,
        "flow_classifier": flow_ready,
    }

    checks = {**event_checks, **flow_checks}

    if not all(event_checks.values()):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"status": "not_ready", "checks": checks},
        )

    return ReadyResponse(status="ready", checks=checks)


def _flow_service_ready(request: Request) -> bool:
    """Whether a validated flow cascade is loaded and able to answer."""
    flow_service = getattr(request.app.state, "flow_service", None)
    return flow_service is not None and bool(flow_service.is_loaded)


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
        PREDICTION_SCORE.labels(track=TRACK_EVENT).observe(result.threat_score)
        ATTACK_TYPE_COUNT.labels(attack_type=result.attack_type, track=TRACK_EVENT).inc()

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
# Flow Prediction Endpoints
# ============================================================================
@app.post("/v1/predict/flow", response_model=FlowPredictionResult, tags=["Prediction"])
@limiter.limit("100/minute")
async def predict_flow(
    request: Request,
    flow: NetworkFlow,
    token_data: TokenData = Depends(verify_token),
) -> FlowPredictionResult:
    """
    Analyze a single network flow and return a threat assessment.

    This endpoint:
    1. Reorders the CICFlowMeter features by name into the model's column order
    2. Runs Stage 1 (Isolation Forest) to decide whether the flow is anomalous
    3. Routes anomalous flows to Stage 2 (XGBoost) to name the attack family
    4. Combines both signals with the flow fusion layer

    Returns 503 when no flow cascade is loaded. That is independent of the event
    endpoints, which keep serving.
    """
    service = _require_flow_service(request)
    flow_id = str(uuid.uuid4())

    logger.info(
        "flow_prediction_request",
        flow_id=flow_id,
        src_ip=flow.src_ip,
        dst_ip=flow.dst_ip,
        dst_port=flow.dst_port,
        protocol=flow.protocol,
        n_features=len(flow.features),
    )

    try:
        assessment = service.predict(flow.model_dump())
        result = _to_flow_result(flow_id, assessment)
    except Exception as e:
        logger.error("flow_prediction_failed", flow_id=flow_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Flow prediction failed",
        )

    _record_flow_metrics(result)
    logger.info(
        "flow_prediction_complete",
        flow_id=flow_id,
        threat_score=result.threat_score,
        severity=result.severity,
        attack_type=result.attack_type,
    )
    return result


@app.post(
    "/v1/analyze/flow/batch",
    response_model=BatchFlowPredictionResult,
    tags=["Prediction"],
)
@limiter.limit("10/minute")
async def analyze_flow_batch(
    request: Request,
    batch: BatchFlowRequest,
    token_data: TokenData = Depends(verify_token),
) -> BatchFlowPredictionResult:
    """
    Analyze a batch of network flows (max 100).

    Unlike the event batch endpoint, which loops over single predictions, this
    hands the whole batch to ``predict_batch``: Stage 1 scores every flow in one
    pass and Stage 2 sees a single call covering all the routed ones. On a
    hundred flows that is two model invocations rather than two hundred.
    """
    service = _require_flow_service(request)

    logger.info("flow_batch_prediction_request", count=len(batch.flows))

    try:
        assessments = service.predict_batch([flow.model_dump() for flow in batch.flows])
    except Exception as e:
        # No partial results to salvage: the batch is one vectorised call, so it
        # either produced every assessment or none.
        logger.error("flow_batch_prediction_failed", count=len(batch.flows), error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Flow batch prediction failed",
        )

    results = [_to_flow_result(str(uuid.uuid4()), assessment) for assessment in assessments]
    for result in results:
        _record_flow_metrics(result)

    return BatchFlowPredictionResult(
        total=len(batch.flows),
        processed=len(results),
        results=results,
    )


def _require_flow_service(request: Request) -> CascadeInferenceService:
    """
    Return the loaded flow service, or refuse the request.

    Raises:
        HTTPException: 503, when no validated cascade is loaded. Only the flow
            routes are affected — the event routes never consult this.
    """
    if not _flow_service_ready(request):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Flow models are not available. Train a cascade with "
                "ml.training.train_cicids and mount it at models/flow/cascade_xgb."
            ),
        )
    service: CascadeInferenceService = request.app.state.flow_service
    return service


def _to_flow_result(flow_id: str, assessment: dict) -> FlowPredictionResult:
    """Shape a fusion assessment into the response model."""
    return FlowPredictionResult(
        flow_id=flow_id,
        threat_score=assessment["threat_score"],
        severity=assessment["severity"],
        attack_type=assessment["attack_type"],
        confidence=assessment["confidence"],
        is_anomaly=assessment["is_anomaly"],
        explanation=assessment["explanation"],
        recommended_actions=assessment["recommended_actions"],
        attack_probabilities=assessment.get("details", {}).get("class_probabilities", {}),
        timestamp=datetime.utcnow(),
    )


def _record_flow_metrics(result: FlowPredictionResult) -> None:
    """Record one flow prediction against the shared collectors."""
    PREDICTION_SCORE.labels(track=TRACK_FLOW).observe(result.threat_score)
    ATTACK_TYPE_COUNT.labels(attack_type=result.attack_type, track=TRACK_FLOW).inc()


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
