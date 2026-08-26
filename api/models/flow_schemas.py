"""
Pydantic models for the flow-based prediction endpoints.

Sits alongside :mod:`api.models.schemas`, which describes *log events*. The two
ontologies stay apart on purpose: a network flow has 78 CICFlowMeter features
and the taxonomy ``BENIGN|Botnet|BruteForce|DDoS|DoS|Scan|WebAttack``, while an
event has 17 features and the taxonomy ``normal|brute_force|…``. Merging them
would give one endpoint that accepts payloads no model can score.

The wire shape puts every model input inside a ``features`` object and every
piece of context — source and destination addresses, port, protocol — outside
it. That separation is structural rather than a denylist: a metadata field added
later cannot accidentally become a feature, because the extractor never looks
outside ``features`` at all.
"""

import ipaddress
from datetime import datetime
from typing import Final

from pydantic import BaseModel, Field, field_validator

from ml.flow.features import FLOW_FEATURE_ALIASES, FLOW_FEATURE_NAMES

# Every spelling the extractor will act on. A payload sharing none of these
# names produces an all-zero vector, which the models score perfectly happily —
# so it is rejected at the edge instead.
_ACCEPTED_FEATURE_NAMES: Final[frozenset[str]] = frozenset(FLOW_FEATURE_NAMES) | frozenset(
    FLOW_FEATURE_ALIASES
)

# Same ceiling as the event batch endpoint.
MAX_BATCH_FLOWS: Final[int] = 100


# ============================================================================
# Request models
# ============================================================================


class NetworkFlow(BaseModel):
    """A network flow to be analyzed."""

    features: dict[str, float] = Field(
        ...,
        min_length=1,
        description=(
            "CICFlowMeter feature name to value. Order is irrelevant — the extractor "
            "reorders by name. Unknown names are ignored and absent features default "
            "to 0.0, so a partial flow is accepted."
        ),
    )
    src_ip: str | None = Field(None, description="Source IP address (context only, not a feature)")
    dst_ip: str | None = Field(
        None, description="Destination IP address (context only, not a feature)"
    )
    dst_port: int | None = Field(
        None, ge=0, le=65535, description="Destination port (context only, not a feature)"
    )
    protocol: str | None = Field(None, max_length=16, description="Transport protocol, e.g. TCP")

    @field_validator("features")
    @classmethod
    def validate_known_features(cls, v: dict[str, float]) -> dict[str, float]:
        """
        Require at least one recognised CICFlowMeter feature name.

        Without this, a client that misspells every key — or sends an entirely
        different schema — gets an all-zero vector scored as a real flow, and a
        confident answer about traffic the model never saw. That failure is
        invisible from the response, so it is caught here.
        """
        if not _ACCEPTED_FEATURE_NAMES.intersection(v):
            raise ValueError(
                "None of the supplied feature names is a known CICFlowMeter feature. "
                f"Expected names such as {', '.join(sorted(FLOW_FEATURE_NAMES)[:3])}."
            )
        return v

    @field_validator("src_ip", "dst_ip")
    @classmethod
    def validate_ip(cls, v: str | None) -> str | None:
        if v is None:
            return None
        try:
            ipaddress.ip_address(v)
        except ValueError:
            raise ValueError("Invalid IP address")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "features": {
                    "Destination Port": 80,
                    "Flow Duration": 1532000,
                    "Total Fwd Packets": 12,
                    "Total Backward Packets": 10,
                    "Flow Bytes/s": 8421.5,
                    "Flow Packets/s": 14.3,
                    "SYN Flag Count": 1,
                    "ACK Flag Count": 20,
                    "Min Packet Length": 0,
                    "Max Packet Length": 1460,
                },
                "src_ip": "192.168.10.5",
                "dst_ip": "10.0.1.20",
                "dst_port": 80,
                "protocol": "TCP",
            }
        }
    }


class BatchFlowRequest(BaseModel):
    """Batch flow prediction request."""

    flows: list[NetworkFlow] = Field(..., min_length=1, max_length=MAX_BATCH_FLOWS)


# ============================================================================
# Response models
# ============================================================================


class FlowPredictionResult(BaseModel):
    """Result of analyzing a network flow."""

    flow_id: str = Field(..., description="Unique identifier for this prediction")
    threat_score: float = Field(..., ge=0, le=1, description="Threat score (0-1)")
    severity: str = Field(..., description="Severity level: LOW, MEDIUM, HIGH, CRITICAL")
    attack_type: str = Field(
        ...,
        description=(
            "Detected attack family: BENIGN, Botnet, BruteForce, DDoS, DoS, Scan, "
            "WebAttack, or unknown_anomaly when Stage 1 flags a flow Stage 2 cannot name"
        ),
    )
    confidence: float = Field(..., ge=0, le=1, description="Confidence in the assessment")
    is_anomaly: bool = Field(..., description="Whether Stage 1 flagged the flow as anomalous")
    explanation: list[str] = Field(default_factory=list, description="Human-readable explanations")
    recommended_actions: list[str] = Field(default_factory=list, description="Suggested responses")
    attack_probabilities: dict[str, float] = Field(
        default_factory=dict,
        description="Stage 2 probability per attack family; empty when Stage 1 stopped the flow",
    )
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    model_config = {
        "json_schema_extra": {
            "example": {
                "flow_id": "550e8400-e29b-41d4-a716-446655440000",
                "threat_score": 0.91,
                "severity": "CRITICAL",
                "attack_type": "DDoS",
                "confidence": 0.88,
                "is_anomaly": True,
                "explanation": [
                    "Flow deviates from benign traffic (anomaly score: 0.94)",
                    "Traffic pattern matches DDoS (97% confidence)",
                    "Very high throughput (42.0 MB/s)",
                ],
                "recommended_actions": [
                    "IMMEDIATE: Escalate to the security team",
                    "Engage upstream DDoS mitigation or scrubbing",
                    "Apply connection rate limits at the edge",
                ],
                "attack_probabilities": {"BENIGN": 0.01, "DDoS": 0.97, "DoS": 0.02},
                "timestamp": "2024-01-15T14:32:19.456Z",
            }
        }
    }


class BatchFlowPredictionResult(BaseModel):
    """Result of a batch flow prediction."""

    total: int = Field(..., description="Total flows submitted")
    processed: int = Field(..., description="Successfully processed flows")
    results: list[FlowPredictionResult]
