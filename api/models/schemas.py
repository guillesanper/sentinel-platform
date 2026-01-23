"""
Pydantic models for API request/response schemas.
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field, field_validator
import ipaddress


# ============================================================================
# Request Models
# ============================================================================

class GeoInfo(BaseModel):
    """Geographic information about the source."""
    country: Optional[str] = None
    city: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None


class SourceInfo(BaseModel):
    """Information about the log source."""
    type: str = Field(..., description="Log type: auth, system, network, app")
    hostname: str = Field(..., min_length=1, max_length=255)
    ip: str = Field(..., description="Source system IP address")
    service: Optional[str] = Field(None, description="Service name (e.g., sshd)")
    
    @field_validator("type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        allowed = {"auth", "system", "network", "app", "application"}
        if v.lower() not in allowed:
            raise ValueError(f"type must be one of: {allowed}")
        return v.lower()
    
    @field_validator("ip")
    @classmethod
    def validate_ip(cls, v: str) -> str:
        try:
            ipaddress.ip_address(v)
        except ValueError:
            raise ValueError("Invalid IP address")
        return v


class ActorInfo(BaseModel):
    """Information about the actor (user/entity performing the action)."""
    user: str = Field(..., min_length=1, max_length=128)
    uid: Optional[int] = Field(None, ge=0)
    session_id: Optional[str] = None
    source_ip: str = Field(..., description="Client IP address")
    geo: Optional[GeoInfo] = None
    
    @field_validator("source_ip")
    @classmethod
    def validate_source_ip(cls, v: str) -> str:
        try:
            ipaddress.ip_address(v)
        except ValueError:
            raise ValueError("Invalid source IP address")
        return v


class ActionInfo(BaseModel):
    """Information about the action performed."""
    type: str = Field(..., description="Action type (e.g., authentication)")
    subtype: Optional[str] = Field(None, description="Action subtype (e.g., login_failed)")
    target: Optional[str] = Field(None, description="Target resource")
    method: Optional[str] = Field(None, description="Method used")
    result: str = Field(..., description="Result: success or failure")
    reason: Optional[str] = Field(None, description="Reason for failure")
    
    @field_validator("result")
    @classmethod
    def validate_result(cls, v: str) -> str:
        allowed = {"success", "failure", "unknown"}
        if v.lower() not in allowed:
            raise ValueError(f"result must be one of: {allowed}")
        return v.lower()


class ContextInfo(BaseModel):
    """Additional context about the event."""
    process: Optional[str] = None
    pid: Optional[int] = Field(None, ge=0)
    parent_pid: Optional[int] = Field(None, ge=0)
    command: Optional[str] = None
    tty: Optional[str] = None


class SecurityEvent(BaseModel):
    """A security event to be analyzed."""
    timestamp: datetime = Field(..., description="Event timestamp (ISO 8601)")
    source: SourceInfo
    actor: ActorInfo
    action: ActionInfo
    context: Optional[ContextInfo] = None
    raw: Optional[str] = Field(None, description="Raw log line", max_length=10000)
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "timestamp": "2024-01-15T14:32:18.123Z",
                "source": {
                    "type": "auth",
                    "hostname": "web-server-01",
                    "ip": "10.0.1.45",
                    "service": "sshd"
                },
                "actor": {
                    "user": "admin",
                    "source_ip": "203.0.113.45"
                },
                "action": {
                    "type": "authentication",
                    "subtype": "login_failed",
                    "result": "failure",
                    "reason": "invalid_password"
                }
            }
        }
    }


class BatchPredictionRequest(BaseModel):
    """Batch prediction request."""
    events: list[SecurityEvent] = Field(..., min_length=1, max_length=100)


# ============================================================================
# Response Models
# ============================================================================

class PredictionResult(BaseModel):
    """Result of analyzing a security event."""
    event_id: str = Field(..., description="Unique identifier for this prediction")
    threat_score: float = Field(..., ge=0, le=1, description="Threat score (0-1)")
    severity: str = Field(..., description="Severity level: LOW, MEDIUM, HIGH, CRITICAL")
    attack_type: str = Field(..., description="Detected attack type")
    confidence: float = Field(..., ge=0, le=1, description="Confidence in classification")
    is_anomaly: bool = Field(..., description="Whether this is anomalous behavior")
    explanation: list[str] = Field(default_factory=list, description="Human-readable explanations")
    recommended_actions: list[str] = Field(default_factory=list, description="Suggested actions")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "event_id": "550e8400-e29b-41d4-a716-446655440000",
                "threat_score": 0.87,
                "severity": "HIGH",
                "attack_type": "brute_force",
                "confidence": 0.92,
                "is_anomaly": True,
                "explanation": [
                    "47 failed logins from this IP in last hour",
                    "Source IP from high-risk region",
                    "Activity outside business hours"
                ],
                "recommended_actions": [
                    "Block source IP temporarily",
                    "Enable MFA for target user",
                    "Review authentication logs"
                ],
                "timestamp": "2024-01-15T14:32:19.456Z"
            }
        }
    }


class BatchPredictionResult(BaseModel):
    """Result of batch prediction."""
    total: int = Field(..., description="Total events submitted")
    processed: int = Field(..., description="Successfully processed events")
    results: list[PredictionResult]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "healthy"
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ReadyResponse(BaseModel):
    """Readiness check response."""
    status: str = "ready"
    checks: dict[str, bool] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    """Error response."""
    detail: str
    error_code: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)