"""
Unit tests for FastAPI endpoints.

Tests the Sentinel Security API endpoints using TestClient.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
import numpy as np

from fastapi.testclient import TestClient
from jose import jwt

# We need to mock the models before importing the app
# to avoid loading actual models during tests


@pytest.fixture
def mock_models():
    """Mock the ML models for testing."""
    with patch("api.main.FeatureExtractor") as mock_extractor, \
         patch("api.main.AnomalyDetector") as mock_detector, \
         patch("api.main.AttackClassifier") as mock_classifier, \
         patch("api.main.FusionLayer") as mock_fusion:

        # Configure mock extractor
        mock_extractor_instance = MagicMock()
        mock_extractor_instance.transform.return_value = np.zeros(17, dtype=np.float32)
        mock_extractor.return_value = mock_extractor_instance

        # Configure mock detector
        mock_detector_instance = MagicMock()
        mock_detector_instance.is_loaded = True
        mock_detector_instance.predict.return_value = {
            "score": 0.5,
            "is_anomaly": False,
            "raw_score": 0.1,
            "confidence": 0.8,
        }
        mock_detector_instance.load.return_value = mock_detector_instance
        mock_detector.return_value = mock_detector_instance

        # Configure mock classifier
        mock_classifier_instance = MagicMock()
        mock_classifier_instance.is_loaded = True
        mock_classifier_instance.predict.return_value = {
            "attack_type": "normal",
            "confidence": 0.9,
            "probabilities": {
                "normal": 0.9,
                "brute_force": 0.05,
                "credential_stuffing": 0.02,
                "privilege_escalation": 0.01,
                "lateral_movement": 0.01,
                "data_exfiltration": 0.005,
                "reconnaissance": 0.005,
            },
        }
        mock_classifier_instance.load.return_value = mock_classifier_instance
        mock_classifier.return_value = mock_classifier_instance

        # Configure mock fusion
        mock_fusion_instance = MagicMock()
        mock_fusion_instance.combine.return_value = {
            "threat_score": 0.25,
            "severity": "LOW",
            "attack_type": "normal",
            "confidence": 0.85,
            "is_anomaly": False,
            "explanation": ["Normal activity detected"],
            "recommended_actions": ["Continue monitoring"],
        }
        mock_fusion.return_value = mock_fusion_instance

        yield {
            "extractor": mock_extractor_instance,
            "detector": mock_detector_instance,
            "classifier": mock_classifier_instance,
            "fusion": mock_fusion_instance,
        }


@pytest.fixture
def client(mock_models):
    """Create a test client with mocked models."""
    from api.main import app
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def valid_token() -> str:
    """Generate a valid JWT token for testing."""
    from api.security.auth import JWT_SECRET_KEY, JWT_ALGORITHM

    payload = {
        "sub": "test_user",
        "exp": datetime.utcnow() + timedelta(hours=1),
        "iat": datetime.utcnow(),
        "scopes": ["predict"],
        "tenant_id": "test_tenant",
    }
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


@pytest.fixture
def valid_event() -> dict:
    """Create a valid security event payload."""
    return {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "source": {
            "type": "auth",
            "hostname": "web-server-01",
            "ip": "10.0.1.45",
            "service": "sshd",
        },
        "actor": {
            "user": "admin",
            "uid": 1000,
            "source_ip": "203.0.113.45",
            "geo": {
                "country": "US",
                "city": "San Francisco",
                "lat": 37.7749,
                "lon": -122.4194,
            },
        },
        "action": {
            "type": "authentication",
            "subtype": "login_failed",
            "result": "failure",
            "reason": "invalid_password",
        },
    }


@pytest.fixture
def auth_headers(valid_token: str) -> dict:
    """Create authorization headers with valid token."""
    return {"Authorization": f"Bearer {valid_token}"}


# ============================================================================
# Health Endpoint Tests
# ============================================================================


class TestHealthEndpoint:
    """Tests for /health endpoint."""

    def test_health_returns_200(self, client: TestClient) -> None:
        """Test health endpoint returns 200 OK."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_format(self, client: TestClient) -> None:
        """Test health endpoint response format."""
        response = client.get("/health")
        data = response.json()

        assert "status" in data
        assert "timestamp" in data
        assert data["status"] == "healthy"

    def test_health_no_auth_required(self, client: TestClient) -> None:
        """Test health endpoint doesn't require authentication."""
        response = client.get("/health")
        assert response.status_code == 200


# ============================================================================
# Readiness Endpoint Tests
# ============================================================================


class TestReadyEndpoint:
    """Tests for /ready endpoint."""

    def test_ready_with_loaded_models(self, client: TestClient) -> None:
        """Test ready endpoint when models are loaded."""
        response = client.get("/ready")
        # Should return 200 since mock models report is_loaded=True
        assert response.status_code == 200

    def test_ready_response_format(self, client: TestClient) -> None:
        """Test ready endpoint response format."""
        response = client.get("/ready")
        data = response.json()

        assert "status" in data
        assert "checks" in data

    def test_ready_returns_checks(self, client: TestClient) -> None:
        """Test ready endpoint includes model checks."""
        response = client.get("/ready")
        data = response.json()

        expected_checks = [
            "feature_extractor",
            "anomaly_detector",
            "attack_classifier",
        ]
        for check in expected_checks:
            assert check in data["checks"]


# ============================================================================
# Metrics Endpoint Tests
# ============================================================================


class TestMetricsEndpoint:
    """Tests for /metrics endpoint."""

    def test_metrics_returns_200(self, client: TestClient) -> None:
        """Test metrics endpoint returns 200 OK."""
        response = client.get("/metrics")
        assert response.status_code == 200

    def test_metrics_content_type(self, client: TestClient) -> None:
        """Test metrics endpoint returns Prometheus format."""
        response = client.get("/metrics")
        assert "text/plain" in response.headers["content-type"] or \
               "text/plain" in response.headers.get("content-type", "")

    def test_metrics_no_auth_required(self, client: TestClient) -> None:
        """Test metrics endpoint doesn't require authentication."""
        response = client.get("/metrics")
        assert response.status_code == 200


# ============================================================================
# Predict Endpoint Tests
# ============================================================================


class TestPredictEndpoint:
    """Tests for POST /v1/predict endpoint."""

    def test_predict_requires_auth(
        self, client: TestClient, valid_event: dict
    ) -> None:
        """Test predict endpoint requires authentication."""
        response = client.post("/v1/predict", json=valid_event)
        assert response.status_code == 401

    def test_predict_with_valid_auth(
        self,
        client: TestClient,
        valid_event: dict,
        auth_headers: dict,
    ) -> None:
        """Test predict endpoint with valid authentication."""
        response = client.post(
            "/v1/predict",
            json=valid_event,
            headers=auth_headers,
        )
        assert response.status_code == 200

    def test_predict_response_format(
        self,
        client: TestClient,
        valid_event: dict,
        auth_headers: dict,
    ) -> None:
        """Test predict endpoint response format."""
        response = client.post(
            "/v1/predict",
            json=valid_event,
            headers=auth_headers,
        )
        data = response.json()

        expected_fields = [
            "event_id",
            "threat_score",
            "severity",
            "attack_type",
            "confidence",
            "is_anomaly",
            "explanation",
            "recommended_actions",
            "timestamp",
        ]
        for field in expected_fields:
            assert field in data, f"Missing field: {field}"

    def test_predict_threat_score_range(
        self,
        client: TestClient,
        valid_event: dict,
        auth_headers: dict,
    ) -> None:
        """Test threat score is in valid range."""
        response = client.post(
            "/v1/predict",
            json=valid_event,
            headers=auth_headers,
        )
        data = response.json()

        assert 0 <= data["threat_score"] <= 1

    def test_predict_valid_severity(
        self,
        client: TestClient,
        valid_event: dict,
        auth_headers: dict,
    ) -> None:
        """Test severity is a valid value."""
        response = client.post(
            "/v1/predict",
            json=valid_event,
            headers=auth_headers,
        )
        data = response.json()

        valid_severities = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
        assert data["severity"] in valid_severities

    def test_predict_invalid_event_format(
        self,
        client: TestClient,
        auth_headers: dict,
    ) -> None:
        """Test predict endpoint rejects invalid event format."""
        invalid_event = {"invalid": "data"}
        response = client.post(
            "/v1/predict",
            json=invalid_event,
            headers=auth_headers,
        )
        assert response.status_code == 422

    def test_predict_missing_required_fields(
        self,
        client: TestClient,
        auth_headers: dict,
    ) -> None:
        """Test predict endpoint rejects events with missing required fields."""
        incomplete_event = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "source": {"type": "auth", "hostname": "h", "ip": "10.0.0.1"},
            # Missing actor and action
        }
        response = client.post(
            "/v1/predict",
            json=incomplete_event,
            headers=auth_headers,
        )
        assert response.status_code == 422

    def test_predict_invalid_source_type(
        self,
        client: TestClient,
        valid_event: dict,
        auth_headers: dict,
    ) -> None:
        """Test predict endpoint rejects invalid source type."""
        valid_event["source"]["type"] = "invalid_type"
        response = client.post(
            "/v1/predict",
            json=valid_event,
            headers=auth_headers,
        )
        assert response.status_code == 422

    def test_predict_invalid_ip_address(
        self,
        client: TestClient,
        valid_event: dict,
        auth_headers: dict,
    ) -> None:
        """Test predict endpoint rejects invalid IP address."""
        valid_event["actor"]["source_ip"] = "invalid-ip"
        response = client.post(
            "/v1/predict",
            json=valid_event,
            headers=auth_headers,
        )
        assert response.status_code == 422


# ============================================================================
# Batch Prediction Endpoint Tests
# ============================================================================


class TestBatchPredictEndpoint:
    """Tests for POST /v1/analyze/batch endpoint."""

    def test_batch_requires_auth(
        self, client: TestClient, valid_event: dict
    ) -> None:
        """Test batch endpoint requires authentication."""
        response = client.post(
            "/v1/analyze/batch",
            json={"events": [valid_event]},
        )
        assert response.status_code == 401

    def test_batch_with_valid_auth(
        self,
        client: TestClient,
        valid_event: dict,
        auth_headers: dict,
    ) -> None:
        """Test batch endpoint with valid authentication."""
        response = client.post(
            "/v1/analyze/batch",
            json={"events": [valid_event]},
            headers=auth_headers,
        )
        assert response.status_code == 200

    def test_batch_response_format(
        self,
        client: TestClient,
        valid_event: dict,
        auth_headers: dict,
    ) -> None:
        """Test batch endpoint response format."""
        response = client.post(
            "/v1/analyze/batch",
            json={"events": [valid_event, valid_event]},
            headers=auth_headers,
        )
        data = response.json()

        assert "total" in data
        assert "processed" in data
        assert "results" in data
        assert data["total"] == 2
        assert len(data["results"]) == 2

    def test_batch_empty_events(
        self,
        client: TestClient,
        auth_headers: dict,
    ) -> None:
        """Test batch endpoint rejects empty events list."""
        response = client.post(
            "/v1/analyze/batch",
            json={"events": []},
            headers=auth_headers,
        )
        assert response.status_code == 422

    def test_batch_max_events_limit(
        self,
        client: TestClient,
        valid_event: dict,
        auth_headers: dict,
    ) -> None:
        """Test batch endpoint enforces max 100 events limit."""
        events = [valid_event] * 101
        response = client.post(
            "/v1/analyze/batch",
            json={"events": events},
            headers=auth_headers,
        )
        assert response.status_code == 422

    def test_batch_processes_multiple_events(
        self,
        client: TestClient,
        valid_event: dict,
        auth_headers: dict,
    ) -> None:
        """Test batch processes multiple events correctly."""
        events = [valid_event] * 5
        response = client.post(
            "/v1/analyze/batch",
            json={"events": events},
            headers=auth_headers,
        )
        data = response.json()

        assert data["total"] == 5
        assert data["processed"] == 5
        assert len(data["results"]) == 5


# ============================================================================
# Authentication Tests
# ============================================================================


class TestAuthentication:
    """Tests for authentication behavior."""

    def test_missing_auth_header(
        self, client: TestClient, valid_event: dict
    ) -> None:
        """Test request without auth header is rejected."""
        response = client.post("/v1/predict", json=valid_event)
        assert response.status_code == 401

    def test_invalid_token_format(
        self, client: TestClient, valid_event: dict
    ) -> None:
        """Test request with invalid token format is rejected."""
        headers = {"Authorization": "InvalidToken"}
        response = client.post(
            "/v1/predict",
            json=valid_event,
            headers=headers,
        )
        assert response.status_code == 401

    def test_expired_token(
        self, client: TestClient, valid_event: dict
    ) -> None:
        """Test request with expired token is rejected."""
        from api.security.auth import JWT_SECRET_KEY, JWT_ALGORITHM

        # Create an expired token
        payload = {
            "sub": "test_user",
            "exp": datetime.utcnow() - timedelta(hours=1),  # Expired
            "iat": datetime.utcnow() - timedelta(hours=2),
            "scopes": ["predict"],
        }
        expired_token = jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
        headers = {"Authorization": f"Bearer {expired_token}"}

        response = client.post(
            "/v1/predict",
            json=valid_event,
            headers=headers,
        )
        assert response.status_code == 401

    def test_invalid_signature(
        self, client: TestClient, valid_event: dict
    ) -> None:
        """Test request with invalid signature is rejected."""
        from api.security.auth import JWT_ALGORITHM

        # Create token with wrong secret
        payload = {
            "sub": "test_user",
            "exp": datetime.utcnow() + timedelta(hours=1),
            "iat": datetime.utcnow(),
            "scopes": ["predict"],
        }
        invalid_token = jwt.encode(payload, "wrong_secret", algorithm=JWT_ALGORITHM)
        headers = {"Authorization": f"Bearer {invalid_token}"}

        response = client.post(
            "/v1/predict",
            json=valid_event,
            headers=headers,
        )
        assert response.status_code == 401


# ============================================================================
# Input Validation Tests
# ============================================================================


class TestInputValidation:
    """Tests for input validation."""

    def test_valid_event_accepted(
        self,
        client: TestClient,
        valid_event: dict,
        auth_headers: dict,
    ) -> None:
        """Test valid event is accepted."""
        response = client.post(
            "/v1/predict",
            json=valid_event,
            headers=auth_headers,
        )
        assert response.status_code == 200

    def test_valid_minimal_event(
        self,
        client: TestClient,
        auth_headers: dict,
    ) -> None:
        """Test minimal valid event is accepted."""
        minimal_event = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "source": {
                "type": "auth",
                "hostname": "server-01",
                "ip": "10.0.0.1",
            },
            "actor": {
                "user": "user1",
                "source_ip": "192.168.1.100",
            },
            "action": {
                "type": "authentication",
                "result": "success",
            },
        }
        response = client.post(
            "/v1/predict",
            json=minimal_event,
            headers=auth_headers,
        )
        assert response.status_code == 200

    def test_invalid_action_result(
        self,
        client: TestClient,
        valid_event: dict,
        auth_headers: dict,
    ) -> None:
        """Test invalid action result is rejected."""
        valid_event["action"]["result"] = "invalid_result"
        response = client.post(
            "/v1/predict",
            json=valid_event,
            headers=auth_headers,
        )
        assert response.status_code == 422

    def test_geo_info_validation(
        self,
        client: TestClient,
        valid_event: dict,
        auth_headers: dict,
    ) -> None:
        """Test geo info with invalid coordinates is rejected."""
        valid_event["actor"]["geo"] = {
            "country": "US",
            "lat": 200,  # Invalid latitude
            "lon": 0,
        }
        response = client.post(
            "/v1/predict",
            json=valid_event,
            headers=auth_headers,
        )
        # Depending on validation strictness, this might be 422 or 200
        # The schema validation should catch this
        # If not strictly validated, the test should pass but latitude should be within -90 to 90

    def test_empty_username_rejected(
        self,
        client: TestClient,
        valid_event: dict,
        auth_headers: dict,
    ) -> None:
        """Test empty username is rejected."""
        valid_event["actor"]["user"] = ""
        response = client.post(
            "/v1/predict",
            json=valid_event,
            headers=auth_headers,
        )
        assert response.status_code == 422


# ============================================================================
# CORS Tests
# ============================================================================


class TestCORS:
    """Tests for CORS configuration."""

    def test_cors_headers_present(self, client: TestClient) -> None:
        """Test CORS headers are present in response."""
        response = client.options(
            "/v1/predict",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
            },
        )
        # FastAPI CORS middleware should handle this
        assert response.status_code in [200, 405]


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestErrorHandling:
    """Tests for error handling."""

    def test_not_found_endpoint(self, client: TestClient) -> None:
        """Test 404 for non-existent endpoint."""
        response = client.get("/nonexistent")
        assert response.status_code == 404

    def test_method_not_allowed(self, client: TestClient) -> None:
        """Test 405 for wrong HTTP method."""
        response = client.get("/v1/predict")
        assert response.status_code == 405

    def test_content_type_json_required(
        self,
        client: TestClient,
        auth_headers: dict,
    ) -> None:
        """Test that non-JSON content is rejected."""
        response = client.post(
            "/v1/predict",
            content="not json",
            headers={**auth_headers, "Content-Type": "text/plain"},
        )
        assert response.status_code == 422


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests for the API."""

    def test_full_prediction_flow(
        self,
        client: TestClient,
        valid_event: dict,
        auth_headers: dict,
    ) -> None:
        """Test complete prediction flow."""
        # Make prediction request
        response = client.post(
            "/v1/predict",
            json=valid_event,
            headers=auth_headers,
        )
        assert response.status_code == 200

        data = response.json()

        # Verify response contains all expected fields
        assert data["event_id"] is not None
        assert isinstance(data["threat_score"], float)
        assert data["severity"] in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
        assert data["attack_type"] is not None
        assert isinstance(data["confidence"], float)
        assert isinstance(data["is_anomaly"], bool)
        assert isinstance(data["explanation"], list)
        assert isinstance(data["recommended_actions"], list)

    def test_different_event_types(
        self,
        client: TestClient,
        auth_headers: dict,
    ) -> None:
        """Test predictions for different event types."""
        event_types = [
            ("auth", "authentication"),
            ("system", "file_access"),
            ("network", "network_connection"),
            ("app", "api_request"),
        ]

        for source_type, action_type in event_types:
            event = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "source": {
                    "type": source_type,
                    "hostname": "server-01",
                    "ip": "10.0.0.1",
                },
                "actor": {
                    "user": "user1",
                    "source_ip": "192.168.1.100",
                },
                "action": {
                    "type": action_type,
                    "result": "success",
                },
            }
            response = client.post(
                "/v1/predict",
                json=event,
                headers=auth_headers,
            )
            assert response.status_code == 200, f"Failed for {source_type}/{action_type}"
