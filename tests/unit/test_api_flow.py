"""
Unit tests for the flow-based prediction endpoints.

Follows the pattern of ``test_api.py``: patch in the ``api.main`` namespace and
drive the app through ``TestClient`` as a context manager, so the lifespan runs
and ``app.state`` is populated the way production does it.

The test that matters most is :class:`TestIndependentDegradation`. The two
tracks share a process, a readiness probe and a metrics registry, and it would
be easy to wire them so that a missing flow artifact takes the event endpoints
down — in production that is an outage caused by a model nobody was using yet.
So: flow models absent must mean 503 on the flow routes, 200 on the event
routes, and a readiness probe that still reports ready.
"""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from jose import jwt

from ml.flow.features import FLOW_FEATURE_NAMES

# ============================================================================
# Fixtures
# ============================================================================

_ASSESSMENT = {
    "threat_score": 0.82,
    "severity": "HIGH",
    "attack_type": "DDoS",
    "confidence": 0.77,
    "is_anomaly": True,
    "explanation": ["Flow deviates from benign traffic (anomaly score: 0.90)"],
    "recommended_actions": ["Engage upstream DDoS mitigation or scrubbing"],
    "details": {
        "anomaly_score": 0.9,
        "classification_threat": 0.95,
        "classification_confidence": 0.95,
        "rules_score": 0.1,
        "class_probabilities": {"BENIGN": 0.05, "DDoS": 0.95},
    },
}


def _patch_event_models():
    """Mocks for the event track, mirroring ``test_api.py``'s ``mock_models``."""
    extractor = MagicMock()
    extractor.transform.return_value = np.zeros(17, dtype=np.float32)

    detector = MagicMock()
    detector.is_loaded = True
    detector.predict.return_value = {"score": 0.5, "is_anomaly": False, "confidence": 0.8}

    classifier = MagicMock()
    classifier.is_loaded = True
    classifier.predict.return_value = {
        "attack_type": "normal",
        "confidence": 0.9,
        "probabilities": {"normal": 0.9},
    }

    fusion = MagicMock()
    fusion.combine.return_value = {
        "threat_score": 0.25,
        "severity": "LOW",
        "attack_type": "normal",
        "confidence": 0.85,
        "is_anomaly": False,
        "explanation": ["Normal activity detected"],
        "recommended_actions": ["Continue monitoring"],
    }

    return {
        "FeatureExtractor": extractor,
        "AnomalyDetector": detector,
        "AttackClassifier": classifier,
        "FusionLayer": fusion,
    }


def _make_client(flow_service: MagicMock) -> TestClient:
    """Build a TestClient with the event track mocked and a given flow service."""
    from api.main import app

    mocks = _patch_event_models()
    with (
        patch("api.main.FeatureExtractor", return_value=mocks["FeatureExtractor"]),
        patch("api.main.AnomalyDetector", return_value=mocks["AnomalyDetector"]),
        patch("api.main.AttackClassifier", return_value=mocks["AttackClassifier"]),
        patch("api.main.FusionLayer", return_value=mocks["FusionLayer"]),
        patch("api.main.CascadeInferenceService", return_value=flow_service),
        TestClient(app) as client,
    ):
        yield client


@pytest.fixture
def loaded_flow_service() -> MagicMock:
    """A flow service that loads cleanly and answers."""
    service = MagicMock()
    service.is_loaded = True
    service.load.return_value = service
    service.predict.return_value = dict(_ASSESSMENT)
    service.predict_batch.side_effect = lambda flows: [dict(_ASSESSMENT) for _ in flows]
    return service


@pytest.fixture
def missing_flow_service() -> MagicMock:
    """A flow service with no artifact on disk — the common production state."""
    service = MagicMock()
    service.is_loaded = False
    service.load.side_effect = FileNotFoundError("models/flow/cascade_xgb")
    return service


@pytest.fixture
def stale_flow_service() -> MagicMock:
    """
    A flow service that rejects its artifact: the columns on disk are not
    ``FLOW_FEATURE_NAMES`` any more, so serving it would be worse than serving
    nothing.
    """
    from ml.flow.serving import FeatureContractError

    service = MagicMock()
    service.is_loaded = False
    service.load.side_effect = FeatureContractError("trained on a different feature space")
    return service


@pytest.fixture
def client(loaded_flow_service: MagicMock):
    yield from _make_client(loaded_flow_service)


@pytest.fixture
def client_without_flow_models(missing_flow_service: MagicMock):
    yield from _make_client(missing_flow_service)


@pytest.fixture
def client_with_stale_flow_models(stale_flow_service: MagicMock):
    yield from _make_client(stale_flow_service)


@pytest.fixture
def auth_headers() -> dict:
    from api.security.auth import JWT_ALGORITHM, JWT_SECRET_KEY

    payload = {
        "sub": "test_user",
        "exp": datetime.utcnow() + timedelta(hours=1),
        "iat": datetime.utcnow(),
        "scopes": ["predict"],
        "tenant_id": "test_tenant",
    }
    token = jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def valid_flow() -> dict:
    """A realistic flow payload: real feature names, plus context metadata."""
    return {
        "features": {
            "Destination Port": 80,
            "Flow Duration": 1532000,
            "Total Fwd Packets": 12,
            "Total Backward Packets": 10,
            "Flow Bytes/s": 6540.4,
            "SYN Flag Count": 1,
            "ACK Flag Count": 20,
        },
        "src_ip": "192.168.10.5",
        "dst_ip": "10.0.1.20",
        "dst_port": 80,
        "protocol": "TCP",
    }


@pytest.fixture
def valid_event() -> dict:
    return {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "source": {"type": "auth", "hostname": "web-server-01", "ip": "10.0.1.45"},
        "actor": {"user": "admin", "source_ip": "203.0.113.45"},
        "action": {"type": "authentication", "result": "failure"},
    }


# ============================================================================
# Independent degradation — the case that keeps an outage from spreading
# ============================================================================


class TestIndependentDegradation:
    """
    The two tracks share a process but must fail apart.

    A missing flow artifact is the *normal* state of a fresh deployment. If it
    took the event endpoints with it, adding the flow track would have made the
    platform strictly less available.
    """

    def test_flow_endpoint_returns_503_without_models(
        self, client_without_flow_models: TestClient, valid_flow: dict, auth_headers: dict
    ):
        response = client_without_flow_models.post(
            "/v1/predict/flow", json=valid_flow, headers=auth_headers
        )

        assert response.status_code == 503

    def test_event_endpoint_still_returns_200_without_flow_models(
        self, client_without_flow_models: TestClient, valid_event: dict, auth_headers: dict
    ):
        """The whole point of the previous test: the blast radius stops there."""
        response = client_without_flow_models.post(
            "/v1/predict", json=valid_event, headers=auth_headers
        )

        assert response.status_code == 200

    def test_flow_batch_returns_503_without_models(
        self, client_without_flow_models: TestClient, valid_flow: dict, auth_headers: dict
    ):
        response = client_without_flow_models.post(
            "/v1/analyze/flow/batch", json={"flows": [valid_flow]}, headers=auth_headers
        )

        assert response.status_code == 503

    def test_event_batch_still_works_without_flow_models(
        self, client_without_flow_models: TestClient, valid_event: dict, auth_headers: dict
    ):
        response = client_without_flow_models.post(
            "/v1/analyze/batch", json={"events": [valid_event]}, headers=auth_headers
        )

        assert response.status_code == 200

    def test_readiness_stays_green_without_flow_models(
        self, client_without_flow_models: TestClient
    ):
        """
        Missing flow models must not fail the probe. If they did, Kubernetes
        would pull the pod out of service over an optional track.
        """
        response = client_without_flow_models.get("/ready")

        assert response.status_code == 200
        assert response.json()["checks"]["flow_detector"] is False

    def test_readiness_reports_the_flow_checks_when_loaded(self, client: TestClient):
        checks = client.get("/ready").json()["checks"]

        assert checks["flow_detector"] is True
        assert checks["flow_classifier"] is True

    def test_readiness_still_reports_the_event_checks(self, client: TestClient):
        """The existing contract of ``/ready`` is unchanged for the event track."""
        checks = client.get("/ready").json()["checks"]

        for check in ("feature_extractor", "anomaly_detector", "attack_classifier"):
            assert check in checks

    def test_a_rejected_flow_artifact_does_not_prevent_startup(
        self, client_with_stale_flow_models: TestClient, valid_event: dict, auth_headers: dict
    ):
        """
        A stale cascade is a *louder* failure than a missing one — it needs a
        retrain — but it must still not abort the lifespan. If it did, one
        outdated flow artifact would stop the whole API from booting and take
        the event track offline with it.
        """
        assert (
            client_with_stale_flow_models.post(
                "/v1/predict", json=valid_event, headers=auth_headers
            ).status_code
            == 200
        )

    def test_a_rejected_flow_artifact_degrades_only_the_flow_routes(
        self, client_with_stale_flow_models: TestClient, valid_flow: dict, auth_headers: dict
    ):
        response = client_with_stale_flow_models.post(
            "/v1/predict/flow", json=valid_flow, headers=auth_headers
        )

        assert response.status_code == 503
        assert client_with_stale_flow_models.get("/ready").status_code == 200

    def test_a_broken_event_track_still_fails_the_probe(
        self, client: TestClient, loaded_flow_service: MagicMock
    ):
        """
        Readiness is gated by the event track alone — so an unloaded event model
        must still produce a 503, flow track healthy or not.
        """
        client.app.state.anomaly_detector.is_loaded = False

        response = client.get("/ready")

        assert response.status_code == 503
        assert response.json()["detail"]["checks"]["flow_detector"] is True


# ============================================================================
# POST /v1/predict/flow
# ============================================================================


class TestPredictFlowEndpoint:
    """Single-flow endpoint."""

    def test_requires_auth(self, client: TestClient, valid_flow: dict):
        assert client.post("/v1/predict/flow", json=valid_flow).status_code == 401

    def test_rejects_an_expired_token(self, client: TestClient, valid_flow: dict):
        from api.security.auth import JWT_ALGORITHM, JWT_SECRET_KEY

        payload = {
            "sub": "test_user",
            "exp": datetime.utcnow() - timedelta(hours=1),
            "iat": datetime.utcnow() - timedelta(hours=2),
            "scopes": ["predict"],
        }
        token = jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)

        response = client.post(
            "/v1/predict/flow", json=valid_flow, headers={"Authorization": f"Bearer {token}"}
        )

        assert response.status_code == 401

    def test_rejects_a_forged_signature(self, client: TestClient, valid_flow: dict):
        from api.security.auth import JWT_ALGORITHM

        payload = {
            "sub": "test_user",
            "exp": datetime.utcnow() + timedelta(hours=1),
            "scopes": ["predict"],
        }
        token = jwt.encode(payload, "wrong_secret", algorithm=JWT_ALGORITHM)

        response = client.post(
            "/v1/predict/flow", json=valid_flow, headers={"Authorization": f"Bearer {token}"}
        )

        assert response.status_code == 401

    def test_returns_200_with_valid_auth(
        self, client: TestClient, valid_flow: dict, auth_headers: dict
    ):
        response = client.post("/v1/predict/flow", json=valid_flow, headers=auth_headers)

        assert response.status_code == 200

    def test_response_carries_every_documented_field(
        self, client: TestClient, valid_flow: dict, auth_headers: dict
    ):
        data = client.post("/v1/predict/flow", json=valid_flow, headers=auth_headers).json()

        for field in (
            "flow_id",
            "threat_score",
            "severity",
            "attack_type",
            "confidence",
            "is_anomaly",
            "explanation",
            "recommended_actions",
            "attack_probabilities",
            "timestamp",
        ):
            assert field in data, f"Missing field: {field}"

    def test_threat_score_and_severity_are_in_range(
        self, client: TestClient, valid_flow: dict, auth_headers: dict
    ):
        data = client.post("/v1/predict/flow", json=valid_flow, headers=auth_headers).json()

        assert 0 <= data["threat_score"] <= 1
        assert data["severity"] in ("LOW", "MEDIUM", "HIGH", "CRITICAL")

    def test_attack_probabilities_come_from_the_fusion_details(
        self, client: TestClient, valid_flow: dict, auth_headers: dict
    ):
        data = client.post("/v1/predict/flow", json=valid_flow, headers=auth_headers).json()

        assert data["attack_probabilities"] == {"BENIGN": 0.05, "DDoS": 0.95}

    def test_each_request_gets_its_own_flow_id(
        self, client: TestClient, valid_flow: dict, auth_headers: dict
    ):
        first = client.post("/v1/predict/flow", json=valid_flow, headers=auth_headers).json()
        second = client.post("/v1/predict/flow", json=valid_flow, headers=auth_headers).json()

        assert first["flow_id"] != second["flow_id"]

    def test_the_service_receives_the_wire_shape(
        self,
        client: TestClient,
        valid_flow: dict,
        auth_headers: dict,
        loaded_flow_service: MagicMock,
    ):
        """
        Features nested, metadata alongside. The service relies on that split to
        keep ``src_ip`` and friends out of the model.
        """
        client.post("/v1/predict/flow", json=valid_flow, headers=auth_headers)

        sent = loaded_flow_service.predict.call_args[0][0]

        assert sent["features"]["Destination Port"] == 80
        assert sent["src_ip"] == "192.168.10.5"
        assert "src_ip" not in sent["features"]

    def test_a_model_failure_becomes_a_500_not_a_stack_trace(
        self,
        client: TestClient,
        valid_flow: dict,
        auth_headers: dict,
        loaded_flow_service: MagicMock,
    ):
        loaded_flow_service.predict.side_effect = RuntimeError("boom")

        response = client.post("/v1/predict/flow", json=valid_flow, headers=auth_headers)

        assert response.status_code == 500
        assert "boom" not in response.text


class TestFlowValidation:
    """Request validation, before anything reaches a model."""

    def test_rejects_a_payload_without_features(self, client: TestClient, auth_headers: dict):
        response = client.post(
            "/v1/predict/flow", json={"src_ip": "10.0.0.1"}, headers=auth_headers
        )

        assert response.status_code == 422

    def test_rejects_an_empty_features_object(self, client: TestClient, auth_headers: dict):
        response = client.post("/v1/predict/flow", json={"features": {}}, headers=auth_headers)

        assert response.status_code == 422

    def test_rejects_a_payload_with_no_recognised_feature_name(
        self, client: TestClient, auth_headers: dict
    ):
        """
        The failure this guards against is silent: unknown names are ignored, so
        a payload of entirely wrong keys becomes an all-zero vector and gets a
        confident answer about traffic the model never saw.
        """
        response = client.post(
            "/v1/predict/flow",
            json={"features": {"duration_ms": 12, "packets": 4}},
            headers=auth_headers,
        )

        assert response.status_code == 422

    def test_accepts_a_partial_flow(self, client: TestClient, auth_headers: dict):
        """Absent features default to 0.0; a client need not send all 78."""
        response = client.post(
            "/v1/predict/flow", json={"features": {"Flow Duration": 1200}}, headers=auth_headers
        )

        assert response.status_code == 200

    def test_accepts_every_contract_name(self, client: TestClient, auth_headers: dict):
        full = {name: 1.0 for name in FLOW_FEATURE_NAMES}

        response = client.post("/v1/predict/flow", json={"features": full}, headers=auth_headers)

        assert response.status_code == 200

    def test_rejects_a_non_numeric_feature_value(self, client: TestClient, auth_headers: dict):
        response = client.post(
            "/v1/predict/flow",
            json={"features": {"Flow Duration": "not a number"}},
            headers=auth_headers,
        )

        assert response.status_code == 422

    def test_rejects_an_invalid_source_ip(
        self, client: TestClient, valid_flow: dict, auth_headers: dict
    ):
        valid_flow["src_ip"] = "not-an-ip"

        response = client.post("/v1/predict/flow", json=valid_flow, headers=auth_headers)

        assert response.status_code == 422

    def test_rejects_an_out_of_range_port(
        self, client: TestClient, valid_flow: dict, auth_headers: dict
    ):
        valid_flow["dst_port"] = 70000

        response = client.post("/v1/predict/flow", json=valid_flow, headers=auth_headers)

        assert response.status_code == 422

    def test_metadata_is_optional(self, client: TestClient, auth_headers: dict):
        response = client.post(
            "/v1/predict/flow", json={"features": {"Flow Duration": 1200}}, headers=auth_headers
        )

        assert response.status_code == 200

    def test_explicit_null_metadata_is_accepted(self, client: TestClient, auth_headers: dict):
        """A client that serialises absent context as ``null`` is not an error."""
        response = client.post(
            "/v1/predict/flow",
            json={
                "features": {"Flow Duration": 1200},
                "src_ip": None,
                "dst_ip": None,
                "dst_port": None,
                "protocol": None,
            },
            headers=auth_headers,
        )

        assert response.status_code == 200


# ============================================================================
# POST /v1/analyze/flow/batch
# ============================================================================


class TestFlowBatchEndpoint:
    """Batch endpoint."""

    def test_requires_auth(self, client: TestClient, valid_flow: dict):
        response = client.post("/v1/analyze/flow/batch", json={"flows": [valid_flow]})

        assert response.status_code == 401

    def test_returns_200_with_valid_auth(
        self, client: TestClient, valid_flow: dict, auth_headers: dict
    ):
        response = client.post(
            "/v1/analyze/flow/batch", json={"flows": [valid_flow]}, headers=auth_headers
        )

        assert response.status_code == 200

    def test_response_format(self, client: TestClient, valid_flow: dict, auth_headers: dict):
        data = client.post(
            "/v1/analyze/flow/batch",
            json={"flows": [valid_flow, valid_flow]},
            headers=auth_headers,
        ).json()

        assert data["total"] == 2
        assert data["processed"] == 2
        assert len(data["results"]) == 2

    def test_rejects_an_empty_batch(self, client: TestClient, auth_headers: dict):
        response = client.post("/v1/analyze/flow/batch", json={"flows": []}, headers=auth_headers)

        assert response.status_code == 422

    def test_enforces_the_hundred_flow_limit(
        self, client: TestClient, valid_flow: dict, auth_headers: dict
    ):
        response = client.post(
            "/v1/analyze/flow/batch", json={"flows": [valid_flow] * 101}, headers=auth_headers
        )

        assert response.status_code == 422

    def test_accepts_exactly_a_hundred_flows(
        self, client: TestClient, valid_flow: dict, auth_headers: dict
    ):
        response = client.post(
            "/v1/analyze/flow/batch", json={"flows": [valid_flow] * 100}, headers=auth_headers
        )

        assert response.status_code == 200

    def test_uses_one_vectorised_call_not_a_loop(
        self,
        client: TestClient,
        valid_flow: dict,
        auth_headers: dict,
        loaded_flow_service: MagicMock,
    ):
        """
        The reason this endpoint exists alongside the single-flow route. The
        event batch endpoint loops over ``predict``; this one must not.
        """
        client.post(
            "/v1/analyze/flow/batch", json={"flows": [valid_flow] * 25}, headers=auth_headers
        )

        assert loaded_flow_service.predict_batch.call_count == 1
        assert len(loaded_flow_service.predict_batch.call_args[0][0]) == 25
        assert loaded_flow_service.predict.call_count == 0

    def test_every_result_gets_a_distinct_id(
        self, client: TestClient, valid_flow: dict, auth_headers: dict
    ):
        data = client.post(
            "/v1/analyze/flow/batch", json={"flows": [valid_flow] * 5}, headers=auth_headers
        ).json()

        assert len({r["flow_id"] for r in data["results"]}) == 5

    def test_a_model_failure_becomes_a_500(
        self,
        client: TestClient,
        valid_flow: dict,
        auth_headers: dict,
        loaded_flow_service: MagicMock,
    ):
        loaded_flow_service.predict_batch.side_effect = RuntimeError("boom")

        response = client.post(
            "/v1/analyze/flow/batch", json={"flows": [valid_flow]}, headers=auth_headers
        )

        assert response.status_code == 500

    def test_rejects_a_batch_containing_an_invalid_flow(
        self, client: TestClient, valid_flow: dict, auth_headers: dict
    ):
        response = client.post(
            "/v1/analyze/flow/batch",
            json={"flows": [valid_flow, {"features": {}}]},
            headers=auth_headers,
        )

        assert response.status_code == 422


# ============================================================================
# Metrics
# ============================================================================


class TestFlowMetrics:
    """
    Both tracks report through the same collectors, separated by a ``track``
    label rather than by a second set of metrics. Duplicated collectors would
    mean every dashboard panel and alert rule written twice.
    """

    def test_flow_predictions_are_labelled_flow(
        self, client: TestClient, valid_flow: dict, auth_headers: dict
    ):
        client.post("/v1/predict/flow", json=valid_flow, headers=auth_headers)

        body = client.get("/metrics").text

        assert 'track="flow"' in body

    def test_event_predictions_are_labelled_event(
        self, client: TestClient, valid_event: dict, auth_headers: dict
    ):
        client.post("/v1/predict", json=valid_event, headers=auth_headers)

        body = client.get("/metrics").text

        assert 'track="event"' in body

    def test_the_collectors_are_not_duplicated_per_track(self, client: TestClient):
        """One metric family, two label values — not two metric families."""
        body = client.get("/metrics").text

        assert "sentinel_flow_attack_type_total" not in body
        assert "sentinel_attack_type_total" in body


# ============================================================================
# data/schemas/flow.json
# ============================================================================


class TestPublishedJsonSchema:
    """
    ``data/schemas/flow.json`` is the contract clients integrate against, and
    nothing else in the codebase reads it — so without these assertions it can
    drift away from the Pydantic model that actually validates requests, and the
    first symptom is an integrator whose conforming payload gets a 422.

    Full JSON Schema validation would need a ``jsonschema`` dependency the
    project does not carry. Checking the two definitions against each other
    catches the drift that matters and costs nothing.
    """

    @pytest.fixture(scope="class")
    def schema(self) -> dict:
        import json
        from pathlib import Path

        path = Path(__file__).resolve().parents[2] / "data" / "schemas" / "flow.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def test_it_is_a_draft_2020_12_schema(self, schema: dict):
        """Same dialect as ``event.json``, which it sits beside."""
        assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        assert schema["title"] == "NetworkFlow"

    def test_its_top_level_fields_match_the_pydantic_model(self, schema: dict):
        from api.models.flow_schemas import NetworkFlow

        assert set(schema["properties"]) == set(NetworkFlow.model_fields)

    def test_features_is_the_only_required_field(self, schema: dict):
        from api.models.flow_schemas import NetworkFlow

        required_in_model = {
            name for name, f in NetworkFlow.model_fields.items() if f.is_required()
        }

        assert set(schema["required"]) == required_in_model == {"features"}

    def test_every_documented_feature_name_is_a_real_one(self, schema: dict):
        """
        A typo in this file tells an integrator to send a name the extractor
        will silently ignore — the exact failure the whole contract exists to
        prevent, arriving through the documentation instead of the code.
        """
        from ml.flow.features import FLOW_FEATURE_ALIASES

        documented = set(schema["$defs"]["FlowFeatures"]["properties"])
        accepted = set(FLOW_FEATURE_NAMES) | set(FLOW_FEATURE_ALIASES)

        assert documented <= accepted

    def test_it_documents_the_duplicated_column_and_its_alias(self, schema: dict):
        documented = schema["$defs"]["FlowFeatures"]["properties"]

        assert "Fwd Header Length" in documented
        assert "Fwd Header Length.1" in documented

    def test_its_examples_are_accepted_by_the_pydantic_model(self, schema: dict):
        """The published examples must actually work against the live endpoint."""
        from api.models.flow_schemas import NetworkFlow

        assert schema["examples"]
        for example in schema["examples"]:
            NetworkFlow.model_validate(example)

    def test_its_examples_are_accepted_by_the_endpoint(
        self, schema: dict, client: TestClient, auth_headers: dict
    ):
        for example in schema["examples"]:
            response = client.post("/v1/predict/flow", json=example, headers=auth_headers)

            assert response.status_code == 200, response.text


# ============================================================================
# Import hygiene
# ============================================================================


def test_api_main_does_not_import_matplotlib():
    """
    ``api/Dockerfile`` copies ``ml/`` wholesale and installs only
    ``requirements.txt``, where matplotlib is absent. Now that ``api.main``
    imports from ``ml.flow``, that chain needs its own guard.
    """
    import subprocess
    import sys
    from pathlib import Path

    result = subprocess.run(
        [sys.executable, "-c", "import api.main, sys; assert 'matplotlib' not in sys.modules"],
        check=False,
        capture_output=True,
        cwd=Path(__file__).resolve().parents[2],
    )

    assert result.returncode == 0, result.stderr.decode(errors="replace")
