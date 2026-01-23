"""
Pytest configuration and shared fixtures.

This file contains fixtures that are available to all tests in the test suite.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pytest

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# ============================================================================
# Configuration
# ============================================================================


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "e2e: marks tests as end-to-end tests"
    )


# ============================================================================
# Shared Fixtures
# ============================================================================


@pytest.fixture(scope="session")
def project_root_path() -> Path:
    """Return the project root path."""
    return project_root


@pytest.fixture
def sample_security_event() -> dict:
    """Create a sample security event for testing."""
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
def sample_brute_force_event() -> dict:
    """Create a sample brute force attack event."""
    return {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "source": {
            "type": "auth",
            "hostname": "web-server-01",
            "ip": "10.0.1.45",
            "service": "sshd",
        },
        "actor": {
            "user": "root",
            "source_ip": "185.220.100.45",  # Tor exit node
            "geo": {
                "country": "RU",
                "city": "Moscow",
                "lat": 55.7558,
                "lon": 37.6173,
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
def sample_privilege_escalation_event() -> dict:
    """Create a sample privilege escalation event."""
    return {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "source": {
            "type": "system",
            "hostname": "db-server-01",
            "ip": "10.0.2.10",
            "service": "auditd",
        },
        "actor": {
            "user": "developer",
            "uid": 1001,
            "source_ip": "10.0.1.50",
        },
        "action": {
            "type": "authorization",
            "subtype": "sudo",
            "target": "root",
            "result": "failure",
            "reason": "not in sudoers",
        },
        "context": {
            "process": "bash",
            "pid": 12345,
            "command": "sudo su -",
        },
    }


@pytest.fixture
def sample_features() -> np.ndarray:
    """Generate sample feature vector."""
    np.random.seed(42)
    return np.random.rand(17).astype(np.float32)


@pytest.fixture
def sample_feature_batch() -> np.ndarray:
    """Generate a batch of sample feature vectors."""
    np.random.seed(42)
    return np.random.rand(100, 17).astype(np.float32)


@pytest.fixture
def sample_labels() -> list[str]:
    """Generate sample labels for training."""
    labels = [
        "normal", "brute_force", "credential_stuffing",
        "privilege_escalation", "lateral_movement",
        "data_exfiltration", "reconnaissance",
    ]
    np.random.seed(42)
    return [labels[i % len(labels)] for i in range(100)]


@pytest.fixture
def jwt_secret() -> str:
    """Return test JWT secret."""
    return "test-secret-key-for-testing-only"


@pytest.fixture
def valid_jwt_token(jwt_secret: str) -> str:
    """Generate a valid JWT token for testing."""
    from jose import jwt

    payload = {
        "sub": "test_user",
        "exp": datetime.utcnow() + timedelta(hours=1),
        "iat": datetime.utcnow(),
        "scopes": ["predict", "batch"],
        "tenant_id": "test_tenant",
    }
    return jwt.encode(payload, jwt_secret, algorithm="HS256")


@pytest.fixture
def expired_jwt_token(jwt_secret: str) -> str:
    """Generate an expired JWT token for testing."""
    from jose import jwt

    payload = {
        "sub": "test_user",
        "exp": datetime.utcnow() - timedelta(hours=1),
        "iat": datetime.utcnow() - timedelta(hours=2),
        "scopes": ["predict"],
    }
    return jwt.encode(payload, jwt_secret, algorithm="HS256")


# ============================================================================
# Utility Fixtures
# ============================================================================


@pytest.fixture
def temp_model_dir(tmp_path: Path) -> Path:
    """Create a temporary directory for model storage."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    return model_dir


@pytest.fixture
def mock_env_vars(monkeypatch):
    """Set up mock environment variables for testing."""
    monkeypatch.setenv("JWT_SECRET_KEY", "test-secret-key")
    monkeypatch.setenv("JWT_ALGORITHM", "HS256")
    monkeypatch.setenv("JWT_EXPIRE_MINUTES", "60")
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")


# ============================================================================
# Async Fixtures
# ============================================================================


@pytest.fixture
def event_loop_policy():
    """Configure event loop for async tests."""
    import asyncio
    return asyncio.DefaultEventLoopPolicy()
