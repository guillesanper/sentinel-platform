"""
Pytest configuration and shared fixtures.

This file contains fixtures that are available to all tests in the test suite.
"""

import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from jose import jwt

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
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "e2e: marks tests as end-to-end tests")


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
        "normal",
        "brute_force",
        "credential_stuffing",
        "privilege_escalation",
        "lateral_movement",
        "data_exfiltration",
        "reconnaissance",
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
# Flow track (CICIDS2017) fixtures
# ============================================================================

# Feature columns of the mini CSVs. Three is enough to exercise loading,
# cleaning and partitioning; the real 78-column contract is pinned separately by
# FLOW_FEATURE_NAMES.
CICIDS_MINI_FEATURES: tuple[str, ...] = (
    "Flow Duration",
    "Total Fwd Packets",
    "Flow Bytes/s",
)

# The three characters U+00EF U+00BF U+00BD, i.e. what the en dash of the
# "Web Attack" labels becomes when the CSV is read with latin-1. Built from code
# points on purpose: written literally, a bad encoding round-trip on this file
# would silently stop exercising the mojibake repair in map_label, and the
# WebAttack class would vanish from the fixture without any test failing.
CICIDS_MOJIBAKE_EN_DASH: str = "".join(chr(cp) for cp in (0xEF, 0xBF, 0xBD))

# Filename → raw labels written into the CSV's attack window, **in order**. The
# names mimic the real MachineLearningCVE distribution so that map_capture_day
# resolves them to a real weekday.
#
# Tuesday and Wednesday carry two raw labels each, run one after the other, and
# that is the single most important property of this fixture. Both pairs collapse
# to one canonical class (FTP-Patator + SSH-Patator → BruteForce; DoS Hulk +
# DoS GoldenEye → DoS), exactly as in the real captures — so a chronological cut
# blocked on the canonical class holds the *second* tool entirely out of
# training, while a cut blocked on the raw label keeps both on both sides. The
# difference between those two protocols is worth 0.47 of F1-macro on the real
# data; see docs/ml/PARTICION-TEMPORAL.md.
CICIDS_MINI_FILES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Tuesday-WorkingHours.pcap_ISCX.csv", ("FTP-Patator", "SSH-Patator")),
    ("Wednesday-workingHours.pcap_ISCX.csv", ("DoS Hulk", "DoS GoldenEye")),
    (
        "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
        (f"Web Attack {CICIDS_MOJIBAKE_EN_DASH} XSS",),
    ),
    ("Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv", ("DDoS",)),
    ("Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv", ("PortScan",)),
)

# Canonical classes that the fixture builds from more than one attack tool.
CICIDS_MINI_MULTI_TOOL_CLASSES: tuple[str, ...] = ("BruteForce", "DoS")

# Rows per file: a benign lead-in, the attack window, and a benign tail. This is
# the shape measured on the real captures (decile 0 benign, deciles 1-8 attack,
# decile 9 benign), and it is what makes a naive global positional cut produce a
# 100% benign test set.
CICIDS_MINI_LEAD_IN: int = 120
CICIDS_MINI_ATTACK: int = 240
CICIDS_MINI_TAIL: int = 120

# The sixth file, used **only** by ``cicids_rare_class_csv``. Friday morning is
# where the real Bot capture lives, and CAPTURE_FILE_ORDER puts it before the two
# Friday afternoon files, so the rare block also lands in the middle of the
# capture week rather than at its end.
CICIDS_RARE_CLASS_FILE: str = "Friday-WorkingHours-Morning.pcap_ISCX.csv"
CICIDS_RARE_CLASS_RAW_LABEL: str = "Bot"
CICIDS_RARE_CLASS_LABEL: str = "Botnet"

# Deliberately between the two thresholds that matter. Above
# FlowConfig.min_class_samples (100), so filter_minority_classes keeps the
# class — the real Botnet survives that filter too. Far below
# ml.flow.loader.MIN_MEANINGFUL_BLOCK_ROWS (2.000), so a proportional draw
# grinds it down to a handful of rows and the support audit reports it.
CICIDS_RARE_CLASS_ATTACK: int = 120


def _write_cicids_mini_csv(
    path: Path,
    raw_labels: tuple[str, ...],
    *,
    seed: int,
    inject_artifacts: bool,
    attack_rows: int | None = None,
) -> None:
    """
    Write one synthetic CICIDS2017-shaped CSV with a temporal structure.

    The attack window is divided evenly among ``raw_labels`` and the parts are
    laid down **in the given order**, so a file with two labels reproduces the
    real "tool A, then tool B" sequence within a single capture day.

    ``attack_rows`` overrides the width of that window; ``None`` keeps
    :data:`CICIDS_MINI_ATTACK`, so every existing caller writes exactly the file
    it wrote before. A narrow window is how ``cicids_rare_class_csv`` builds a
    class small enough to be sampled into noise.
    """
    rng = np.random.default_rng(seed)

    def block(n: int, label: str, centre: float) -> dict:
        return {
            "Flow Duration": rng.normal(centre, 5.0, n),
            "Total Fwd Packets": rng.integers(1, 40, n).astype(float),
            "Flow Bytes/s": rng.normal(centre * 10, 50.0, n),
            "Label": [label] * n,
        }

    n_attack = CICIDS_MINI_ATTACK if attack_rows is None else attack_rows
    per_label = n_attack // len(raw_labels)
    frames = [pd.DataFrame(block(CICIDS_MINI_LEAD_IN, "BENIGN", 100.0))]
    for i, raw_label in enumerate(raw_labels):
        # Each tool gets its own feature centre: two tools of the same family are
        # related but not identical, which is what makes holding one out a real
        # generalisation demand rather than a relabelling.
        frames.append(pd.DataFrame(block(per_label, raw_label, 500.0 + 60.0 * i)))
    frames.append(pd.DataFrame(block(CICIDS_MINI_TAIL, "BENIGN", 100.0)))
    df = pd.concat(frames, ignore_index=True)

    if inject_artifacts:
        # Real CSVs carry these three; the loader must survive all of them.
        df.loc[5, "Flow Bytes/s"] = np.inf
        df.loc[7, "Flow Duration"] = np.nan
        df.loc[9] = df.loc[8]  # exact duplicate row
        # An unmapped label: its rows must be discarded, not crash the load.
        df.loc[11, "Label"] = "Some Future Attack"

    # Leading whitespace in the header is present in the real files.
    df.columns = [f" {c}" for c in df.columns]
    df.to_csv(path, index=False, encoding="latin-1")


@pytest.fixture
def cicids_mini_csv(tmp_path: Path) -> Path:
    """
    Generate a miniature CICIDS2017 dataset and return its directory.

    The CSVs are written at test time rather than committed: ``.gitignore`` has
    a global ``*.csv`` rule, so a committed fixture would silently disappear on
    a fresh clone.

    Each file holds one attack class surrounded by benign traffic, mirroring the
    real capture layout, so the temporal split has a meaningful notion of
    "earlier" inside every block. Two of the files run **two attack tools in
    sequence** under a single canonical class — see
    :data:`CICIDS_MINI_FILES` for why that detail carries the weight of the
    temporal-blocking tests.
    """
    data_dir = tmp_path / "cicids-mini"
    data_dir.mkdir()

    for seed, (filename, raw_labels) in enumerate(CICIDS_MINI_FILES):
        _write_cicids_mini_csv(
            data_dir / filename,
            raw_labels,
            seed=1000 + seed,
            inject_artifacts=(seed == 0),
        )

    return data_dir


@pytest.fixture
def cicids_rare_class_csv(tmp_path: Path) -> Path:
    """
    The mini dataset plus one deliberately tiny attack class, and its directory.

    Reproduces the Botnet problem in miniature. In the real captures Botnet is
    one sub-label (``Bot``), one day, ~1.966 rows out of ~2.8M; it clears
    ``min_class_samples`` comfortably and is then destroyed by the sampler,
    because strictly proportional sampling at ``--sample-size 200000`` keeps
    ~155 of those rows and the split leaves ~24 in test. The published F1 of
    0.00 for that class is measured on those 24 rows.

    So this fixture writes the five :data:`CICIDS_MINI_FILES` unchanged plus a
    sixth with a :data:`CICIDS_RARE_CLASS_ATTACK`-row ``Bot`` window. It is a
    **separate** fixture rather than a sixth entry in ``CICIDS_MINI_FILES``: the
    loader, split and pipeline tests assert exact class sets and split sizes
    against that tuple, and a ~40-row class may not survive Stage 1's routing at
    all, which would turn tests about other things flaky.

    Used by the ``class_sample_floor`` and ``audit_class_support`` tests.
    """
    data_dir = tmp_path / "cicids-rare-class"
    data_dir.mkdir()

    for seed, (filename, raw_labels) in enumerate(CICIDS_MINI_FILES):
        _write_cicids_mini_csv(
            data_dir / filename,
            raw_labels,
            seed=1000 + seed,
            inject_artifacts=(seed == 0),
        )

    _write_cicids_mini_csv(
        data_dir / CICIDS_RARE_CLASS_FILE,
        (CICIDS_RARE_CLASS_RAW_LABEL,),
        seed=1000 + len(CICIDS_MINI_FILES),
        inject_artifacts=False,
        attack_rows=CICIDS_RARE_CLASS_ATTACK,
    )

    return data_dir


# ============================================================================
# Async Fixtures
# ============================================================================


@pytest.fixture
def event_loop_policy():
    """Configure event loop for async tests."""
    return asyncio.DefaultEventLoopPolicy()
