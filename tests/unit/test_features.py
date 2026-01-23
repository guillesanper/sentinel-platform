"""
Unit tests for the feature extraction pipeline.

Tests the FeatureExtractor class to ensure correct feature
extraction from security events.
"""

import numpy as np
import pytest
from datetime import datetime

from ml.features.pipeline import (
    FeatureExtractor,
    SOURCE_TYPE_MAP,
    ACTION_TYPE_MAP,
    PRIVATE_RANGES,
    TOR_EXIT_NODES,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def extractor() -> FeatureExtractor:
    """Create a FeatureExtractor instance."""
    return FeatureExtractor()


@pytest.fixture
def valid_event() -> dict:
    """Create a valid security event."""
    return {
        "timestamp": "2024-01-15T14:32:18.123Z",
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
def minimal_event() -> dict:
    """Create a minimal security event with only required fields."""
    return {
        "timestamp": "2024-01-15T10:00:00.000Z",
        "source": {
            "type": "system",
            "hostname": "server-01",
            "ip": "192.168.1.1",
        },
        "actor": {
            "user": "user1",
            "source_ip": "192.168.1.100",
        },
        "action": {
            "type": "file_access",
            "result": "success",
        },
    }


# ============================================================================
# Basic Functionality Tests
# ============================================================================


class TestFeatureExtractorInit:
    """Tests for FeatureExtractor initialization."""

    def test_init_creates_instance(self, extractor: FeatureExtractor) -> None:
        """Test that FeatureExtractor initializes correctly."""
        assert extractor is not None
        assert extractor.n_features == 17

    def test_feature_names_length(self, extractor: FeatureExtractor) -> None:
        """Test that feature names list has correct length."""
        names = extractor.get_feature_names()
        assert len(names) == 17
        assert names == extractor.FEATURE_NAMES


class TestTransformValidEvent:
    """Tests for transforming valid events."""

    def test_transform_returns_ndarray(
        self, extractor: FeatureExtractor, valid_event: dict
    ) -> None:
        """Test that transform returns numpy array."""
        features = extractor.transform(valid_event)
        assert isinstance(features, np.ndarray)

    def test_transform_correct_shape(
        self, extractor: FeatureExtractor, valid_event: dict
    ) -> None:
        """Test that transformed features have correct shape."""
        features = extractor.transform(valid_event)
        assert features.shape == (17,)

    def test_transform_correct_dtype(
        self, extractor: FeatureExtractor, valid_event: dict
    ) -> None:
        """Test that features are float32."""
        features = extractor.transform(valid_event)
        assert features.dtype == np.float32

    def test_transform_values_in_range(
        self, extractor: FeatureExtractor, valid_event: dict
    ) -> None:
        """Test that all feature values are in valid range."""
        features = extractor.transform(valid_event)
        # Most features should be between 0 and 1
        assert np.all(features >= 0)
        assert np.all(features <= 1)


class TestTransformMinimalEvent:
    """Tests for transforming minimal events."""

    def test_transform_minimal_event(
        self, extractor: FeatureExtractor, minimal_event: dict
    ) -> None:
        """Test that minimal event transforms correctly."""
        features = extractor.transform(minimal_event)
        assert features.shape == (17,)

    def test_transform_missing_optional_fields(
        self, extractor: FeatureExtractor
    ) -> None:
        """Test transform handles missing optional fields."""
        event = {
            "source": {"type": "auth", "hostname": "h", "ip": "10.0.0.1"},
            "actor": {"user": "u", "source_ip": "10.0.0.2"},
            "action": {"type": "authentication", "result": "success"},
        }
        # Should not raise, will use current time
        features = extractor.transform(event)
        assert features.shape == (17,)


# ============================================================================
# Temporal Features Tests
# ============================================================================


class TestTemporalFeatures:
    """Tests for temporal feature extraction."""

    def test_hour_of_day_normalized(self, extractor: FeatureExtractor) -> None:
        """Test hour of day is normalized to [0, 1]."""
        for hour in [0, 6, 12, 18, 23]:
            event = {
                "timestamp": f"2024-01-15T{hour:02d}:00:00.000Z",
                "source": {"type": "auth", "hostname": "h", "ip": "10.0.0.1"},
                "actor": {"user": "u", "source_ip": "10.0.0.2"},
                "action": {"type": "authentication", "result": "success"},
            }
            features = extractor.transform(event)
            assert 0 <= features[0] <= 1

    def test_day_of_week_normalized(self, extractor: FeatureExtractor) -> None:
        """Test day of week is normalized to [0, 1]."""
        # Monday = 0, Sunday = 6
        # 2024-01-15 is a Monday
        event = {
            "timestamp": "2024-01-15T12:00:00.000Z",
            "source": {"type": "auth", "hostname": "h", "ip": "10.0.0.1"},
            "actor": {"user": "u", "source_ip": "10.0.0.2"},
            "action": {"type": "authentication", "result": "success"},
        }
        features = extractor.transform(event)
        assert features[1] == 0.0  # Monday

    def test_is_weekend_detection(self, extractor: FeatureExtractor) -> None:
        """Test weekend detection."""
        # Saturday
        weekend_event = {
            "timestamp": "2024-01-20T12:00:00.000Z",
            "source": {"type": "auth", "hostname": "h", "ip": "10.0.0.1"},
            "actor": {"user": "u", "source_ip": "10.0.0.2"},
            "action": {"type": "authentication", "result": "success"},
        }
        features = extractor.transform(weekend_event)
        assert features[2] == 1.0  # is_weekend

        # Weekday
        weekday_event = {
            "timestamp": "2024-01-15T12:00:00.000Z",
            "source": {"type": "auth", "hostname": "h", "ip": "10.0.0.1"},
            "actor": {"user": "u", "source_ip": "10.0.0.2"},
            "action": {"type": "authentication", "result": "success"},
        }
        features = extractor.transform(weekday_event)
        assert features[2] == 0.0  # is_weekend

    def test_is_night_detection(self, extractor: FeatureExtractor) -> None:
        """Test night hours detection (22:00-06:00)."""
        # Night time (3 AM)
        night_event = {
            "timestamp": "2024-01-15T03:00:00.000Z",
            "source": {"type": "auth", "hostname": "h", "ip": "10.0.0.1"},
            "actor": {"user": "u", "source_ip": "10.0.0.2"},
            "action": {"type": "authentication", "result": "success"},
        }
        features = extractor.transform(night_event)
        assert features[3] == 1.0  # is_night

        # Day time (2 PM)
        day_event = {
            "timestamp": "2024-01-15T14:00:00.000Z",
            "source": {"type": "auth", "hostname": "h", "ip": "10.0.0.1"},
            "actor": {"user": "u", "source_ip": "10.0.0.2"},
            "action": {"type": "authentication", "result": "success"},
        }
        features = extractor.transform(day_event)
        assert features[3] == 0.0  # is_night

    def test_is_business_hours_detection(self, extractor: FeatureExtractor) -> None:
        """Test business hours detection (9-17 weekdays)."""
        # Business hours (10 AM Monday)
        business_event = {
            "timestamp": "2024-01-15T10:00:00.000Z",
            "source": {"type": "auth", "hostname": "h", "ip": "10.0.0.1"},
            "actor": {"user": "u", "source_ip": "10.0.0.2"},
            "action": {"type": "authentication", "result": "success"},
        }
        features = extractor.transform(business_event)
        assert features[4] == 1.0  # is_business_hours

        # After hours (8 PM)
        after_hours_event = {
            "timestamp": "2024-01-15T20:00:00.000Z",
            "source": {"type": "auth", "hostname": "h", "ip": "10.0.0.1"},
            "actor": {"user": "u", "source_ip": "10.0.0.2"},
            "action": {"type": "authentication", "result": "success"},
        }
        features = extractor.transform(after_hours_event)
        assert features[4] == 0.0  # is_business_hours


# ============================================================================
# IP Features Tests
# ============================================================================


class TestIPFeatures:
    """Tests for IP-based feature extraction."""

    def test_internal_ip_detection(self, extractor: FeatureExtractor) -> None:
        """Test detection of internal IP addresses."""
        internal_ips = ["10.0.0.1", "192.168.1.1", "172.16.0.1", "127.0.0.1"]

        for ip in internal_ips:
            event = {
                "timestamp": "2024-01-15T12:00:00.000Z",
                "source": {"type": "auth", "hostname": "h", "ip": "10.0.0.1"},
                "actor": {"user": "u", "source_ip": ip},
                "action": {"type": "authentication", "result": "success"},
            }
            features = extractor.transform(event)
            assert features[9] == 1.0, f"IP {ip} should be detected as internal"

    def test_external_ip_detection(self, extractor: FeatureExtractor) -> None:
        """Test detection of external IP addresses."""
        external_ips = ["203.0.113.45", "8.8.8.8", "1.1.1.1"]

        for ip in external_ips:
            event = {
                "timestamp": "2024-01-15T12:00:00.000Z",
                "source": {"type": "auth", "hostname": "h", "ip": "10.0.0.1"},
                "actor": {"user": "u", "source_ip": ip},
                "action": {"type": "authentication", "result": "success"},
            }
            features = extractor.transform(event)
            assert features[9] == 0.0, f"IP {ip} should be detected as external"

    def test_tor_exit_node_detection(self, extractor: FeatureExtractor) -> None:
        """Test detection of Tor exit node IPs."""
        tor_ip = "185.220.100.123"
        event = {
            "timestamp": "2024-01-15T12:00:00.000Z",
            "source": {"type": "auth", "hostname": "h", "ip": "10.0.0.1"},
            "actor": {"user": "u", "source_ip": tor_ip},
            "action": {"type": "authentication", "result": "success"},
        }
        features = extractor.transform(event)
        assert features[10] == 1.0  # actor_ip_is_tor

    def test_ip_octets_normalized(self, extractor: FeatureExtractor) -> None:
        """Test that IP octets are normalized to [0, 1]."""
        event = {
            "timestamp": "2024-01-15T12:00:00.000Z",
            "source": {"type": "auth", "hostname": "h", "ip": "10.0.0.1"},
            "actor": {"user": "u", "source_ip": "255.128.64.0"},
            "action": {"type": "authentication", "result": "success"},
        }
        features = extractor.transform(event)

        # Octets are at indices 11, 12, 13, 14 (use approx for float32 precision)
        assert features[11] == pytest.approx(255 / 255)  # 1.0
        assert features[12] == pytest.approx(128 / 255)
        assert features[13] == pytest.approx(64 / 255)
        assert features[14] == pytest.approx(0 / 255)  # 0.0

    def test_invalid_ip_handled(self, extractor: FeatureExtractor) -> None:
        """Test that invalid IPs are handled gracefully."""
        event = {
            "timestamp": "2024-01-15T12:00:00.000Z",
            "source": {"type": "auth", "hostname": "h", "ip": "invalid"},
            "actor": {"user": "u", "source_ip": "not-an-ip"},
            "action": {"type": "authentication", "result": "success"},
        }
        # Should not raise
        features = extractor.transform(event)
        assert features.shape == (17,)


# ============================================================================
# Action Features Tests
# ============================================================================


class TestActionFeatures:
    """Tests for action-based feature extraction."""

    def test_source_type_encoding(self, extractor: FeatureExtractor) -> None:
        """Test source type encoding."""
        for source_type in ["auth", "system", "network", "app"]:
            event = {
                "timestamp": "2024-01-15T12:00:00.000Z",
                "source": {"type": source_type, "hostname": "h", "ip": "10.0.0.1"},
                "actor": {"user": "u", "source_ip": "10.0.0.2"},
                "action": {"type": "authentication", "result": "success"},
            }
            features = extractor.transform(event)
            # source_type is at index 5, should be normalized
            assert 0 <= features[5] <= 1

    def test_action_type_encoding(self, extractor: FeatureExtractor) -> None:
        """Test action type encoding."""
        action_types = [
            "authentication",
            "authorization",
            "file_access",
            "network_connection",
        ]

        for action_type in action_types:
            event = {
                "timestamp": "2024-01-15T12:00:00.000Z",
                "source": {"type": "auth", "hostname": "h", "ip": "10.0.0.1"},
                "actor": {"user": "u", "source_ip": "10.0.0.2"},
                "action": {"type": action_type, "result": "success"},
            }
            features = extractor.transform(event)
            # action_type is at index 6
            assert 0 <= features[6] <= 1

    def test_privileged_action_detection(self, extractor: FeatureExtractor) -> None:
        """Test detection of privileged actions."""
        privileged_event = {
            "timestamp": "2024-01-15T12:00:00.000Z",
            "source": {"type": "system", "hostname": "h", "ip": "10.0.0.1"},
            "actor": {"user": "u", "source_ip": "10.0.0.2"},
            "action": {"type": "authorization", "subtype": "sudo", "result": "success"},
        }
        features = extractor.transform(privileged_event)
        assert features[7] == 1.0  # action_is_privileged

        normal_event = {
            "timestamp": "2024-01-15T12:00:00.000Z",
            "source": {"type": "system", "hostname": "h", "ip": "10.0.0.1"},
            "actor": {"user": "u", "source_ip": "10.0.0.2"},
            "action": {"type": "file_access", "subtype": "read", "result": "success"},
        }
        features = extractor.transform(normal_event)
        assert features[7] == 0.0  # action_is_privileged

    def test_failure_detection(self, extractor: FeatureExtractor) -> None:
        """Test detection of failed actions."""
        failure_event = {
            "timestamp": "2024-01-15T12:00:00.000Z",
            "source": {"type": "auth", "hostname": "h", "ip": "10.0.0.1"},
            "actor": {"user": "u", "source_ip": "10.0.0.2"},
            "action": {"type": "authentication", "result": "failure"},
        }
        features = extractor.transform(failure_event)
        assert features[8] == 1.0  # action_is_failure

        success_event = {
            "timestamp": "2024-01-15T12:00:00.000Z",
            "source": {"type": "auth", "hostname": "h", "ip": "10.0.0.1"},
            "actor": {"user": "u", "source_ip": "10.0.0.2"},
            "action": {"type": "authentication", "result": "success"},
        }
        features = extractor.transform(success_event)
        assert features[8] == 0.0  # action_is_failure


# ============================================================================
# Batch Transform Tests
# ============================================================================


class TestBatchTransform:
    """Tests for batch transformation."""

    def test_batch_transform_shape(
        self, extractor: FeatureExtractor, valid_event: dict
    ) -> None:
        """Test batch transform returns correct shape."""
        events = [valid_event] * 5
        features = extractor.transform_batch(events)
        assert features.shape == (5, 17)

    def test_batch_transform_empty_list(self, extractor: FeatureExtractor) -> None:
        """Test batch transform with empty list."""
        features = extractor.transform_batch([])
        assert features.shape == (0,) or len(features) == 0

    def test_batch_transform_consistency(
        self, extractor: FeatureExtractor, valid_event: dict
    ) -> None:
        """Test that batch transform gives same results as single transform."""
        single_features = extractor.transform(valid_event)
        batch_features = extractor.transform_batch([valid_event])

        np.testing.assert_array_equal(single_features, batch_features[0])


# ============================================================================
# Feature Importance Context Tests
# ============================================================================


class TestFeatureImportanceContext:
    """Tests for feature importance context generation."""

    def test_get_feature_importance_context(
        self, extractor: FeatureExtractor
    ) -> None:
        """Test feature importance context returns descriptions."""
        # Create features that trigger various descriptions
        features = np.zeros(17, dtype=np.float32)
        features[3] = 1.0  # is_night
        features[7] = 1.0  # action_is_privileged
        features[8] = 1.0  # action_is_failure
        features[10] = 1.0  # actor_ip_is_tor

        descriptions = extractor.get_feature_importance_context(features, top_k=5)

        assert isinstance(descriptions, list)
        assert len(descriptions) <= 5

    def test_get_feature_names_immutable(self, extractor: FeatureExtractor) -> None:
        """Test that get_feature_names returns a copy."""
        names1 = extractor.get_feature_names()
        names2 = extractor.get_feature_names()

        # Modify one copy
        names1[0] = "modified"

        # Original should be unchanged
        assert names2[0] != "modified"
        assert extractor.FEATURE_NAMES[0] != "modified"


# ============================================================================
# Edge Cases Tests
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_missing_timestamp_uses_current(self, extractor: FeatureExtractor) -> None:
        """Test that missing timestamp defaults to current time."""
        event = {
            "source": {"type": "auth", "hostname": "h", "ip": "10.0.0.1"},
            "actor": {"user": "u", "source_ip": "10.0.0.2"},
            "action": {"type": "authentication", "result": "success"},
        }
        features = extractor.transform(event)
        assert features.shape == (17,)

    def test_invalid_timestamp_format(self, extractor: FeatureExtractor) -> None:
        """Test handling of invalid timestamp format."""
        event = {
            "timestamp": "not-a-date",
            "source": {"type": "auth", "hostname": "h", "ip": "10.0.0.1"},
            "actor": {"user": "u", "source_ip": "10.0.0.2"},
            "action": {"type": "authentication", "result": "success"},
        }
        # Should not raise
        features = extractor.transform(event)
        assert features.shape == (17,)

    def test_unknown_source_type(self, extractor: FeatureExtractor) -> None:
        """Test handling of unknown source type."""
        event = {
            "timestamp": "2024-01-15T12:00:00.000Z",
            "source": {"type": "unknown_type", "hostname": "h", "ip": "10.0.0.1"},
            "actor": {"user": "u", "source_ip": "10.0.0.2"},
            "action": {"type": "authentication", "result": "success"},
        }
        features = extractor.transform(event)
        assert features.shape == (17,)

    def test_empty_nested_dicts(self, extractor: FeatureExtractor) -> None:
        """Test handling of empty nested dictionaries."""
        event = {
            "timestamp": "2024-01-15T12:00:00.000Z",
            "source": {},
            "actor": {},
            "action": {},
        }
        # Should not raise
        features = extractor.transform(event)
        assert features.shape == (17,)


# ============================================================================
# Geo Features Tests
# ============================================================================


class TestGeoFeatures:
    """Tests for geographic feature extraction."""

    def test_has_geo_info_with_geo(self, extractor: FeatureExtractor) -> None:
        """Test has_geo_info when geo data is present."""
        event = {
            "timestamp": "2024-01-15T12:00:00.000Z",
            "source": {"type": "auth", "hostname": "h", "ip": "10.0.0.1"},
            "actor": {
                "user": "u",
                "source_ip": "10.0.0.2",
                "geo": {"country": "US", "city": "NYC"},
            },
            "action": {"type": "authentication", "result": "success"},
        }
        features = extractor.transform(event)
        assert features[16] == 1.0  # has_geo_info

    def test_has_geo_info_without_geo(self, extractor: FeatureExtractor) -> None:
        """Test has_geo_info when geo data is absent."""
        event = {
            "timestamp": "2024-01-15T12:00:00.000Z",
            "source": {"type": "auth", "hostname": "h", "ip": "10.0.0.1"},
            "actor": {"user": "u", "source_ip": "10.0.0.2"},
            "action": {"type": "authentication", "result": "success"},
        }
        features = extractor.transform(event)
        assert features[16] == 0.0  # has_geo_info
