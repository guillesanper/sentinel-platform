"""
Tests for window feature extraction.

Validates that window features correctly distinguish between:
- brute_force: SAME user, SAME IP (low unique_users_from_ip)
- credential_stuffing: MULTIPLE users, SAME IP (high unique_users_from_ip)
"""

import pytest

from ml.features.window_features import WindowFeatureExtractor


class TestWindowFeatureExtractor:
    """Tests for WindowFeatureExtractor."""

    @pytest.fixture
    def extractor(self) -> WindowFeatureExtractor:
        """Create a WindowFeatureExtractor instance."""
        return WindowFeatureExtractor()

    def test_feature_names(self, extractor: WindowFeatureExtractor) -> None:
        """Test that feature names are correctly defined."""
        names = extractor.get_feature_names()
        assert len(names) == 8
        assert "unique_users_from_ip" in names
        assert "attempts_per_user" in names
        assert "failure_rate" in names
        assert "user_entropy" in names

    def test_empty_events(self, extractor: WindowFeatureExtractor) -> None:
        """Test handling of empty event list."""
        features = extractor.transform([])
        assert features.shape == (8,)
        assert all(f == 0.0 for f in features)

    def test_brute_force_pattern(self, extractor: WindowFeatureExtractor) -> None:
        """Test that brute_force pattern has low unique_users_from_ip."""
        # Brute force: SAME user, many failures
        events = [
            {
                "actor": {"user": "admin", "source_ip": "1.2.3.4"},
                "action": {"result": "failure"},
                "timestamp": f"2024-01-15T10:00:{i:02d}Z",
            }
            for i in range(20)
        ]

        features = extractor.transform(events)

        # unique_users_from_ip should be low (1 user / 50 = 0.02)
        unique_users_idx = 0
        assert features[unique_users_idx] == pytest.approx(1 / 50, rel=0.01)

        # failure_rate should be high (100%)
        failure_rate_idx = 2
        assert features[failure_rate_idx] == pytest.approx(1.0, rel=0.01)

        # user_entropy should be 0 (only one user)
        entropy_idx = 4
        assert features[entropy_idx] == pytest.approx(0.0, rel=0.01)

    def test_credential_stuffing_pattern(self, extractor: WindowFeatureExtractor) -> None:
        """Test that credential_stuffing pattern has high unique_users_from_ip."""
        # Credential stuffing: MULTIPLE users, mix of failures and successes
        users = [f"user_{i}" for i in range(30)]
        events = []

        for i, user in enumerate(users):
            result = "success" if i % 10 == 0 else "failure"
            events.append(
                {
                    "actor": {"user": user, "source_ip": "1.2.3.4"},
                    "action": {"result": result},
                    "timestamp": f"2024-01-15T10:00:{i:02d}Z",
                }
            )

        features = extractor.transform(events)

        # unique_users_from_ip should be high (30 users / 50 = 0.6)
        unique_users_idx = 0
        assert features[unique_users_idx] == pytest.approx(30 / 50, rel=0.01)

        # failure_rate should be ~90%
        failure_rate_idx = 2
        assert features[failure_rate_idx] == pytest.approx(0.9, rel=0.05)

        # user_entropy should be > 0 (multiple users)
        entropy_idx = 4
        assert features[entropy_idx] > 0.5  # Normalized by 5

        # has_success_after_failures should be 1.0
        success_after_failures_idx = 6
        assert features[success_after_failures_idx] == 1.0

    def test_brute_force_vs_credential_stuffing_discrimination(
        self, extractor: WindowFeatureExtractor
    ) -> None:
        """
        Critical test: verify that the key discriminating feature
        (unique_users_from_ip) correctly separates attack types.
        """
        # Brute force campaign
        bf_events = [
            {
                "actor": {"user": "victim", "source_ip": "1.2.3.4"},
                "action": {"result": "failure"},
                "timestamp": f"2024-01-15T10:00:{i:02d}Z",
            }
            for i in range(50)
        ]

        # Credential stuffing campaign
        cs_events = [
            {
                "actor": {"user": f"user_{i}", "source_ip": "1.2.3.4"},
                "action": {"result": "failure" if i % 8 != 0 else "success"},
                "timestamp": f"2024-01-15T10:00:{i:02d}Z",
            }
            for i in range(50)
        ]

        bf_features = extractor.transform(bf_events)
        cs_features = extractor.transform(cs_events)

        unique_users_idx = 0

        # Brute force: 1 user -> 0.02
        assert bf_features[unique_users_idx] < 0.05

        # Credential stuffing: 50 users -> 1.0
        assert cs_features[unique_users_idx] > 0.5

        # The difference should be substantial
        assert cs_features[unique_users_idx] > bf_features[unique_users_idx] * 10

    def test_max_consecutive_failures(self, extractor: WindowFeatureExtractor) -> None:
        """Test max consecutive failures calculation."""
        events = [
            {
                "actor": {"user": "a"},
                "action": {"result": "failure"},
                "timestamp": "2024-01-15T10:00:00Z",
            },
            {
                "actor": {"user": "a"},
                "action": {"result": "failure"},
                "timestamp": "2024-01-15T10:00:01Z",
            },
            {
                "actor": {"user": "a"},
                "action": {"result": "failure"},
                "timestamp": "2024-01-15T10:00:02Z",
            },
            {
                "actor": {"user": "a"},
                "action": {"result": "success"},
                "timestamp": "2024-01-15T10:00:03Z",
            },
            {
                "actor": {"user": "a"},
                "action": {"result": "failure"},
                "timestamp": "2024-01-15T10:00:04Z",
            },
        ]

        features = extractor.transform(events)

        # max_consecutive_failures = 3, normalized by 50 = 0.06
        max_consec_idx = 5
        assert features[max_consec_idx] == pytest.approx(3 / 50, rel=0.01)

    def test_single_event_defaults(self, extractor: WindowFeatureExtractor) -> None:
        """Test single event transformation with conservative defaults."""
        event = {
            "actor": {"user": "test", "source_ip": "1.2.3.4"},
            "action": {"result": "failure"},
            "timestamp": "2024-01-15T10:00:00Z",
        }

        features = extractor.transform_single_event(event)

        assert features.shape == (8,)
        # Default unique_users should be low (1/50)
        assert features[0] == pytest.approx(0.02, rel=0.01)
        # failure_rate should reflect the single event
        assert features[2] == 1.0  # failure


class TestCombinedFeatureExtractor:
    """Tests for CombinedFeatureExtractor."""

    def test_combined_features_shape(self) -> None:
        """Test that combined features have correct shape (25)."""
        from ml.features.pipeline import CombinedFeatureExtractor

        extractor = CombinedFeatureExtractor()

        event = {
            "timestamp": "2024-01-15T14:32:18.123Z",
            "source": {"type": "auth", "hostname": "web-01", "ip": "10.0.1.5"},
            "actor": {"user": "admin", "source_ip": "203.0.113.45"},
            "action": {"type": "authentication", "result": "failure"},
        }

        features = extractor.transform(event)
        assert features.shape == (25,)

    def test_combined_group_features(self) -> None:
        """Test group transformation with window features."""
        from ml.features.pipeline import CombinedFeatureExtractor

        extractor = CombinedFeatureExtractor()

        events = [
            {
                "timestamp": f"2024-01-15T10:00:{i:02d}Z",
                "source": {"type": "auth", "hostname": "web-01", "ip": "10.0.1.5"},
                "actor": {"user": f"user_{i}", "source_ip": "203.0.113.45"},
                "action": {"type": "authentication", "result": "failure"},
            }
            for i in range(10)
        ]

        features = extractor.transform_group(events)
        assert features.shape == (25,)

        # First 17 features from event extractor
        # Last 8 features from window extractor
        # unique_users_from_ip should reflect 10 users
        unique_users_idx = 17  # First window feature
        assert features[unique_users_idx] == pytest.approx(10 / 50, rel=0.01)

    def test_feature_names_combined(self) -> None:
        """Test that combined feature names include both extractors."""
        from ml.features.pipeline import CombinedFeatureExtractor

        extractor = CombinedFeatureExtractor()
        names = extractor.get_feature_names()

        assert len(names) == 25
        assert "hour_of_day" in names  # From event extractor
        assert "unique_users_from_ip" in names  # From window extractor
