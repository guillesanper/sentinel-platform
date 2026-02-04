"""
Window-based feature extraction for security event analysis.

Extracts aggregate features from groups of events sharing the same source IP,
enabling detection of attack patterns like brute_force vs credential_stuffing.
"""

import math
from typing import Any, ClassVar

import numpy as np
import structlog

logger = structlog.get_logger()


class WindowFeatureExtractor:
    """
    Extract window-based features from groups of security events.

    These features capture patterns across multiple events from the same
    source IP, which is critical for distinguishing:
    - brute_force: SAME user, SAME IP, many failures
    - credential_stuffing: MULTIPLE users, SAME IP, some successes

    Features extracted:
    - unique_users_from_ip: Number of distinct users / 50 (normalized)
    - attempts_per_user: Total events / unique users / 100
    - failure_rate: Failures / total events
    - event_rate_per_minute: Events / window duration / 200
    - user_entropy: Shannon entropy of user distribution / 5
    - max_consecutive_failures: Longest streak of failures / 50
    - has_success_after_failures: 1.0 if success follows failures
    - time_spread_seconds: (max_ts - min_ts) / 1800

    Example:
        >>> extractor = WindowFeatureExtractor()
        >>> events = [
        ...     {"actor": {"user": "admin", "source_ip": "1.2.3.4"},
        ...      "action": {"result": "failure"}, "timestamp": "2024-01-15T10:00:00Z"},
        ...     {"actor": {"user": "admin", "source_ip": "1.2.3.4"},
        ...      "action": {"result": "failure"}, "timestamp": "2024-01-15T10:00:05Z"},
        ... ]
        >>> features = extractor.transform(events)
        >>> features.shape
        (8,)
    """

    FEATURE_NAMES: ClassVar[list[str]] = [
        "unique_users_from_ip",
        "attempts_per_user",
        "failure_rate",
        "event_rate_per_minute",
        "user_entropy",
        "max_consecutive_failures",
        "has_success_after_failures",
        "time_spread_seconds",
    ]

    def __init__(self) -> None:
        """Initialize the window feature extractor."""
        self.n_features = len(self.FEATURE_NAMES)
        logger.info("window_feature_extractor_initialized", n_features=self.n_features)

    def transform(self, events: list[dict[str, Any]]) -> np.ndarray:
        """
        Transform a group of events into window-based features.

        Args:
            events: List of security event dictionaries from the same source IP.

        Returns:
            numpy array of shape (n_features,) with extracted window features.
        """
        if not events:
            return np.zeros(self.n_features, dtype=np.float32)

        # Extract users and results
        users: list[str] = []
        results: list[str] = []
        timestamps: list[float] = []

        for event in events:
            actor = event.get("actor", {})
            action = event.get("action", {})

            user = actor.get("user", "unknown")
            users.append(user)

            result = action.get("result", "unknown")
            results.append(result)

            ts = self._parse_timestamp(event.get("timestamp"))
            timestamps.append(ts)

        n_events = len(events)
        unique_users = set(users)
        n_unique_users = len(unique_users)

        # Feature 1: unique_users_from_ip (normalized by 50)
        unique_users_from_ip = n_unique_users / 50.0

        # Feature 2: attempts_per_user (normalized by 100)
        attempts_per_user = (n_events / max(n_unique_users, 1)) / 100.0

        # Feature 3: failure_rate
        n_failures = sum(1 for r in results if r == "failure")
        failure_rate = n_failures / max(n_events, 1)

        # Feature 4: event_rate_per_minute (normalized by 200)
        time_spread = max(timestamps) - min(timestamps) if timestamps else 0
        minutes = max(time_spread / 60.0, 1 / 60.0)  # At least 1 second
        event_rate_per_minute = (n_events / minutes) / 200.0

        # Feature 5: user_entropy (Shannon entropy, normalized by 5)
        user_entropy = self._calculate_entropy(users) / 5.0

        # Feature 6: max_consecutive_failures (normalized by 50)
        max_consecutive_failures = self._max_consecutive(results, "failure") / 50.0

        # Feature 7: has_success_after_failures
        has_success_after_failures = self._has_success_after_failures(results)

        # Feature 8: time_spread_seconds (normalized by 1800 = 30 min)
        time_spread_normalized = time_spread / 1800.0

        features = [
            unique_users_from_ip,
            attempts_per_user,
            failure_rate,
            event_rate_per_minute,
            user_entropy,
            max_consecutive_failures,
            has_success_after_failures,
            time_spread_normalized,
        ]

        return np.array(features, dtype=np.float32)

    def transform_single_event(self, event: dict[str, Any]) -> np.ndarray:
        """
        Transform a single event with default window features.

        Used for real-time inference when only one event is available.
        Returns conservative defaults that don't strongly indicate any attack type.

        Args:
            event: Single security event dictionary.

        Returns:
            numpy array of shape (n_features,) with default window features.
        """
        action = event.get("action", {})

        result = action.get("result", "unknown")
        is_failure = 1.0 if result == "failure" else 0.0

        # Conservative defaults for single event
        return np.array(
            [
                0.02,  # unique_users_from_ip: assume 1 user / 50
                0.01,  # attempts_per_user: 1 attempt / 100
                is_failure,  # failure_rate: based on this event
                0.005,  # event_rate_per_minute: 1 event / 200
                0.0,  # user_entropy: 0 for single user
                is_failure / 50.0,  # max_consecutive_failures: 0 or 1
                0.0,  # has_success_after_failures: unknown
                0.0,  # time_spread_seconds: 0 for single event
            ],
            dtype=np.float32,
        )

    def _parse_timestamp(self, ts: Any) -> float:
        """Parse timestamp to Unix epoch seconds."""
        if ts is None:
            return 0.0

        if isinstance(ts, int | float):
            return float(ts)

        if isinstance(ts, str):
            from datetime import datetime

            try:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                return dt.timestamp()
            except ValueError:
                return 0.0

        return 0.0

    def _calculate_entropy(self, items: list[str]) -> float:
        """Calculate Shannon entropy of a list of items."""
        if not items:
            return 0.0

        from collections import Counter

        counts = Counter(items)
        n = len(items)

        entropy = 0.0
        for count in counts.values():
            if count > 0:
                p = count / n
                entropy -= p * math.log2(p)

        return entropy

    def _max_consecutive(self, items: list[str], target: str) -> int:
        """Find maximum consecutive occurrences of target value."""
        max_count = 0
        current_count = 0

        for item in items:
            if item == target:
                current_count += 1
                max_count = max(max_count, current_count)
            else:
                current_count = 0

        return max_count

    def _has_success_after_failures(self, results: list[str]) -> float:
        """Check if there's a success after one or more failures."""
        seen_failure = False
        for result in results:
            if result == "failure":
                seen_failure = True
            elif result == "success" and seen_failure:
                return 1.0
        return 0.0

    def get_feature_names(self) -> list[str]:
        """Return list of feature names for interpretability."""
        return self.FEATURE_NAMES.copy()
