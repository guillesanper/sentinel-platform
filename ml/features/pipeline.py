"""
Feature extraction pipeline for security events.

Transforms raw security events into numerical feature vectors
for ML model consumption.
"""

import ipaddress
from datetime import datetime
from typing import Any, ClassVar

import numpy as np
import structlog

logger = structlog.get_logger()

# ============================================================================
# Constants
# ============================================================================

# Known Tor exit node ranges (sample - in production, use updated list)
TOR_EXIT_NODES = {
    "185.220.100.",
    "185.220.101.",
    "185.220.102.",
    "192.42.116.",
    "51.15.",
    "162.247.74.",
}

# Private IP ranges
PRIVATE_RANGES = [
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("127.0.0.0/8"),
]

# Privileged actions
PRIVILEGED_ACTIONS = {
    "sudo",
    "su",
    "root_login",
    "privilege_escalation",
    "admin_access",
    "chmod",
    "chown",
    "passwd",
    "useradd",
    "userdel",
    "groupadd",
    "visudo",
}

# Source type encoding
SOURCE_TYPE_MAP = {
    "auth": 0,
    "system": 1,
    "network": 2,
    "app": 3,
    "application": 3,
}

# Action type encoding
ACTION_TYPE_MAP = {
    "authentication": 0,
    "authorization": 1,
    "file_access": 2,
    "network_connection": 3,
    "process_execution": 4,
    "configuration_change": 5,
    "data_transfer": 6,
    "unknown": 7,
}


# ============================================================================
# Feature Extractor
# ============================================================================


class FeatureExtractor:
    """
    Extract numerical features from security events.

    Features extracted:
    - Temporal: hour_of_day, day_of_week, is_weekend, is_night, is_business_hours
    - IP-based: ip_is_internal, ip_is_tor, ip_octets
    - Action-based: action_is_privileged, action_is_failure, source_type, action_type

    Example:
        >>> extractor = FeatureExtractor()
        >>> event = {
        ...     "timestamp": "2024-01-15T14:32:18.123Z",
        ...     "source": {"type": "auth", "hostname": "web-01", "ip": "10.0.1.5"},
        ...     "actor": {"user": "admin", "source_ip": "203.0.113.45"},
        ...     "action": {"type": "authentication", "result": "failure"}
        ... }
        >>> features = extractor.transform(event)
        >>> features.shape
        (17,)
    """

    # Feature names for interpretability
    FEATURE_NAMES: ClassVar[list[str]] = [
        "hour_of_day",
        "day_of_week",
        "is_weekend",
        "is_night",
        "is_business_hours",
        "source_type",
        "action_type",
        "action_is_privileged",
        "action_is_failure",
        "actor_ip_is_internal",
        "actor_ip_is_tor",
        "actor_ip_octet_1",
        "actor_ip_octet_2",
        "actor_ip_octet_3",
        "actor_ip_octet_4",
        "source_ip_is_internal",
        "has_geo_info",
    ]

    def __init__(self) -> None:
        """Initialize the feature extractor."""
        self.n_features = len(self.FEATURE_NAMES)
        logger.info("feature_extractor_initialized", n_features=self.n_features)

    def transform(self, event: dict[str, Any]) -> np.ndarray:
        """
        Transform a single event into a feature vector.

        Args:
            event: Security event dictionary.

        Returns:
            numpy array of shape (n_features,) with extracted features.
        """
        features = []

        # Temporal features
        temporal = self._extract_temporal_features(event)
        features.extend(temporal)

        # Source and action type features
        source_action = self._extract_source_action_features(event)
        features.extend(source_action)

        # IP-based features
        ip_features = self._extract_ip_features(event)
        features.extend(ip_features)

        # Geo features
        geo_features = self._extract_geo_features(event)
        features.extend(geo_features)

        return np.array(features, dtype=np.float32)

    def transform_batch(self, events: list[dict[str, Any]]) -> np.ndarray:
        """
        Transform multiple events into a feature matrix.

        Args:
            events: List of security event dictionaries.

        Returns:
            numpy array of shape (n_events, n_features).
        """
        return np.array([self.transform(event) for event in events], dtype=np.float32)

    def _extract_temporal_features(self, event: dict[str, Any]) -> list[float]:
        """Extract time-based features."""
        timestamp = event.get("timestamp")

        if timestamp is None:
            # Default to current time if missing
            dt = datetime.utcnow()
        elif isinstance(timestamp, str):
            # Parse ISO format
            try:
                dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            except ValueError:
                dt = datetime.utcnow()
        elif isinstance(timestamp, datetime):
            dt = timestamp
        else:
            dt = datetime.utcnow()

        hour = dt.hour
        day_of_week = dt.weekday()  # 0=Monday, 6=Sunday

        # Derived temporal features
        is_weekend = 1.0 if day_of_week >= 5 else 0.0
        is_night = 1.0 if hour < 6 or hour >= 22 else 0.0
        is_business_hours = 1.0 if 9 <= hour < 17 and day_of_week < 5 else 0.0

        return [
            float(hour) / 23.0,  # Normalize to [0, 1]
            float(day_of_week) / 6.0,  # Normalize to [0, 1]
            is_weekend,
            is_night,
            is_business_hours,
        ]

    def _extract_source_action_features(self, event: dict[str, Any]) -> list[float]:
        """Extract source and action type features."""
        source = event.get("source", {})
        action = event.get("action", {})

        # Source type
        source_type = source.get("type", "unknown").lower()
        source_type_encoded = SOURCE_TYPE_MAP.get(source_type, len(SOURCE_TYPE_MAP))

        # Action type
        action_type = action.get("type", "unknown").lower()
        action_type_encoded = ACTION_TYPE_MAP.get(action_type, ACTION_TYPE_MAP["unknown"])

        # Action subtype for privileged detection
        action_subtype = action.get("subtype", "").lower()
        is_privileged = (
            1.0
            if any(priv in action_type or priv in action_subtype for priv in PRIVILEGED_ACTIONS)
            else 0.0
        )

        # Action result
        result = action.get("result", "unknown").lower()
        is_failure = 1.0 if result == "failure" else 0.0

        return [
            float(source_type_encoded) / max(len(SOURCE_TYPE_MAP), 1),
            float(action_type_encoded) / max(len(ACTION_TYPE_MAP), 1),
            is_privileged,
            is_failure,
        ]

    def _extract_ip_features(self, event: dict[str, Any]) -> list[float]:
        """Extract IP address features."""
        actor = event.get("actor", {})
        source = event.get("source", {})

        # Actor's source IP
        actor_ip_str = actor.get("source_ip", "0.0.0.0")
        actor_ip_internal, actor_ip_tor, actor_octets = self._analyze_ip(actor_ip_str)

        # Source system IP
        source_ip_str = source.get("ip", "0.0.0.0")
        source_ip_internal, _, _ = self._analyze_ip(source_ip_str)

        return [
            actor_ip_internal,
            actor_ip_tor,
            *actor_octets,
            source_ip_internal,
        ]

    def _analyze_ip(self, ip_str: str) -> tuple[float, float, list[float]]:
        """
        Analyze an IP address for various features.

        Returns:
            Tuple of (is_internal, is_tor, [octet1, octet2, octet3, octet4])
        """
        try:
            ip = ipaddress.ip_address(ip_str)

            # Check if internal
            is_internal = 1.0 if any(ip in net for net in PRIVATE_RANGES) else 0.0

            # Check if Tor exit node (simple prefix check)
            is_tor = 1.0 if any(ip_str.startswith(prefix) for prefix in TOR_EXIT_NODES) else 0.0

            # Extract octets (normalized)
            if isinstance(ip, ipaddress.IPv4Address):
                octets = [float(int(o)) / 255.0 for o in ip_str.split(".")]
            else:
                # For IPv6, use first 4 bytes
                octets = [float(b) / 255.0 for b in ip.packed[:4]]

            return is_internal, is_tor, octets

        except ValueError:
            # Invalid IP
            return 0.0, 0.0, [0.0, 0.0, 0.0, 0.0]

    def _extract_geo_features(self, event: dict[str, Any]) -> list[float]:
        """Extract geographic features."""
        actor = event.get("actor", {})
        geo = actor.get("geo", {})

        has_geo = 1.0 if geo and (geo.get("country") or geo.get("lat")) else 0.0

        return [has_geo]

    def get_feature_names(self) -> list[str]:
        """Return list of feature names for interpretability."""
        return self.FEATURE_NAMES.copy()

    def get_feature_importance_context(
        self,
        features: np.ndarray,
        top_k: int = 5,
    ) -> list[str]:
        """
        Get human-readable context for top features.

        Args:
            features: Feature vector.
            top_k: Number of top features to return.

        Returns:
            List of human-readable feature descriptions.
        """
        descriptions = []

        # Map feature indices to descriptions
        if features[3] > 0.5:  # is_night
            descriptions.append("Activity during night hours (22:00-06:00)")

        if features[4] < 0.5 and features[2] < 0.5:  # not business hours, not weekend
            descriptions.append("Activity outside business hours")

        if features[7] > 0.5:  # action_is_privileged
            descriptions.append("Privileged action attempted")

        if features[8] > 0.5:  # action_is_failure
            descriptions.append("Action resulted in failure")

        if features[10] > 0.5:  # actor_ip_is_tor
            descriptions.append("Source IP is known Tor exit node")

        if features[9] < 0.5:  # actor_ip_is_internal (not internal = external)
            descriptions.append("Connection from external IP address")

        return descriptions[:top_k]


# ============================================================================
# Combined Feature Extractor (Event + Window Features)
# ============================================================================


class CombinedFeatureExtractor:
    """
    Combined extractor that concatenates event features with window features.

    Produces 25 total features:
    - 17 from FeatureExtractor (per-event features)
    - 8 from WindowFeatureExtractor (aggregate features over event groups)

    This is essential for distinguishing attacks that differ in group behavior:
    - brute_force: unique_users_from_ip ~ 0.02 (1 user)
    - credential_stuffing: unique_users_from_ip > 0.5 (many users)

    Example:
        >>> extractor = CombinedFeatureExtractor()
        >>> # For training with campaigns (groups of events):
        >>> events_group = [event1, event2, event3]  # same source_ip
        >>> features = extractor.transform_group(events_group)
        >>> features.shape
        (25,)

        >>> # For single event inference:
        >>> features = extractor.transform(single_event)
        >>> features.shape
        (25,)
    """

    def __init__(self) -> None:
        """Initialize the combined feature extractor."""
        from ml.features.window_features import WindowFeatureExtractor

        self.event_extractor = FeatureExtractor()
        self.window_extractor = WindowFeatureExtractor()
        self.n_features = self.event_extractor.n_features + self.window_extractor.n_features

        logger.info(
            "combined_feature_extractor_initialized",
            event_features=self.event_extractor.n_features,
            window_features=self.window_extractor.n_features,
            total_features=self.n_features,
        )

    def transform(self, event: dict[str, Any]) -> np.ndarray:
        """
        Transform a single event into combined features.

        Uses default window features suitable for real-time inference.

        Args:
            event: Security event dictionary.

        Returns:
            numpy array of shape (25,) with event + window features.
        """
        event_features = self.event_extractor.transform(event)
        window_features = self.window_extractor.transform_single_event(event)

        return np.concatenate([event_features, window_features])

    def transform_group(self, events: list[dict[str, Any]]) -> np.ndarray:
        """
        Transform a group of events into combined features.

        Uses the first event as representative for event features,
        and all events for window features.

        Args:
            events: List of security event dictionaries (same source_ip).

        Returns:
            numpy array of shape (25,) with event + window features.
        """
        if not events:
            return np.zeros(self.n_features, dtype=np.float32)

        # Use first event as representative for event-level features
        representative_event = events[0]
        event_features = self.event_extractor.transform(representative_event)

        # Use all events for window features
        window_features = self.window_extractor.transform(events)

        return np.concatenate([event_features, window_features])

    def transform_batch_groups(
        self,
        event_groups: list[list[dict[str, Any]]],
    ) -> np.ndarray:
        """
        Transform multiple groups of events into a feature matrix.

        Args:
            event_groups: List of event groups, where each group contains
                          events from the same campaign/source_ip.

        Returns:
            numpy array of shape (n_groups, 25).
        """
        return np.array(
            [self.transform_group(group) for group in event_groups],
            dtype=np.float32,
        )

    def get_feature_names(self) -> list[str]:
        """Return list of all feature names."""
        return self.event_extractor.get_feature_names() + self.window_extractor.get_feature_names()
