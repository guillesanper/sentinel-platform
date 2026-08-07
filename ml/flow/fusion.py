"""
Threat assessment for the flow-based track.

The flow counterpart of :mod:`ml.models.fusion`: it takes the Stage 1 anomaly
signal and the Stage 2 attack probabilities and turns them into a
``threat_score``, a ``severity``, an ``explanation`` and a list of
``recommended_actions``.

The weighted-ensemble skeleton is the same as the event track's, and the
severity scale is deliberately identical so that an analyst comparing a flow
alert with an event alert is comparing like with like. What is *not* shared is
the rule layer. The event rules read failed logins, business hours, sudo and
private IP ranges — a CICFlowMeter record has none of that. It has timing,
volume, a destination port and TCP flag counts, so those are what
:meth:`FlowFusionLayer._apply_rules` reads instead.

The module lives apart from ``ml.models.fusion`` for the same reason the models
do: the two tracks share no label space, and one import at the wrong site would
produce recommendations for the wrong taxonomy without raising anything.
"""

from collections.abc import Mapping
from typing import Any, Final

import structlog

from ml.flow.features import feature_mapping
from ml.flow.labels import BENIGN_LABEL, CLASS_ORDER

logger = structlog.get_logger()


# ============================================================================
# Severity
# ============================================================================

# Same cut points as the event track, on purpose: severity is the vocabulary an
# analyst triages by, and two scales with the same names and different meanings
# would be worse than no scale at all. Ordered high to low — _calculate_severity
# returns the first match.
FLOW_SEVERITY_THRESHOLDS: Final[dict[str, float]] = {
    "CRITICAL": 0.9,
    "HIGH": 0.7,
    "MEDIUM": 0.4,
    "LOW": 0.0,
}

# Attack type reported when Stage 1 is confident a flow is anomalous but Stage 2
# cannot name a family. It is not one of CLASS_ORDER: it is the honest answer to
# "something is off and we do not know what", which is the case worth escalating
# rather than the one worth hiding.
UNKNOWN_ANOMALY: Final[str] = "unknown_anomaly"

# Anomaly score above which an anomalous-but-BENIGN-looking flow is relabelled
# UNKNOWN_ANOMALY.
_UNKNOWN_ANOMALY_MIN_SCORE: Final[float] = 0.7

# Most explanations are noise past the first handful.
_MAX_EXPLANATIONS: Final[int] = 7
_MAX_RECOMMENDATIONS: Final[int] = 5


# ============================================================================
# Response playbooks
# ============================================================================

# One entry per canonical class of ml.flow.labels.CLASS_ORDER, plus the
# unknown-anomaly fallback. Completeness is asserted by the test suite: a class
# with no playbook silently degrades to the BENIGN advice, which reads as "no
# action required" for an attack.
FLOW_ATTACK_RECOMMENDATIONS: Final[dict[str, list[str]]] = {
    "BENIGN": [
        "No immediate action required",
        "Continue monitoring",
    ],
    "Botnet": [
        "Isolate the internal host from the network",
        "Block the external command-and-control endpoint at the egress firewall",
        "Hunt for the same destination across every other host",
        "Image the host before remediation — persistence is likely",
    ],
    "BruteForce": [
        "Rate-limit or temporarily block the source IP",
        "Enable account lockout on the targeted service",
        "Enforce MFA on the exposed service",
        "Review authentication logs for a successful login from this source",
    ],
    "DDoS": [
        "Engage upstream DDoS mitigation or scrubbing",
        "Apply connection rate limits at the edge",
        "Scale or shed load on the targeted service",
        "Collect source distribution for the abuse report",
    ],
    "DoS": [
        "Rate-limit the source IP at the edge",
        "Check the targeted service for resource exhaustion",
        "Tune connection and request timeouts",
        "Verify whether the service degraded or stayed available",
    ],
    "Scan": [
        "Monitor the source for follow-up exploitation attempts",
        "Verify which ports actually answered",
        "Confirm the exposed services are patched and intended to be reachable",
        "Add the source to the watchlist and correlate with later alerts",
    ],
    "WebAttack": [
        "Block the source IP at the WAF",
        "Review application logs for successful injection or XSS payloads",
        "Check the affected endpoint for input validation gaps",
        "Verify database and session integrity for the targeted application",
    ],
    UNKNOWN_ANOMALY: [
        "Capture the full flow for manual analysis",
        "Correlate with other traffic from the same host pair",
        "Monitor the source and destination closely",
    ],
}


# ============================================================================
# Rule layer signals
# ============================================================================

# Services whose exposure makes an anomalous flow materially more interesting.
_SENSITIVE_PORTS: Final[dict[int, str]] = {
    21: "FTP",
    22: "SSH",
    23: "Telnet",
    445: "SMB",
    1433: "MSSQL",
    3306: "MySQL",
    3389: "RDP",
    5432: "PostgreSQL",
}

# CICFlowMeter reports Flow Duration in MICROSECONDS. A sub-millisecond flow
# carrying almost no packets is the shape of a port-scan probe, not of a session.
_SHORT_FLOW_MICROSECONDS: Final[float] = 1_000.0
_SCAN_MAX_PACKETS: Final[float] = 4.0

# 10 MB/s sustained on a single flow is far above ordinary interactive traffic.
_HIGH_THROUGHPUT_BYTES_PER_SEC: Final[float] = 1e7


class FlowFusionLayer:
    """
    Combines Stage 1 and Stage 2 output into a flow threat assessment.

    Weighted ensemble, same shape as the event track:

    * anomaly score — 30% (unsupervised signal, Stage 1)
    * classification — 50% (supervised signal, Stage 2)
    * rules — 20% (flow-shaped domain heuristics)

    Attributes:
        anomaly_weight: Weight of the Stage 1 anomaly score.
        classification_weight: Weight of the Stage 2 attack probability.
        rules_weight: Weight of the rule layer.

    Example:
        >>> fusion = FlowFusionLayer()
        >>> assessment = fusion.combine(
        ...     anomaly_result={"score": 0.8, "is_anomaly": True},
        ...     classification_result={
        ...         "attack_type": "DDoS", "confidence": 0.9,
        ...         "probabilities": {"BENIGN": 0.05, "DDoS": 0.9},
        ...     },
        ...     flow={"features": {"Destination Port": 80}},
        ... )
        >>> assessment["severity"]
        'HIGH'
    """

    def __init__(
        self,
        anomaly_weight: float = 0.3,
        classification_weight: float = 0.5,
        rules_weight: float = 0.2,
    ) -> None:
        """
        Args:
            anomaly_weight: Weight for the Stage 1 signal (0-1).
            classification_weight: Weight for the Stage 2 signal (0-1).
            rules_weight: Weight for the rule layer (0-1).

        Note:
            Weights that do not sum to 1.0 are normalised, with a warning.
        """
        total = anomaly_weight + classification_weight + rules_weight
        if total <= 0:
            raise ValueError("Fusion weights must sum to a positive value.")

        if abs(total - 1.0) > 0.01:
            logger.warning(
                "flow_fusion_weights_not_normalized",
                total=total,
                message="Weights will be normalized to sum to 1.0",
            )
            anomaly_weight /= total
            classification_weight /= total
            rules_weight /= total

        self.anomaly_weight = anomaly_weight
        self.classification_weight = classification_weight
        self.rules_weight = rules_weight

        logger.info(
            "flow_fusion_layer_initialized",
            anomaly_weight=anomaly_weight,
            classification_weight=classification_weight,
            rules_weight=rules_weight,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def combine(
        self,
        anomaly_result: Mapping[str, Any],
        classification_result: Mapping[str, Any],
        flow: Mapping[str, Any],
    ) -> dict[str, Any]:
        """
        Merge both stages into a single assessment.

        Args:
            anomaly_result: Stage 1 output. Keys ``score`` (0-1, higher is more
                anomalous), ``is_anomaly`` and optionally ``confidence``.
            classification_result: Stage 2 output. Keys ``attack_type``,
                ``confidence`` and ``probabilities`` (canonical class → p).
            flow: The flow as received, i.e. a ``features`` mapping plus
                optional ``src_ip``/``dst_ip``/``dst_port``/``protocol``
                metadata. A flat feature mapping is accepted too.

        Returns:
            ``threat_score``, ``severity``, ``attack_type``, ``confidence``,
            ``is_anomaly``, ``explanation``, ``recommended_actions`` and a
            ``details`` breakdown of the three contributing scores.
        """
        anomaly_score = _clamp(float(anomaly_result.get("score", 0.0)))
        is_anomaly = bool(anomaly_result.get("is_anomaly", False))
        anomaly_confidence = float(anomaly_result.get("confidence", 0.5))

        attack_type = str(classification_result.get("attack_type", BENIGN_LABEL))
        classification_confidence = float(classification_result.get("confidence", 0.5))
        probabilities: dict[str, float] = dict(classification_result.get("probabilities", {}))

        features = feature_mapping(flow)
        classification_threat = _classification_threat(
            attack_type, classification_confidence, probabilities
        )
        rules_score, rules_explanations = self._apply_rules(features, attack_type)

        threat_score = _clamp(
            self.anomaly_weight * anomaly_score
            + self.classification_weight * classification_threat
            + self.rules_weight * rules_score
        )

        # Stage 1 flagged the flow, Stage 2 could not tell it apart from benign
        # traffic. Naming that state is more useful than reporting BENIGN.
        if (
            is_anomaly
            and attack_type == BENIGN_LABEL
            and anomaly_score > _UNKNOWN_ANOMALY_MIN_SCORE
        ):
            attack_type = UNKNOWN_ANOMALY

        severity = self._calculate_severity(threat_score)
        overall_confidence = self._calculate_confidence(
            anomaly_confidence,
            classification_confidence,
            anomaly_score,
            classification_threat,
        )
        explanations = self._generate_explanations(
            anomaly_result,
            classification_result,
            flow,
            rules_explanations,
        )
        recommended_actions = self._get_recommendations(attack_type, severity)

        result: dict[str, Any] = {
            "threat_score": round(threat_score, 4),
            "severity": severity,
            "attack_type": attack_type,
            "confidence": round(overall_confidence, 4),
            "is_anomaly": is_anomaly,
            "explanation": explanations,
            "recommended_actions": recommended_actions,
            "details": {
                "anomaly_score": round(anomaly_score, 4),
                "classification_threat": round(classification_threat, 4),
                "classification_confidence": round(classification_confidence, 4),
                "rules_score": round(rules_score, 4),
                "class_probabilities": {k: round(float(v), 4) for k, v in probabilities.items()},
            },
        }

        logger.debug(
            "flow_fusion_complete",
            threat_score=result["threat_score"],
            severity=severity,
            attack_type=attack_type,
        )
        return result

    def get_params(self) -> dict[str, float]:
        """Return the ensemble weights."""
        return {
            "anomaly_weight": self.anomaly_weight,
            "classification_weight": self.classification_weight,
            "rules_weight": self.rules_weight,
        }

    # ------------------------------------------------------------------
    # Rule layer
    # ------------------------------------------------------------------

    def _apply_rules(
        self,
        features: Mapping[str, Any],
        attack_type: str,
    ) -> tuple[float, list[str]]:
        """
        Score flow-shaped heuristics, independently of the models.

        Every signal here is read straight off the CICFlowMeter record: timing,
        volume, destination port and TCP flag counts. None of them is decisive
        on its own — the layer carries 20% of the weight — but together they are
        what lets an analyst see *why* a flow was escalated.

        Args:
            features: The flow's feature mapping.
            attack_type: Stage 2's verdict, used for the family-specific boost.

        Returns:
            ``(score, explanations)`` with the score clamped to [0, 1].
        """
        score = 0.0
        explanations: list[str] = []

        duration = _num(features, "Flow Duration")
        total_packets = _num(features, "Total Fwd Packets") + _num(
            features, "Total Backward Packets"
        )
        bytes_per_sec = _num(features, "Flow Bytes/s")
        dst_port = int(_num(features, "Destination Port"))
        syn = _num(features, "SYN Flag Count")
        ack = _num(features, "ACK Flag Count")
        rst = _num(features, "RST Flag Count")

        # Rule 1: a sub-millisecond flow carrying a handful of packets is a
        # probe, not a session.
        if 0 < duration < _SHORT_FLOW_MICROSECONDS and total_packets <= _SCAN_MAX_PACKETS:
            score += 0.2
            explanations.append(
                f"Very short flow ({duration:.0f} us) with {total_packets:.0f} packets — probe-like"
            )

        # Rule 2: a half-open handshake. SYN with no ACK anywhere in the flow is
        # the signature shared by SYN scans and SYN floods.
        if syn > 0 and ack == 0:
            score += 0.2
            explanations.append("SYN without ACK — half-open connection attempt")

        # Rule 3: the peer refused. On its own it is noise; alongside the rest it
        # is the tail of a scan.
        if rst > 0:
            score += 0.1
            explanations.append("Connection reset observed (RST flag set)")

        # Rule 4: sustained throughput well past interactive traffic.
        if bytes_per_sec > _HIGH_THROUGHPUT_BYTES_PER_SEC:
            score += 0.15
            explanations.append(f"Very high throughput ({bytes_per_sec / 1e6:.1f} MB/s)")

        # Rule 5: the destination is a service worth defending.
        service = _SENSITIVE_PORTS.get(dst_port)
        if service:
            score += 0.15
            explanations.append(f"Targets a sensitive service: {service} (port {dst_port})")

        # Rule 6: families whose consequences are worst if the call is right.
        if attack_type in ("Botnet", "WebAttack"):
            score += 0.2
            explanations.append(f"High-risk attack family: {attack_type}")
        elif attack_type in ("DDoS", "DoS", "BruteForce"):
            score += 0.1

        return min(1.0, score), explanations

    # ------------------------------------------------------------------
    # Scoring helpers
    # ------------------------------------------------------------------

    def _calculate_severity(self, threat_score: float) -> str:
        """Map a threat score to its severity band."""
        for severity, threshold in FLOW_SEVERITY_THRESHOLDS.items():
            if threat_score >= threshold:
                return severity
        return "LOW"

    def _calculate_confidence(
        self,
        anomaly_confidence: float,
        classification_confidence: float,
        anomaly_score: float,
        classification_threat: float,
    ) -> float:
        """
        How much to trust the assessment.

        Highest when both stages point the same way and both are individually
        confident; lowest when one calls the flow benign and the other does not.
        """
        agreement = 1.0 - abs(anomaly_score - classification_threat)
        avg_confidence = (
            self.anomaly_weight * anomaly_confidence
            + self.classification_weight * classification_confidence
        ) / (self.anomaly_weight + self.classification_weight)
        return _clamp(0.6 * avg_confidence + 0.4 * agreement)

    def _generate_explanations(
        self,
        anomaly_result: Mapping[str, Any],
        classification_result: Mapping[str, Any],
        flow: Mapping[str, Any],
        rules_explanations: list[str],
    ) -> list[str]:
        """Build the human-readable account of the assessment."""
        explanations: list[str] = []

        if anomaly_result.get("is_anomaly"):
            score = float(anomaly_result.get("score", 0.0))
            explanations.append(f"Flow deviates from benign traffic (anomaly score: {score:.2f})")

        attack_type = str(classification_result.get("attack_type", BENIGN_LABEL))
        if attack_type != BENIGN_LABEL:
            confidence = float(classification_result.get("confidence", 0.0))
            explanations.append(
                f"Traffic pattern matches {attack_type} ({confidence:.0%} confidence)"
            )

        explanations.extend(rules_explanations)

        # Metadata never reaches the model — the endpoints on both ends are not
        # features, and training on them would fit the lab topology. It is
        # exactly what an analyst needs to act, though, so it goes here.
        endpoints = _describe_endpoints(flow)
        if endpoints:
            explanations.append(endpoints)

        return explanations[:_MAX_EXPLANATIONS]

    def _get_recommendations(self, attack_type: str, severity: str) -> list[str]:
        """Pick the playbook for the family, escalated by severity."""
        recommendations = list(
            FLOW_ATTACK_RECOMMENDATIONS.get(
                attack_type,
                FLOW_ATTACK_RECOMMENDATIONS[UNKNOWN_ANOMALY],
            )
        )

        if severity == "CRITICAL":
            recommendations.insert(0, "IMMEDIATE: Escalate to the security team")
            recommendations.insert(1, "Consider blocking the source at the perimeter")
        elif severity == "HIGH":
            recommendations.insert(0, "Alert the security team within 15 minutes")

        return recommendations[:_MAX_RECOMMENDATIONS]


# ============================================================================
# Module-level helpers
# ============================================================================


def _classification_threat(
    attack_type: str,
    confidence: float,
    probabilities: Mapping[str, float],
) -> float:
    """
    Reduce Stage 2's output to a single "how likely is this an attack" number.

    Monotonically increasing in the probability of attack under all three
    branches, which is the property the threat score inherits and the test suite
    pins.

    Args:
        attack_type: The predicted class.
        confidence: Probability assigned to ``attack_type``.
        probabilities: Full class → probability mapping, when available.

    Returns:
        A value in [0, 1].
    """
    if BENIGN_LABEL in probabilities:
        # The direct reading: everything that is not benign.
        return _clamp(1.0 - float(probabilities[BENIGN_LABEL]))
    if attack_type == BENIGN_LABEL:
        return _clamp(1.0 - confidence)
    return _clamp(confidence)


def _describe_endpoints(flow: Mapping[str, Any]) -> str:
    """Format whatever endpoint metadata the payload carried, or ``""``."""
    src = flow.get("src_ip")
    dst = flow.get("dst_ip")
    port = flow.get("dst_port")
    protocol = flow.get("protocol")

    if src is None and dst is None:
        return ""

    target = f"{dst}:{port}" if dst is not None and port is not None else str(dst)
    described = f"Flow {src or 'unknown'} → {target}"
    return f"{described} ({protocol})" if protocol else described


def _num(features: Mapping[str, Any], name: str) -> float:
    """Read a numeric feature, defaulting to 0.0 when absent or unparseable."""
    try:
        return float(features.get(name, 0.0))
    except (TypeError, ValueError):
        return 0.0


def _clamp(value: float) -> float:
    """Constrain a score to [0, 1]."""
    return max(0.0, min(1.0, value))


# Guard against a class gaining a playbook-less entry in CLASS_ORDER. Cheap at
# import time and impossible to forget, unlike a comment.
_MISSING_PLAYBOOKS = [c for c in CLASS_ORDER if c not in FLOW_ATTACK_RECOMMENDATIONS]
if _MISSING_PLAYBOOKS:  # pragma: no cover - a code change is needed to reach it
    raise RuntimeError(
        f"FLOW_ATTACK_RECOMMENDATIONS is missing entries for {_MISSING_PLAYBOOKS}. "
        "A class without a playbook falls back to advice written for benign traffic."
    )
