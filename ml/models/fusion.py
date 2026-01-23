"""
Fusion layer for combining anomaly detection and classification results.

Implements an ensemble approach that weighs anomaly scores, classification
confidence, and rule-based heuristics to produce final threat assessments.
"""

from typing import Any

import structlog

logger = structlog.get_logger()

# Severity thresholds
SEVERITY_THRESHOLDS = {
    "CRITICAL": 0.9,
    "HIGH": 0.7,
    "MEDIUM": 0.4,
    "LOW": 0.0,
}

# Attack-specific action recommendations
ATTACK_RECOMMENDATIONS = {
    "brute_force": [
        "Block source IP temporarily (15-30 minutes)",
        "Enable account lockout after 5 failed attempts",
        "Enable MFA for the target account",
        "Review authentication logs for the target user",
    ],
    "credential_stuffing": [
        "Implement CAPTCHA on login form",
        "Block source IP range",
        "Enable rate limiting on authentication endpoint",
        "Check if user credentials appear in known breaches",
    ],
    "privilege_escalation": [
        "Immediately revoke elevated privileges",
        "Audit all actions performed with elevated access",
        "Review sudo/admin group membership",
        "Check for unauthorized SSH keys or certificates",
    ],
    "lateral_movement": [
        "Isolate affected systems from network",
        "Review network flow logs for data exfiltration",
        "Check for persistence mechanisms",
        "Scan connected systems for compromise indicators",
    ],
    "data_exfiltration": [
        "Block outbound connections from source",
        "Identify what data was accessed/transferred",
        "Review DLP logs and file access patterns",
        "Preserve forensic evidence",
    ],
    "reconnaissance": [
        "Monitor source IP for follow-up attacks",
        "Review firewall logs for blocked attempts",
        "Check for successful service enumeration",
        "Update intrusion detection signatures",
    ],
    "normal": [
        "No immediate action required",
        "Continue monitoring",
    ],
}


class FusionLayer:
    """
    Combines results from anomaly detection and attack classification.

    Uses a weighted ensemble approach:
    - Anomaly score: 30% weight (unsupervised signal)
    - Classification confidence: 50% weight (supervised signal)
    - Rule-based heuristics: 20% weight (domain knowledge)

    Attributes:
        anomaly_weight: Weight for anomaly detection score.
        classification_weight: Weight for classification confidence.
        rules_weight: Weight for rule-based adjustments.

    Example:
        >>> fusion = FusionLayer()
        >>> result = fusion.combine(
        ...     anomaly_result={"score": 0.8, "is_anomaly": True},
        ...     classification_result={"attack_type": "brute_force", "confidence": 0.9},
        ...     event={"action": {"result": "failure"}}
        ... )
        >>> print(f"Threat score: {result['threat_score']:.2f}")
    """

    def __init__(
        self,
        anomaly_weight: float = 0.3,
        classification_weight: float = 0.5,
        rules_weight: float = 0.2,
    ) -> None:
        """
        Initialize the fusion layer.

        Args:
            anomaly_weight: Weight for anomaly detection (0-1).
            classification_weight: Weight for classification (0-1).
            rules_weight: Weight for rule-based scoring (0-1).

        Note:
            Weights should sum to 1.0 for proper normalization.
        """
        total = anomaly_weight + classification_weight + rules_weight
        if abs(total - 1.0) > 0.01:
            logger.warning(
                "fusion_weights_not_normalized",
                total=total,
                message="Weights will be normalized to sum to 1.0",
            )
            # Normalize weights
            anomaly_weight /= total
            classification_weight /= total
            rules_weight /= total

        self.anomaly_weight = anomaly_weight
        self.classification_weight = classification_weight
        self.rules_weight = rules_weight

        logger.info(
            "fusion_layer_initialized",
            anomaly_weight=anomaly_weight,
            classification_weight=classification_weight,
            rules_weight=rules_weight,
        )

    def combine(
        self,
        anomaly_result: dict[str, Any],
        classification_result: dict[str, Any],
        event: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Combine model outputs into a final threat assessment.

        Args:
            anomaly_result: Output from AnomalyDetector.predict()
                Expected keys: score, is_anomaly, confidence
            classification_result: Output from AttackClassifier.predict()
                Expected keys: attack_type, confidence, probabilities
            event: Original security event dictionary.

        Returns:
            Dictionary containing:
            - threat_score: Combined threat score (0-1)
            - severity: Severity level (LOW/MEDIUM/HIGH/CRITICAL)
            - attack_type: Detected attack type
            - confidence: Overall confidence in assessment
            - is_anomaly: Whether behavior is anomalous
            - explanation: List of human-readable explanations
            - recommended_actions: List of suggested responses
        """
        # Extract scores
        anomaly_score = anomaly_result.get("score", 0.0)
        is_anomaly = anomaly_result.get("is_anomaly", False)
        anomaly_confidence = anomaly_result.get("confidence", 0.5)

        attack_type = classification_result.get("attack_type", "normal")
        classification_confidence = classification_result.get("confidence", 0.5)
        class_probabilities = classification_result.get("probabilities", {})

        # Calculate rule-based score
        rules_score, rules_explanations = self._apply_rules(event, attack_type)

        # Combine scores with weights
        # For classification, use (1 - normal_probability) as threat indicator
        normal_prob = class_probabilities.get("normal", 0.5)
        classification_threat = 1.0 - normal_prob if attack_type == "normal" else classification_confidence

        threat_score = (
            self.anomaly_weight * anomaly_score
            + self.classification_weight * classification_threat
            + self.rules_weight * rules_score
        )

        # Clamp to [0, 1]
        threat_score = max(0.0, min(1.0, threat_score))

        # Determine severity
        severity = self._calculate_severity(threat_score)

        # Override attack type if anomaly but classified as normal
        if is_anomaly and attack_type == "normal" and anomaly_score > 0.7:
            attack_type = "unknown_anomaly"

        # Calculate overall confidence
        overall_confidence = self._calculate_confidence(
            anomaly_confidence,
            classification_confidence,
            anomaly_score,
            classification_threat,
        )

        # Generate explanations
        explanations = self._generate_explanations(
            anomaly_result,
            classification_result,
            event,
            rules_explanations,
        )

        # Get recommended actions
        recommended_actions = self._get_recommendations(
            attack_type,
            severity,
            threat_score,
        )

        result = {
            "threat_score": round(threat_score, 4),
            "severity": severity,
            "attack_type": attack_type,
            "confidence": round(overall_confidence, 4),
            "is_anomaly": is_anomaly,
            "explanation": explanations,
            "recommended_actions": recommended_actions,
            "details": {
                "anomaly_score": round(anomaly_score, 4),
                "classification_confidence": round(classification_confidence, 4),
                "rules_score": round(rules_score, 4),
                "class_probabilities": {
                    k: round(v, 4) for k, v in class_probabilities.items()
                },
            },
        }

        logger.debug(
            "fusion_complete",
            threat_score=threat_score,
            severity=severity,
            attack_type=attack_type,
        )

        return result

    def _apply_rules(
        self,
        event: dict[str, Any],
        attack_type: str,
    ) -> tuple[float, list[str]]:
        """
        Apply rule-based heuristics to adjust threat score.

        Returns:
            Tuple of (rules_score, explanations)
        """
        score = 0.0
        explanations = []

        action = event.get("action", {})
        actor = event.get("actor", {})
        source = event.get("source", {})

        # Rule 1: Failed actions are more suspicious
        if action.get("result") == "failure":
            score += 0.2
            explanations.append("Action resulted in failure")

        # Rule 2: Actions outside business hours
        # (Assuming feature extractor already computed this)
        timestamp = event.get("timestamp")
        if timestamp:
            from datetime import datetime
            try:
                if isinstance(timestamp, str):
                    dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                else:
                    dt = timestamp
                hour = dt.hour
                if hour < 6 or hour >= 22:
                    score += 0.15
                    explanations.append("Activity during unusual hours (night)")
                elif dt.weekday() >= 5:
                    score += 0.1
                    explanations.append("Activity during weekend")
            except (ValueError, AttributeError):
                pass

        # Rule 3: Privileged actions
        action_type = action.get("type", "").lower()
        action_subtype = action.get("subtype", "").lower()
        privileged_keywords = {"sudo", "root", "admin", "privilege", "escalat"}
        if any(kw in action_type or kw in action_subtype for kw in privileged_keywords):
            score += 0.25
            explanations.append("Privileged action attempted")

        # Rule 4: External IP for internal actions
        source_ip = actor.get("source_ip", "")
        if source_ip and not self._is_internal_ip(source_ip):
            if source.get("type") in ("system", "auth"):
                score += 0.15
                explanations.append("External IP accessing internal resource")

        # Rule 5: Attack type specific boosts
        if attack_type in ("privilege_escalation", "data_exfiltration"):
            score += 0.2
            explanations.append(f"High-risk attack type: {attack_type}")
        elif attack_type in ("lateral_movement", "credential_stuffing"):
            score += 0.1

        # Normalize to [0, 1]
        score = min(1.0, score)

        return score, explanations

    def _is_internal_ip(self, ip_str: str) -> bool:
        """Check if IP is in private range."""
        import ipaddress
        try:
            ip = ipaddress.ip_address(ip_str)
            return ip.is_private
        except ValueError:
            return False

    def _calculate_severity(self, threat_score: float) -> str:
        """Map threat score to severity level."""
        for severity, threshold in SEVERITY_THRESHOLDS.items():
            if threat_score >= threshold:
                return severity
        return "LOW"

    def _calculate_confidence(
        self,
        anomaly_conf: float,
        classification_conf: float,
        anomaly_score: float,
        classification_threat: float,
    ) -> float:
        """
        Calculate overall confidence in the assessment.

        Higher confidence when:
        - Both models agree (both high or both low threat)
        - Individual model confidence is high
        """
        # Agreement factor: both models should point in same direction
        agreement = 1.0 - abs(anomaly_score - classification_threat)

        # Weighted average of individual confidences
        avg_confidence = (
            self.anomaly_weight * anomaly_conf
            + self.classification_weight * classification_conf
        ) / (self.anomaly_weight + self.classification_weight)

        # Combine with agreement factor
        overall = 0.6 * avg_confidence + 0.4 * agreement

        return max(0.0, min(1.0, overall))

    def _generate_explanations(
        self,
        anomaly_result: dict[str, Any],
        classification_result: dict[str, Any],
        event: dict[str, Any],
        rules_explanations: list[str],
    ) -> list[str]:
        """Generate human-readable explanations for the assessment."""
        explanations = []

        # Anomaly explanation
        if anomaly_result.get("is_anomaly"):
            score = anomaly_result.get("score", 0)
            explanations.append(
                f"Anomalous behavior detected (score: {score:.2f})"
            )

        # Classification explanation
        attack_type = classification_result.get("attack_type", "normal")
        confidence = classification_result.get("confidence", 0)
        if attack_type != "normal":
            explanations.append(
                f"Attack pattern matches {attack_type.replace('_', ' ')} "
                f"({confidence:.0%} confidence)"
            )

        # Add rule-based explanations
        explanations.extend(rules_explanations)

        # Event-specific context
        actor = event.get("actor", {})
        action = event.get("action", {})

        if actor.get("user"):
            explanations.append(f"User involved: {actor['user']}")

        if actor.get("source_ip"):
            explanations.append(f"Source IP: {actor['source_ip']}")

        # Limit to most relevant explanations
        return explanations[:7]

    def _get_recommendations(
        self,
        attack_type: str,
        severity: str,
        threat_score: float,
    ) -> list[str]:
        """Get recommended actions based on attack type and severity."""
        # Get attack-specific recommendations
        recommendations = ATTACK_RECOMMENDATIONS.get(
            attack_type,
            ATTACK_RECOMMENDATIONS["normal"],
        ).copy()

        # Add severity-based recommendations
        if severity == "CRITICAL":
            recommendations.insert(0, "IMMEDIATE: Escalate to security team")
            recommendations.insert(1, "Consider isolating affected systems")
        elif severity == "HIGH":
            recommendations.insert(0, "Alert security team within 15 minutes")

        # Limit recommendations
        return recommendations[:5]

    def get_params(self) -> dict[str, float]:
        """Get fusion layer parameters."""
        return {
            "anomaly_weight": self.anomaly_weight,
            "classification_weight": self.classification_weight,
            "rules_weight": self.rules_weight,
        }
