"""
Unit tests for the flow-track threat assessment.

Two properties carry most of the weight here.

**Monotonicity.** The threat score has to rise with the probability of attack.
Anything else — a score that plateaus, dips, or reacts to the class name rather
than the probability — makes triage by threshold meaningless, and it is the kind
of defect that a spot-check of one example never catches.

**Playbook completeness.** A canonical class with no entry in
``FLOW_ATTACK_RECOMMENDATIONS`` falls back to advice written for benign traffic:
an analyst reading "No immediate action required" under a Botnet alert. The
suite pins all seven.
"""

import pytest

from ml.flow.fusion import (
    FLOW_ATTACK_RECOMMENDATIONS,
    FLOW_SEVERITY_THRESHOLDS,
    UNKNOWN_ANOMALY,
    FlowFusionLayer,
)
from ml.flow.labels import BENIGN_LABEL, CLASS_ORDER

# ============================================================================
# Helpers
# ============================================================================


def _anomaly(score: float = 0.5, *, is_anomaly: bool = True) -> dict:
    return {"score": score, "is_anomaly": is_anomaly, "confidence": 0.8}


def _classification(attack_type: str = "DDoS", attack_prob: float = 0.9) -> dict:
    """A Stage 2 verdict carrying an explicit benign probability."""
    return {
        "attack_type": attack_type,
        "confidence": attack_prob,
        "probabilities": {BENIGN_LABEL: 1.0 - attack_prob, attack_type: attack_prob},
    }


@pytest.fixture
def fusion() -> FlowFusionLayer:
    return FlowFusionLayer()


@pytest.fixture
def quiet_flow() -> dict:
    """A flow that trips no rule, so the rule layer contributes a constant."""
    return {
        "features": {
            "Flow Duration": 5_000_000.0,
            "Total Fwd Packets": 40.0,
            "Total Backward Packets": 38.0,
            "Flow Bytes/s": 2_000.0,
            "Destination Port": 8080.0,
            "SYN Flag Count": 1.0,
            "ACK Flag Count": 1.0,
            "RST Flag Count": 0.0,
        }
    }


# ============================================================================
# Monotonicity — the property triage depends on
# ============================================================================


class TestThreatScoreMonotonicity:
    """The score must be monotone in the probability of attack."""

    def test_rises_with_attack_probability(self, fusion: FlowFusionLayer, quiet_flow: dict):
        """
        Attack family and anomaly signal held fixed; only the probability moves.
        Holding the family fixed matters: the rule layer boosts some families,
        so varying both at once would not isolate the property under test.
        """
        scores = [
            fusion.combine(_anomaly(), _classification("DDoS", p), quiet_flow)["threat_score"]
            for p in (0.1, 0.3, 0.5, 0.7, 0.9, 0.99)
        ]

        assert scores == sorted(scores)
        assert scores[-1] > scores[0]

    def test_is_strictly_increasing_between_distinguishable_probabilities(
        self, fusion: FlowFusionLayer, quiet_flow: dict
    ):
        low = fusion.combine(_anomaly(), _classification("DoS", 0.2), quiet_flow)["threat_score"]
        high = fusion.combine(_anomaly(), _classification("DoS", 0.8), quiet_flow)["threat_score"]

        assert high > low

    def test_rises_with_the_anomaly_score(self, fusion: FlowFusionLayer, quiet_flow: dict):
        """The Stage 1 signal carries its own weight and must be felt."""
        scores = [
            fusion.combine(_anomaly(s), _classification("Scan", 0.5), quiet_flow)["threat_score"]
            for s in (0.0, 0.25, 0.5, 0.75, 1.0)
        ]

        assert scores == sorted(scores)
        assert scores[-1] > scores[0]

    def test_falls_back_to_confidence_when_no_benign_probability_is_given(
        self, fusion: FlowFusionLayer, quiet_flow: dict
    ):
        """
        Stage 2 only ever sees classes Stage 1 routed to it, so BENIGN may be
        absent from its class list. The score must stay monotone anyway.
        """
        scores = [
            fusion.combine(
                _anomaly(),
                {"attack_type": "Botnet", "confidence": c, "probabilities": {"Botnet": c}},
                quiet_flow,
            )["threat_score"]
            for c in (0.2, 0.5, 0.8)
        ]

        assert scores == sorted(scores)

    def test_a_confident_benign_verdict_scores_lower_than_a_hesitant_one(
        self, fusion: FlowFusionLayer, quiet_flow: dict
    ):
        confident = fusion.combine(
            _anomaly(0.1, is_anomaly=False),
            {"attack_type": BENIGN_LABEL, "confidence": 0.99, "probabilities": {}},
            quiet_flow,
        )["threat_score"]
        hesitant = fusion.combine(
            _anomaly(0.1, is_anomaly=False),
            {"attack_type": BENIGN_LABEL, "confidence": 0.51, "probabilities": {}},
            quiet_flow,
        )["threat_score"]

        assert confident < hesitant

    def test_score_stays_within_bounds_at_the_extremes(
        self, fusion: FlowFusionLayer, quiet_flow: dict
    ):
        worst = fusion.combine(_anomaly(1.0), _classification("WebAttack", 1.0), quiet_flow)
        best = fusion.combine(
            _anomaly(0.0, is_anomaly=False), _classification(BENIGN_LABEL, 0.0), quiet_flow
        )

        assert 0.0 <= best["threat_score"] <= worst["threat_score"] <= 1.0


# ============================================================================
# Playbooks
# ============================================================================


class TestRecommendations:
    """Every class an analyst can be shown must come with something to do."""

    @pytest.mark.parametrize("attack_type", CLASS_ORDER)
    def test_every_canonical_class_has_a_playbook(self, attack_type: str):
        assert attack_type in FLOW_ATTACK_RECOMMENDATIONS
        assert FLOW_ATTACK_RECOMMENDATIONS[attack_type]

    def test_the_unknown_anomaly_fallback_has_one_too(self):
        assert FLOW_ATTACK_RECOMMENDATIONS[UNKNOWN_ANOMALY]

    @pytest.mark.parametrize("attack_type", [c for c in CLASS_ORDER if c != BENIGN_LABEL])
    def test_an_attack_never_returns_the_benign_advice(
        self, fusion: FlowFusionLayer, quiet_flow: dict, attack_type: str
    ):
        """
        The failure this guards against: a missing playbook degrades to BENIGN's
        "No immediate action required", printed under an attack alert.
        """
        result = fusion.combine(_anomaly(0.9), _classification(attack_type, 0.95), quiet_flow)

        assert result["recommended_actions"]
        assert "No immediate action required" not in result["recommended_actions"]

    def test_benign_traffic_gets_the_benign_advice(self, fusion: FlowFusionLayer, quiet_flow: dict):
        result = fusion.combine(
            _anomaly(0.05, is_anomaly=False),
            {
                "attack_type": BENIGN_LABEL,
                "confidence": 0.99,
                "probabilities": {BENIGN_LABEL: 0.99},
            },
            quiet_flow,
        )

        assert result["attack_type"] == BENIGN_LABEL
        assert "No immediate action required" in result["recommended_actions"]

    def test_critical_severity_escalates_first(self, fusion: FlowFusionLayer):
        """
        Both models certain *and* the rule layer agreeing. Model certainty alone
        tops out at 0.8 + the family boost, which is HIGH — reaching CRITICAL
        takes corroborating evidence off the flow itself, which is the intended
        behaviour of a 20% rule weight.
        """
        alarming_flow = {
            "features": {
                "Destination Port": 22.0,
                "SYN Flag Count": 1.0,
                "ACK Flag Count": 0.0,
            }
        }

        result = fusion.combine(_anomaly(1.0), _classification("Botnet", 1.0), alarming_flow)

        assert result["severity"] == "CRITICAL"
        assert result["recommended_actions"][0].startswith("IMMEDIATE")

    def test_recommendations_are_capped(self, fusion: FlowFusionLayer, quiet_flow: dict):
        result = fusion.combine(_anomaly(1.0), _classification("WebAttack", 1.0), quiet_flow)

        assert len(result["recommended_actions"]) <= 5

    def test_playbooks_are_not_mutated_between_calls(
        self, fusion: FlowFusionLayer, quiet_flow: dict
    ):
        """Severity escalation inserts entries; it must not do so in the constant."""
        before = list(FLOW_ATTACK_RECOMMENDATIONS["Botnet"])
        fusion.combine(_anomaly(1.0), _classification("Botnet", 1.0), quiet_flow)

        assert FLOW_ATTACK_RECOMMENDATIONS["Botnet"] == before


# ============================================================================
# Severity
# ============================================================================


class TestSeverity:
    """Mapping from score to the band an analyst triages by."""

    @pytest.mark.parametrize(
        ("score", "expected"),
        [
            (0.95, "CRITICAL"),
            (0.90, "CRITICAL"),
            (0.75, "HIGH"),
            (0.70, "HIGH"),
            (0.50, "MEDIUM"),
            (0.40, "MEDIUM"),
            (0.10, "LOW"),
            (0.00, "LOW"),
        ],
    )
    def test_bands(self, fusion: FlowFusionLayer, score: float, expected: str):
        assert fusion._calculate_severity(score) == expected

    def test_the_scale_matches_the_event_track(self):
        """
        Deliberately identical to ``ml.models.fusion.SEVERITY_THRESHOLDS``: two
        scales with the same names and different cut points would be worse than
        no scale at all for an analyst reading both tracks' alerts.
        """
        from ml.models.fusion import SEVERITY_THRESHOLDS

        assert FLOW_SEVERITY_THRESHOLDS == SEVERITY_THRESHOLDS


# ============================================================================
# The rule layer
# ============================================================================


class TestRuleLayer:
    """
    Flow-shaped heuristics. The event track's rules — failed logins, business
    hours, sudo, private IP ranges — have no counterpart in a CICFlowMeter
    record, so none of them appear here.
    """

    def test_a_probe_shaped_flow_scores_above_a_session(self, fusion: FlowFusionLayer):
        probe, _ = fusion._apply_rules(
            {"Flow Duration": 200.0, "Total Fwd Packets": 1.0, "Total Backward Packets": 0.0},
            BENIGN_LABEL,
        )
        session, _ = fusion._apply_rules(
            {
                "Flow Duration": 5_000_000.0,
                "Total Fwd Packets": 200.0,
                "Total Backward Packets": 180.0,
            },
            BENIGN_LABEL,
        )

        assert probe > session

    def test_syn_without_ack_is_flagged(self, fusion: FlowFusionLayer):
        score, explanations = fusion._apply_rules(
            {"SYN Flag Count": 1.0, "ACK Flag Count": 0.0}, BENIGN_LABEL
        )

        assert score > 0
        assert any("SYN without ACK" in e for e in explanations)

    def test_a_completed_handshake_is_not_flagged(self, fusion: FlowFusionLayer):
        _, explanations = fusion._apply_rules(
            {"SYN Flag Count": 1.0, "ACK Flag Count": 3.0}, BENIGN_LABEL
        )

        assert not any("SYN without ACK" in e for e in explanations)

    def test_a_reset_is_flagged(self, fusion: FlowFusionLayer):
        _, explanations = fusion._apply_rules({"RST Flag Count": 2.0}, BENIGN_LABEL)

        assert any("reset" in e.lower() for e in explanations)

    def test_high_throughput_is_flagged(self, fusion: FlowFusionLayer):
        _, explanations = fusion._apply_rules({"Flow Bytes/s": 5e7}, BENIGN_LABEL)

        assert any("throughput" in e.lower() for e in explanations)

    @pytest.mark.parametrize(
        ("port", "service"), [(22, "SSH"), (3389, "RDP"), (445, "SMB"), (1433, "MSSQL")]
    )
    def test_sensitive_destination_ports_are_named(
        self, fusion: FlowFusionLayer, port: int, service: str
    ):
        _, explanations = fusion._apply_rules({"Destination Port": float(port)}, BENIGN_LABEL)

        assert any(service in e for e in explanations)

    def test_an_ordinary_port_is_not_flagged(self, fusion: FlowFusionLayer):
        _, explanations = fusion._apply_rules({"Destination Port": 8080.0}, BENIGN_LABEL)

        assert not any("sensitive service" in e for e in explanations)

    def test_high_risk_families_are_boosted(self, fusion: FlowFusionLayer):
        botnet, _ = fusion._apply_rules({}, "Botnet")
        scan, _ = fusion._apply_rules({}, "Scan")

        assert botnet > scan

    def test_score_is_capped_at_one(self, fusion: FlowFusionLayer):
        """Every signal firing at once must not push the layer out of range."""
        score, _ = fusion._apply_rules(
            {
                "Flow Duration": 100.0,
                "Total Fwd Packets": 1.0,
                "SYN Flag Count": 1.0,
                "ACK Flag Count": 0.0,
                "RST Flag Count": 1.0,
                "Flow Bytes/s": 1e9,
                "Destination Port": 22.0,
            },
            "Botnet",
        )

        assert score == 1.0

    def test_missing_and_junk_features_do_not_raise(self, fusion: FlowFusionLayer):
        score, _ = fusion._apply_rules({"Flow Duration": "nonsense", "SYN Flag Count": None}, "DoS")

        assert 0.0 <= score <= 1.0


# ============================================================================
# The assembled assessment
# ============================================================================


class TestCombine:
    """Shape and content of the returned assessment."""

    def test_returns_every_documented_field(self, fusion: FlowFusionLayer, quiet_flow: dict):
        result = fusion.combine(_anomaly(), _classification(), quiet_flow)

        assert set(result) == {
            "threat_score",
            "severity",
            "attack_type",
            "confidence",
            "is_anomaly",
            "explanation",
            "recommended_actions",
            "details",
        }

    def test_confidence_stays_in_range(self, fusion: FlowFusionLayer, quiet_flow: dict):
        result = fusion.combine(_anomaly(), _classification(), quiet_flow)

        assert 0.0 <= result["confidence"] <= 1.0

    def test_details_break_the_score_down(self, fusion: FlowFusionLayer, quiet_flow: dict):
        details = fusion.combine(_anomaly(), _classification(), quiet_flow)["details"]

        assert set(details) == {
            "anomaly_score",
            "classification_threat",
            "classification_confidence",
            "rules_score",
            "class_probabilities",
        }

    def test_an_unexplained_anomaly_is_named_rather_than_reported_benign(
        self, fusion: FlowFusionLayer, quiet_flow: dict
    ):
        """
        Stage 1 is sure the flow is odd, Stage 2 cannot tell it from benign
        traffic. That state deserves a name — reporting BENIGN would hide it.
        """
        result = fusion.combine(
            _anomaly(0.95),
            {"attack_type": BENIGN_LABEL, "confidence": 0.6, "probabilities": {BENIGN_LABEL: 0.6}},
            quiet_flow,
        )

        assert result["attack_type"] == UNKNOWN_ANOMALY

    def test_a_mild_anomaly_is_left_alone(self, fusion: FlowFusionLayer, quiet_flow: dict):
        result = fusion.combine(
            _anomaly(0.55),
            {"attack_type": BENIGN_LABEL, "confidence": 0.6, "probabilities": {BENIGN_LABEL: 0.6}},
            quiet_flow,
        )

        assert result["attack_type"] == BENIGN_LABEL

    def test_explanations_are_capped(self, fusion: FlowFusionLayer):
        flow = {
            "features": {
                "Flow Duration": 100.0,
                "Total Fwd Packets": 1.0,
                "SYN Flag Count": 1.0,
                "ACK Flag Count": 0.0,
                "RST Flag Count": 1.0,
                "Flow Bytes/s": 1e9,
                "Destination Port": 22.0,
            },
            "src_ip": "10.0.0.1",
            "dst_ip": "10.0.0.2",
            "dst_port": 22,
            "protocol": "TCP",
        }

        result = fusion.combine(_anomaly(0.99), _classification("Botnet", 0.99), flow)

        assert len(result["explanation"]) <= 7

    def test_endpoint_metadata_reaches_the_explanation(self, fusion: FlowFusionLayer):
        flow = {
            "features": {"Destination Port": 8080.0},
            "src_ip": "203.0.113.7",
            "dst_ip": "10.0.1.20",
            "dst_port": 8080,
            "protocol": "TCP",
        }

        result = fusion.combine(_anomaly(0.9), _classification("Scan", 0.9), flow)

        assert any("203.0.113.7" in e and "10.0.1.20" in e for e in result["explanation"])

    def test_a_flow_without_metadata_still_explains_itself(
        self, fusion: FlowFusionLayer, quiet_flow: dict
    ):
        result = fusion.combine(_anomaly(0.9), _classification("Scan", 0.9), quiet_flow)

        assert result["explanation"]

    def test_accepts_a_flat_feature_mapping(self, fusion: FlowFusionLayer):
        """The rule layer must read features whichever shape the flow arrives in."""
        nested = fusion.combine(
            _anomaly(), _classification(), {"features": {"Destination Port": 22.0}}
        )
        flat = fusion.combine(_anomaly(), _classification(), {"Destination Port": 22.0})

        assert nested["threat_score"] == flat["threat_score"]

    def test_missing_stage_output_falls_back_instead_of_raising(self, fusion: FlowFusionLayer):
        result = fusion.combine({}, {}, {})

        assert result["attack_type"] == BENIGN_LABEL
        assert 0.0 <= result["threat_score"] <= 1.0


# ============================================================================
# Weights
# ============================================================================


class TestWeights:
    """Ensemble weighting."""

    def test_defaults_match_the_documented_split(self):
        assert FlowFusionLayer().get_params() == {
            "anomaly_weight": 0.3,
            "classification_weight": 0.5,
            "rules_weight": 0.2,
        }

    def test_unnormalised_weights_are_rescaled(self):
        params = FlowFusionLayer(3.0, 5.0, 2.0).get_params()

        assert sum(params.values()) == pytest.approx(1.0)
        assert params["classification_weight"] == pytest.approx(0.5)

    def test_zero_weights_are_rejected(self):
        """Normalising by zero would silently produce NaN scores."""
        with pytest.raises(ValueError, match="positive"):
            FlowFusionLayer(0.0, 0.0, 0.0)

    def test_a_pure_classification_layer_tracks_the_classifier(self):
        fusion = FlowFusionLayer(0.0, 1.0, 0.0)

        result = fusion.combine(_anomaly(0.0, is_anomaly=False), _classification("DoS", 0.8), {})

        assert result["threat_score"] == pytest.approx(0.8)
