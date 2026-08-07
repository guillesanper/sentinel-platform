"""
Inference service for the flow-based cascade.

Loads a cascade written by :meth:`ml.flow.pipelines.cascade_xgb.CascadeXGBPipeline.save`
and answers single flows and batches. It is the only path from an HTTP payload
to the flow models, and it exists mainly to enforce one invariant:

    **the column order the artifact was trained on must equal**
    :data:`ml.flow.features.FLOW_FEATURE_NAMES`.

That check is the whole reason the feature names are persisted. A mismatch does
not crash a model, does not warn, and does not degrade any offline metric — it
produces confident predictions from features fed into the wrong columns. So the
service refuses to come up at all when the two disagree, and
:attr:`CascadeInferenceService.is_loaded` stays False, which is what the
readiness probe reports and what makes the flow endpoints answer 503.

Nothing here imports matplotlib, pandas or anything from the event-based track:
``api/Dockerfile`` copies ``ml/`` wholesale into the production image.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np
import structlog

from ml.flow.config import FlowConfig
from ml.flow.features import FLOW_FEATURE_NAMES, FlowFeatureExtractor, feature_mapping
from ml.flow.fusion import FlowFusionLayer
from ml.flow.labels import BENIGN_LABEL
from ml.flow.pipelines.cascade_xgb import CascadeXGBPipeline

logger = structlog.get_logger()


# Spread, in decision-function units, over which the anomaly score sweeps from
# "clearly benign" to "clearly anomalous". IsolationForest.decision_function is
# bounded in roughly [-0.5, 0.5] by construction, so 0.05 puts most of the
# transition within a tenth of that range around the operating point.
#
# This is presentation, not calibration: the map is monotone in the raw score
# and lands on exactly 0.5 at the threshold, so it preserves the ranking and the
# BENIGN/anomalous decision. It does not claim to be a probability.
_SCORE_TEMPERATURE: Final[float] = 0.05


class FeatureContractError(ValueError):
    """
    The artifact's column order disagrees with :data:`FLOW_FEATURE_NAMES`.

    Raised loudly and never caught inside this module. Serving a cascade whose
    columns mean something other than what the extractor produces is worse than
    serving nothing, because the wrong answers look exactly like right ones.
    """


class CascadeInferenceService:
    """
    Serves the Isolation Forest → XGBoost cascade over network flows.

    Attributes:
        is_loaded: Whether a validated cascade is ready to answer.

    Example:
        >>> service = CascadeInferenceService()
        >>> service.load()                                   # doctest: +SKIP
        >>> service.predict({"features": {"Flow Duration": 1200}})  # doctest: +SKIP
    """

    def __init__(
        self,
        config: FlowConfig | None = None,
        fusion: FlowFusionLayer | None = None,
    ) -> None:
        """
        Args:
            config: Track configuration; supplies the default artifact
                directory. Defaults to ``FlowConfig()``.
            fusion: Threat-assessment layer. Defaults to ``FlowFusionLayer()``.
        """
        self._config = config or FlowConfig()
        self._extractor = FlowFeatureExtractor()
        self._fusion = fusion or FlowFusionLayer()
        self._pipeline: CascadeXGBPipeline | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self, models_dir: Path | str | None = None) -> "CascadeInferenceService":
        """
        Load a saved cascade and validate its feature contract.

        Reuses :meth:`CascadeXGBPipeline.load` rather than reading the joblib
        files directly, so the on-disk layout is described in exactly one place.

        Args:
            models_dir: Directory written by ``CascadeXGBPipeline.save``.
                Defaults to ``config.cascade_xgb_path`` (``models/flow/cascade_xgb``).

        Returns:
            self

        Raises:
            FileNotFoundError: The directory, an artifact or the feature-name
                contract is missing. Treated as "no trained models yet" by the
                API, exactly like the event track's models.
            FeatureContractError: The artifact's column order is not
                :data:`FLOW_FEATURE_NAMES`.
        """
        directory = Path(models_dir) if models_dir is not None else self._config.cascade_xgb_path

        pipeline = CascadeXGBPipeline(self._config).load(directory)
        self._validate_feature_contract(pipeline.feature_names, directory)
        self._pipeline = pipeline

        logger.info(
            "flow_inference_service_loaded",
            path=str(directory),
            n_features=self._extractor.n_features,
            classes=sorted(str(c) for c in pipeline.stage2_classifier.classes_),
        )
        return self

    @property
    def is_loaded(self) -> bool:
        """Whether a validated cascade is loaded and able to predict."""
        return self._pipeline is not None

    @property
    def feature_names(self) -> list[str]:
        """The column order this service feeds the models."""
        return self._extractor.feature_names

    def get_operating_point(self) -> dict[str, Any]:
        """
        Stage 1's measured precision/recall at its active threshold.

        Available without a validation set because ``save()`` persists the
        operating point alongside the forest.

        Returns:
            The Stage 1 diagnostics, or ``{}`` when the loaded artifact never
            had its threshold optimised.

        Raises:
            RuntimeError: No cascade is loaded.
        """
        pipeline = self._require_loaded()
        try:
            return pipeline.get_if_operating_point()
        except RuntimeError:
            return {}

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, flow: Mapping[str, Any]) -> dict[str, Any]:
        """
        Assess a single network flow.

        Args:
            flow: Wire-shaped flow (``{"features": {...}, "src_ip": ...}``) or a
                flat feature mapping.

        Returns:
            The fusion assessment: ``threat_score``, ``severity``,
            ``attack_type``, ``confidence``, ``is_anomaly``, ``explanation``,
            ``recommended_actions`` and ``details``.

        Raises:
            RuntimeError: No cascade is loaded.
        """
        return self.predict_batch([flow])[0]

    def predict_batch(self, flows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        """
        Assess a batch of flows with one pass through each model.

        Stage 1 scores the whole matrix at once and Stage 2 sees a single
        ``predict_proba`` call covering every routed flow, rather than one model
        call per flow. Only the fusion layer — pure Python arithmetic over a
        handful of numbers — runs per row.

        Args:
            flows: Flows in either accepted shape.

        Returns:
            One assessment per input flow, in input order.

        Raises:
            RuntimeError: No cascade is loaded.
        """
        pipeline = self._require_loaded()

        if not flows:
            return []

        X = self._extractor.transform_batch(feature_mapping(flow) for flow in flows)

        detector = pipeline.if_model
        raw_scores = detector.decision_scores(X)
        threshold = detector.threshold if detector.threshold is not None else 0.0
        anomaly_mask = detector.predict_binary(X).astype(bool)
        anomaly_scores = _normalise_scores(raw_scores, threshold)

        classifications = self._classify(pipeline, X, anomaly_mask, anomaly_scores)

        results = [
            self._fusion.combine(
                anomaly_result={
                    "score": float(anomaly_scores[i]),
                    "is_anomaly": bool(anomaly_mask[i]),
                    # How far the flow sits from the decision boundary is the
                    # only confidence Stage 1 can offer: right at the threshold
                    # it is a coin flip, far from it it is not.
                    "confidence": float(abs(2.0 * anomaly_scores[i] - 1.0)),
                },
                classification_result=classifications[i],
                flow=flow,
            )
            for i, flow in enumerate(flows)
        ]

        logger.info(
            "flow_batch_predicted",
            n_flows=len(flows),
            n_anomalous=int(anomaly_mask.sum()),
        )
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _classify(
        self,
        pipeline: CascadeXGBPipeline,
        X: np.ndarray,
        anomaly_mask: np.ndarray,
        anomaly_scores: np.ndarray,
    ) -> list[dict[str, Any]]:
        """
        Run Stage 2 on the routed rows and lay the results back out in order.

        Flows Stage 1 called benign never reach Stage 2 — that is the defining
        property of the cascade — so they get a synthetic BENIGN classification
        whose confidence is Stage 1's own certainty.
        """
        classifications: list[dict[str, Any]] = [
            {
                "attack_type": BENIGN_LABEL,
                "confidence": float(1.0 - anomaly_scores[i]),
                "probabilities": {BENIGN_LABEL: float(1.0 - anomaly_scores[i])},
            }
            for i in range(X.shape[0])
        ]

        if not anomaly_mask.any():
            return classifications

        classifier = pipeline.stage2_classifier
        classes = [str(c) for c in classifier.classes_]
        probabilities = classifier.predict_proba(X[anomaly_mask])

        # A two-class booster returns (n, 2) columns ordered as classes_, same
        # as the multiclass case, so a single argmax covers both.
        best = probabilities.argmax(axis=1)

        for row, index in enumerate(np.flatnonzero(anomaly_mask)):
            per_class = {classes[j]: float(probabilities[row, j]) for j in range(len(classes))}
            classifications[int(index)] = {
                "attack_type": classes[int(best[row])],
                "confidence": float(probabilities[row, best[row]]),
                "probabilities": per_class,
            }

        return classifications

    def _validate_feature_contract(self, names: list[str] | None, directory: Path) -> None:
        """
        Refuse an artifact whose column order is not the frozen contract.

        Raises:
            FeatureContractError: The lists differ in length, in content or in
                order. Order counts: two lists holding the same names in a
                different arrangement describe two different models.
        """
        expected = list(FLOW_FEATURE_NAMES)
        if names == expected:
            return

        detail = _describe_mismatch(names, expected)
        logger.error(
            "flow_feature_contract_mismatch",
            path=str(directory),
            n_expected=len(expected),
            n_found=len(names) if names is not None else 0,
            detail=detail,
        )
        raise FeatureContractError(
            f"Cascade at {directory} was trained on a different feature space than "
            f"FLOW_FEATURE_NAMES: {detail}. Serving it would feed each column's values "
            "into the wrong feature, producing confident nonsense with no error. "
            "Retrain the cascade against the current contract."
        )

    def _require_loaded(self) -> CascadeXGBPipeline:
        """Raise when no cascade has been loaded."""
        if self._pipeline is None:
            raise RuntimeError(
                "No flow cascade loaded. Call load() with a directory written by "
                "CascadeXGBPipeline.save()."
            )
        return self._pipeline


# ============================================================================
# Module-level helpers
# ============================================================================


def _normalise_scores(raw_scores: np.ndarray, threshold: float) -> np.ndarray:
    """
    Squash raw decision-function scores into a [0, 1] anomaly score.

    Monotonically **decreasing** in the raw score, because Stage 1's convention
    is that lower means more anomalous, and equal to 0.5 exactly at the
    threshold. See :data:`_SCORE_TEMPERATURE` on what this is and is not.

    Args:
        raw_scores: Output of ``FlowAnomalyDetector.decision_scores``.
        threshold: The detector's active operating point.

    Returns:
        Float array of the same shape, in [0, 1].
    """
    if raw_scores.size == 0:
        return np.empty(0, dtype=float)

    # Clipped before exp() so that a score far from the threshold saturates at
    # 0 or 1 instead of overflowing.
    exponent = np.clip((raw_scores - threshold) / _SCORE_TEMPERATURE, -50.0, 50.0)
    scores: np.ndarray = 1.0 / (1.0 + np.exp(exponent))
    return scores


def _describe_mismatch(names: list[str] | None, expected: list[str]) -> str:
    """Explain, in one line, how a persisted feature list differs from the contract."""
    if names is None:
        return "the artifact carries no feature names at all"

    if len(names) != len(expected):
        return f"expected {len(expected)} features, artifact has {len(names)}"

    missing = sorted(set(expected) - set(names))
    extra = sorted(set(names) - set(expected))
    if missing or extra:
        return f"missing {missing[:5]}, unexpected {extra[:5]}"

    first = next(i for i, (a, b) in enumerate(zip(names, expected, strict=True)) if a != b)
    return (
        f"same names in a different order — position {first} holds {names[first]!r}, "
        f"expected {expected[first]!r}"
    )
