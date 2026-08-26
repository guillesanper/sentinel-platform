"""
Shared contract for the two-stage cascades of the flow-based track.

The architecture is fixed: Stage 1 (Isolation Forest) decides *whether* a flow
is anomalous, Stage 2 decides *which* attack it is. Flows Stage 1 calls BENIGN
never reach Stage 2. Both concrete cascades — XGBoost and the NumPy MLP —
implement this interface, so training, evaluation and persistence treat them
interchangeably.

The class is ``BaseCascadePipeline`` rather than ``BasePipeline`` because
"pipeline" already means something else in this repo: ``ml/features/pipeline.py``
is the feature-engineering pipeline of the event-based track.

Persisting the feature names is part of the contract, not a convenience. A
cascade trained on columns in one order and served data in another produces
plausible-looking garbage and raises nothing at all, so the name list travels
with the model and :mod:`ml.flow.serving` validates against it.
"""

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Final

import numpy as np
import structlog

from ml.flow.cv import TemporalIndex

logger = structlog.get_logger()


# ============================================================================
# Feature-name contract
# ============================================================================

# Written at the root of every saved cascade directory.
FEATURE_NAMES_FILE: Final[str] = "feature_names.json"

# Bumped when the layout of FEATURE_NAMES_FILE changes incompatibly.
FEATURE_NAMES_FORMAT_VERSION: Final[str] = "1.0.0"


class BaseCascadePipeline(ABC):
    """
    Abstract two-stage cascade over network flows.

    Subclasses own their Stage 2 model; everything to do with the feature-name
    contract and evaluation logging lives here.

    Attributes:
        _feature_names: Training-time column order, or None when unknown.
    """

    _feature_names: list[str] | None

    # ------------------------------------------------------------------
    # Abstract contract
    # ------------------------------------------------------------------

    @abstractmethod
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        *,
        beta: float = 1.0,
        skip_tuning: bool = False,
        feature_names: list[str] | None = None,
        train_index: TemporalIndex | None = None,
    ) -> "BaseCascadePipeline":
        """
        Train Stage 1 and Stage 2 in sequence.

        Args:
            X_train: Full training feature matrix, all classes. Build it with
                :func:`ml.flow.loader.get_feature_columns` — never with a
                hand-written column list, or ``capture_day``/``order_idx`` leak
                in and the model learns the capture schedule.
            y_train: Canonical string class labels for the training set.
            X_val: Validation feature matrix, used for the Stage 1 threshold
                search and Stage 2 early stopping.
            y_val: Canonical string class labels for the validation set.
            beta: F-beta weight for the Stage 1 threshold. ``beta > 1``
                penalises missed attacks, ``beta < 1`` penalises false alarms.
            feature_names: Column names of ``X_train``, in order. Recorded so
                that :meth:`save` can persist them.
            skip_tuning: Skip the Stage 2 hyperparameter search and use the
                config defaults.
            train_index: Chronological position and temporal block of every row
                of ``X_train``, from
                :func:`ml.flow.loader.build_temporal_index`. Lets Stage 2 tune
                on chronological folds instead of shuffled ones; see
                :mod:`ml.flow.cv`. Optional, and ignored by cascades whose Stage
                2 has no hyperparameter search. Pass None for a random split:
                there is no temporal claim to make about it.

        Returns:
            self
        """

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Run end-to-end cascade inference.

        Stage 1 labels each flow BENIGN or anomalous; the anomalous ones go to
        Stage 2 for an attack family, the rest get ``BENIGN`` directly.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.

        Returns:
            Canonical string labels of shape ``(n_samples,)``.
        """

    @abstractmethod
    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        *,
        labels: list[str] | None = None,
    ) -> dict[str, float]:
        """
        Score the cascade end to end on a held-out test set.

        Args:
            X_test: Test feature matrix.
            y_test: Ground-truth canonical string labels.
            labels: Canonical class list to average the F1 scores over. Pass the
                classes the run declares — ``_Dataset.labels`` in the training
                CLI — so that a class the test set happens not to contain still
                counts as a 0.0 in the macro. Left as None, sklearn averages
                over the union of ``y_test`` and ``y_pred``, which **drops** an
                unsupported class from the denominator and silently inflates the
                headline figure relative to the ``classification_report``
                printed beside it. The default is None only so that callers
                written before this argument existed keep their measured
                behaviour.

        Returns:
            Metric name → value. Implementations must provide at least
            ``accuracy``, ``f1_macro`` and ``f1_weighted``.

        Note:
            Per-class figures deliberately do not belong here: this dict is
            ``dict[str, float]`` and :meth:`_log_evaluation` rounds every value
            as a float. Compute them with
            :func:`ml.evaluation.metrics.compute_metrics` — the training CLI
            records them under the summary's ``per_class`` key.
        """

    @abstractmethod
    def save(self, directory: Path | str) -> None:
        """
        Persist every fitted component plus the feature-name contract.

        Implementations must serialise enough to restore the cascade for
        inference without retraining: both stage models, label encoders,
        thresholds, and the feature names.

        Args:
            directory: Target directory, created recursively if absent.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier used in log events and output paths."""

    # ------------------------------------------------------------------
    # Feature-name contract — concrete, shared by every cascade
    # ------------------------------------------------------------------

    @property
    def feature_names(self) -> list[str] | None:
        """Training-time column order, or None if it was never recorded."""
        return list(self._feature_names) if self._feature_names is not None else None

    def _save_feature_names(self, directory: Path) -> None:
        """
        Write the feature-name contract next to the models.

        Raises:
            RuntimeError: The cascade has no feature names. Refusing to save is
                deliberate: a model saved without them cannot be validated at
                inference, and a feature-order mismatch is the one failure in
                this design that produces wrong predictions with no error
                anywhere. Pass ``feature_names`` to :meth:`train` or set them at
                construction.
        """
        if not self._feature_names:
            raise RuntimeError(
                f"[{self.name}] cannot be saved without feature names. Pass "
                "feature_names=get_feature_columns(df) to train() or to the "
                "constructor — serving validates against this list, and without "
                "it a column-order mismatch would go undetected."
            )

        payload = {
            "format_version": FEATURE_NAMES_FORMAT_VERSION,
            "pipeline": self.name,
            "n_features": len(self._feature_names),
            "feature_names": list(self._feature_names),
        }
        (directory / FEATURE_NAMES_FILE).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info(
            "cascade_feature_names_saved",
            pipeline=self.name,
            n_features=len(self._feature_names),
        )

    def _load_feature_names(self, directory: Path) -> list[str]:
        """
        Read the feature-name contract back, in the order it was written.

        Args:
            directory: Directory previously written by :meth:`save`.

        Returns:
            The feature names, order preserved.

        Raises:
            FileNotFoundError: The contract file is absent, so the artifact
                cannot be served safely.
            ValueError: The file is malformed or its ``n_features`` disagrees
                with the list it carries.
        """
        path = directory / FEATURE_NAMES_FILE
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {FEATURE_NAMES_FILE} in {directory}. The artifact predates the "
                "feature-name contract or was written by hand; it cannot be served safely."
            )

        payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        names = payload.get("feature_names")
        if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
            raise ValueError(f"{path} does not carry a list of feature names.")

        declared = payload.get("n_features")
        if declared is not None and declared != len(names):
            raise ValueError(f"{path} declares n_features={declared} but lists {len(names)} names.")

        self._feature_names = list(names)
        logger.info(
            "cascade_feature_names_loaded",
            pipeline=self.name,
            n_features=len(names),
        )
        return list(names)

    # ------------------------------------------------------------------
    # Logging helper
    # ------------------------------------------------------------------

    def _log_evaluation(self, metrics: dict[str, float]) -> None:
        """Emit one structured event carrying the whole metrics dict."""
        logger.info(
            "cascade_evaluated",
            pipeline=self.name,
            **{k: round(float(v), 6) for k, v in metrics.items()},
        )
