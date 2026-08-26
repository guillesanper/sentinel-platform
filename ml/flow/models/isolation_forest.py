"""
Stage 1 anomaly detector for the flow-based track.

An Isolation Forest fitted **exclusively on BENIGN flows**, so that anything
structurally unlike normal traffic scores low regardless of whether its attack
family was ever seen. After fitting, the decision-function threshold is tuned on
the validation set to maximise F-beta, which is what makes the precision/recall
trade-off an explicit knob rather than an accident of ``contamination``:

* ``beta > 1`` penalises false negatives — fewer missed attacks reach Stage 2.
* ``beta < 1`` penalises false positives — fewer benign flows raise an alarm.

The class is named ``FlowAnomalyDetector`` and not ``AnomalyDetector`` on
purpose: ``ml.models.anomaly_detector.AnomalyDetector`` is the event-based
track's detector, it takes a different feature space and a different label
space, and the two must never be confused at an import site.
"""

import json
from pathlib import Path
from typing import Any, Final

import joblib
import numpy as np
import structlog
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix

from ml.evaluation.metrics import best_fbeta_threshold
from ml.flow.config import IsolationForestConfig
from ml.flow.labels import BENIGN_LABEL

logger = structlog.get_logger()


# ============================================================================
# Persistence contract
# ============================================================================

MODEL_FILE: Final[str] = "isolation_forest.joblib"
THRESHOLD_FILE: Final[str] = "if_threshold.npy"
THRESHOLD_INFO_FILE: Final[str] = "if_threshold_info.json"


class FlowAnomalyDetector:
    """
    Isolation Forest anomaly detector for CICFlowMeter network flows.

    Attributes:
        config: Hyperparameters for the underlying Isolation Forest.
        random_seed: Seed forwarded to sklearn for reproducibility.

    Example:
        >>> detector = FlowAnomalyDetector(IsolationForestConfig())
        >>> detector.fit(X_train, y_train)              # doctest: +SKIP
        >>> detector.optimize_threshold(X_val, y_val, beta=2.0)  # doctest: +SKIP
        >>> mask = detector.predict_binary(X_test)      # doctest: +SKIP
    """

    def __init__(self, config: IsolationForestConfig, random_seed: int = 42) -> None:
        """
        Args:
            config: Isolation Forest hyperparameters.
            random_seed: Seed for the forest's random state.
        """
        self.config = config
        self.random_seed = random_seed
        self._model: IsolationForest | None = None
        self._threshold: float | None = None
        self._threshold_info: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "FlowAnomalyDetector":
        """
        Train the forest on the BENIGN rows of ``X_train``.

        Args:
            X_train: Feature matrix of shape ``(n_samples, n_features)``, built
                with :func:`ml.flow.loader.get_feature_columns` — never by
                hand, or the capture-day columns leak in.
            y_train: Canonical string class labels of shape ``(n_samples,)``.

        Returns:
            self

        Raises:
            ValueError: ``y_train`` holds no BENIGN sample, so there is nothing
                to model normality from.
        """
        benign_mask = y_train == BENIGN_LABEL
        n_benign = int(benign_mask.sum())
        if n_benign == 0:
            raise ValueError("No BENIGN samples found in the training set.")

        logger.info(
            "flow_anomaly_detector_training",
            n_benign=n_benign,
            n_train=int(len(y_train)),
            benign_pct=round(100.0 * n_benign / len(y_train), 2),
        )

        self._model = IsolationForest(
            n_estimators=self.config.n_estimators,
            max_samples=self.config.max_samples,
            contamination=self.config.contamination,
            max_features=self.config.max_features,
            bootstrap=self.config.bootstrap,
            n_jobs=self.config.n_jobs,
            random_state=self.random_seed,
        )
        self._model.fit(X_train[benign_mask])

        # sklearn's own convention: decision_function < 0 → outlier. Replaced by
        # optimize_threshold as soon as a validation set is available.
        self._threshold = 0.0
        logger.info("flow_anomaly_detector_trained", threshold=self._threshold)
        return self

    def optimize_threshold(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
        beta: float = 1.0,
        n_candidates: int | None = None,
    ) -> float:
        """
        Pick the decision-function threshold maximising F-beta on ``X_val``.

        The decision rule at threshold ``t`` is::

            score(x) <  t  →  anomaly (positive class, 1)
            score(x) >= t  →  BENIGN  (negative class, 0)

        Every operating point the validation set actually offers is evaluated,
        so the threshold is the exact maximiser; see
        :func:`ml.evaluation.metrics.best_fbeta_threshold`.

        Args:
            X_val: Validation feature matrix.
            y_val: Canonical string class labels; anything other than
                ``BENIGN`` counts as the positive class.
            beta: Recall weight relative to precision. ``beta > 1`` favours
                recall, ``beta < 1`` favours precision.
            n_candidates: Optional cap on the operating points evaluated. None
                evaluates all of them.

        Returns:
            The selected threshold, also stored on the instance.

        Raises:
            RuntimeError: The model has not been fitted.
        """
        self._check_fitted()
        assert self._model is not None

        scores = self._model.decision_function(X_val)
        y_binary = (y_val != BENIGN_LABEL).astype(int)

        best_t, best_fbeta = best_fbeta_threshold(
            scores, y_binary, beta=beta, n_candidates=n_candidates
        )

        self._threshold = best_t
        y_pred_best = (scores < best_t).astype(int)
        self._threshold_info = self._compute_operating_info(y_binary, y_pred_best, beta)

        info = self._threshold_info
        logger.info(
            "flow_threshold_optimized",
            beta=beta,
            threshold=best_t,
            fbeta=round(best_fbeta, 6),
            precision=info["precision"],
            recall=info["recall"],
            fpr=info["fpr"],
            fnr=info["fnr"],
        )
        return best_t

    def measure_operating_point(
        self,
        X: np.ndarray,
        y: np.ndarray,
        beta: float | None = None,
    ) -> dict[str, Any]:
        """
        Measure the active threshold's operating point on an arbitrary set.

        :meth:`optimize_threshold` reports the operating point of the set it
        *selected* the threshold on, which is a fit statistic and not an estimate
        of held-out performance. Two effects push it optimistic, and both were
        measured on CICIDS2017: the threshold is the argmax of F-beta over every
        operating point that set offers, and — under the temporal protocol —
        validation is the middle slice of each block while test is the last, so
        validation sits systematically closer in time to training.

        On the reference run the gap is not a rounding detail: Stage 1 recall
        reads 0.760 on validation and 0.560 on test, i.e. the share of attacks
        the cascade can never recover goes from 24.0% to 44.0%. Since Stage 1's
        recall is a hard ceiling for the whole cascade, that number has to come
        from held-out data. Use this method for anything reported as a result.

        Args:
            X: Feature matrix to measure on.
            y: Canonical string class labels; anything other than ``BENIGN``
                counts as the positive class.
            beta: Recall weight used for the reported ``f<beta>_score``. Defaults
                to the beta the threshold was selected with, so the two operating
                points stay comparable.

        Returns:
            Same shape as :meth:`get_operating_point_info`.

        Raises:
            RuntimeError: The model has not been fitted or loaded.
        """
        self._check_fitted()
        assert self._model is not None

        if beta is None:
            beta = float(self._threshold_info.get("beta", 1.0))

        scores = self._model.decision_function(X)
        y_binary = (y != BENIGN_LABEL).astype(int)
        y_pred = (scores < self._threshold).astype(int)

        return self._compute_operating_info(y_binary, y_pred, beta)

    def predict_binary(self, X: np.ndarray) -> np.ndarray:
        """
        Apply the active threshold and return the binary anomaly mask.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.

        Returns:
            Integer array of shape ``(n_samples,)``: 1 = anomalous (routed to
            Stage 2), 0 = BENIGN.

        Raises:
            RuntimeError: The model has not been fitted or loaded.
        """
        self._check_fitted()
        assert self._model is not None

        # sklearn rejects a zero-row matrix outright. An empty batch is a
        # perfectly ordinary thing for a serving layer to hand down, and the
        # answer to it is an empty result, not an exception.
        if X.shape[0] == 0:
            return np.empty(0, dtype=int)

        scores = self._model.decision_function(X)
        mask: np.ndarray = (scores < self._threshold).astype(int)
        return mask

    def decision_scores(self, X: np.ndarray) -> np.ndarray:
        """
        Return the raw decision-function scores, before thresholding.

        :meth:`predict_binary` answers "anomalous or not"; this answers "by how
        much", which is what the serving layer needs to report a graded threat
        score instead of a bit. **Lower is more anomalous**, and the threshold
        is the point where the answer flips.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.

        Returns:
            Float array of shape ``(n_samples,)``; empty for an empty batch.

        Raises:
            RuntimeError: The model has not been fitted or loaded.
        """
        self._check_fitted()
        assert self._model is not None

        if X.shape[0] == 0:
            return np.empty(0, dtype=float)

        scores: np.ndarray = self._model.decision_function(X)
        return scores

    def get_operating_point_info(self) -> dict[str, Any]:
        """
        Return the operating point **on the set the threshold was selected on**.

        This is a record of the selection, not an estimate of held-out
        performance: the threshold is the argmax of F-beta over the operating
        points of this very set. Report it as "where the threshold came from",
        and use :meth:`measure_operating_point` on test for anything published
        as a result — on CICIDS2017 the two differ by 18 points of recall.

        Returns:
            Dictionary with ``threshold``, ``beta``, ``f<beta>_score``,
            ``precision``, ``recall``, ``fpr``, ``fnr`` and the raw
            ``tp``/``fp``/``tn``/``fn`` counts. A copy, so callers cannot
            mutate the detector's state.

        Raises:
            RuntimeError: :meth:`optimize_threshold` has never run and no saved
                operating point was restored, so no metrics exist. Fitting
                alone is not enough — the default threshold has never been
                measured against anything.
        """
        if not self._threshold_info:
            raise RuntimeError(
                "No operating-point info available. "
                "Call optimize_threshold() on a validation set first."
            )
        return dict(self._threshold_info)

    @property
    def is_fitted(self) -> bool:
        """Whether the detector can predict (fitted or loaded from disk)."""
        return self._model is not None and self._threshold is not None

    @property
    def threshold(self) -> float | None:
        """The active decision-function threshold, or None before fitting."""
        return self._threshold

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, directory: Path | str) -> None:
        """
        Persist the forest, its threshold and the operating point.

        The operating point is written as JSON so that a loaded detector can
        report precision/recall without a validation set at hand — the flow
        serving path needs it to explain its decisions.

        Args:
            directory: Target directory, created recursively if absent.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        joblib.dump(self._model, directory / MODEL_FILE)
        np.save(directory / THRESHOLD_FILE, np.array([self._threshold]))
        if self._threshold_info:
            (directory / THRESHOLD_INFO_FILE).write_text(
                json.dumps(self._threshold_info, indent=2), encoding="utf-8"
            )

        logger.info("flow_anomaly_detector_saved", path=str(directory))

    def load(self, directory: Path | str) -> "FlowAnomalyDetector":
        """
        Restore the forest, its threshold and — when present — the operating
        point.

        Args:
            directory: Directory previously written by :meth:`save`.

        Returns:
            self

        Raises:
            FileNotFoundError: The model or threshold file is missing.
        """
        directory = Path(directory)

        self._model = joblib.load(directory / MODEL_FILE)
        self._threshold = float(np.load(directory / THRESHOLD_FILE)[0])

        info_path = directory / THRESHOLD_INFO_FILE
        # Tolerated rather than required: a detector saved before ever being
        # optimised has no operating point, and that is a legitimate state.
        self._threshold_info = (
            json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {}
        )

        logger.info(
            "flow_anomaly_detector_loaded",
            path=str(directory),
            threshold=self._threshold,
            has_operating_point=bool(self._threshold_info),
        )
        return self

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        """Raise when the detector cannot predict yet."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fit or loaded before calling this method.")

    def _compute_operating_info(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        beta: float,
    ) -> dict[str, Any]:
        """
        Derive the full confusion-matrix diagnostics at the active threshold.

        F-beta is recomputed from the confusion matrix rather than carried in
        from the threshold search. On the selection set the two agree exactly —
        ``best_fbeta_threshold`` derives it from the same precision and recall —
        and off it there is no search result to carry, which is precisely the
        case :meth:`measure_operating_point` exists to cover.
        """
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

        beta_sq = beta * beta
        denominator = beta_sq * precision + recall
        fbeta = (1.0 + beta_sq) * precision * recall / denominator if denominator > 0 else 0.0

        beta_str = f"{beta:.1f}".rstrip("0").rstrip(".")
        return {
            "threshold": self._threshold,
            "beta": beta,
            f"f{beta_str}_score": round(fbeta, 6),
            "precision": round(float(precision), 6),
            "recall": round(float(recall), 6),
            "fpr": round(float(fpr), 6),
            "fnr": round(float(fnr), 6),
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn),
        }
