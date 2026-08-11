"""
Anomaly detection model using Isolation Forest.

Detects unusual patterns in security events that may indicate
malicious activity or novel attack vectors.
"""

from pathlib import Path
from typing import Any, Final

import joblib
import numpy as np
import structlog
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix

from ml.evaluation.metrics import best_fbeta_threshold

logger = structlog.get_logger()

# Default model path
DEFAULT_MODEL_PATH = Path("models/anomaly_detector.joblib")

# Artifact format written by save(). 1.1.0 adds "threshold_info"; load() still
# accepts 1.0.0 files, which simply carry no operating point.
ARTIFACT_VERSION: Final[str] = "1.1.0"

# Label of the negative class when y_val arrives as strings rather than 0/1.
NORMAL_LABEL: Final[str] = "normal"


def _binarize_labels(y: np.ndarray) -> np.ndarray:
    """
    Normalise validation labels to a 0/1 anomaly indicator.

    The event track hands down labels in two shapes and they are told apart by
    dtype, never by guessing:

    * numeric — already the binary indicator, 0 = normal, 1 = attack;
    * strings — canonical class names, anything but ``"normal"`` is an attack.

    ``dtype=object`` arrays are inspected element-wise, because numpy reports
    ``O`` for both a list of ints and a list of class names, and treating the
    former as strings would mark every row an attack.

    Args:
        y: Validation labels of shape ``(n_samples,)``.

    Returns:
        Integer array of shape ``(n_samples,)`` with values in ``{0, 1}``.
    """
    arr = np.asarray(y)
    is_string = arr.dtype.kind in {"U", "S"} or (
        arr.dtype.kind == "O" and arr.size > 0 and isinstance(arr.reshape(-1)[0], str)
    )
    result: np.ndarray = (arr != NORMAL_LABEL).astype(int) if is_string else (arr != 0).astype(int)
    return result


class AnomalyDetector:
    """
    Anomaly detection using Isolation Forest algorithm.

    Isolation Forest works by isolating observations through random partitioning.
    Anomalies are easier to isolate and thus have shorter path lengths in the tree.

    Attributes:
        contamination: Expected proportion of outliers in the data.
        n_estimators: Number of base estimators in the ensemble.
        is_loaded: Whether a trained model has been loaded.

    Example:
        >>> detector = AnomalyDetector(contamination=0.1)
        >>> detector.fit(X_train)
        >>> result = detector.predict(X_test[0])
        >>> print(f"Anomaly score: {result['score']:.3f}")
    """

    def __init__(
        self,
        contamination: float = 0.1,
        n_estimators: int = 100,
        max_samples: str | int = "auto",
        random_state: int = 42,
        model_path: Path | str | None = None,
    ) -> None:
        """
        Initialize the anomaly detector.

        Args:
            contamination: Expected proportion of outliers (0.0 to 0.5).
            n_estimators: Number of trees in the forest.
            max_samples: Number of samples to train each tree.
            random_state: Random seed for reproducibility.
            model_path: Path for saving/loading the model.
        """
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH

        self._model: IsolationForest | None = None
        self._is_fitted = False
        self._threshold: float = 0.0
        self._threshold_info: dict[str, Any] = {}

        logger.info(
            "anomaly_detector_initialized",
            contamination=contamination,
            n_estimators=n_estimators,
        )

    @property
    def is_loaded(self) -> bool:
        """Check if a trained model is available."""
        return self._is_fitted and self._model is not None

    def fit(self, X: np.ndarray, _y: np.ndarray | None = None) -> "AnomalyDetector":
        """
        Train the anomaly detector.

        Args:
            X: Training data of shape (n_samples, n_features).
            _y: Ignored. Present for API consistency.

        Returns:
            self
        """
        logger.info("anomaly_detector_training", n_samples=X.shape[0], n_features=X.shape[1])

        self._model = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            random_state=self.random_state,
            n_jobs=-1,  # Use all cores
        )

        self._model.fit(X)
        self._is_fitted = True

        # Calculate threshold based on training data
        scores = self._model.decision_function(X)
        self._threshold = np.percentile(scores, self.contamination * 100)

        logger.info(
            "anomaly_detector_trained",
            threshold=float(self._threshold),
            score_mean=float(np.mean(scores)),
            score_std=float(np.std(scores)),
        )

        return self

    def predict(self, X: np.ndarray) -> dict[str, Any]:
        """
        Predict anomaly score for a single sample or batch.

        Args:
            X: Feature vector(s) of shape (n_features,) or (n_samples, n_features).

        Returns:
            Dictionary containing:
            - score: Anomaly score normalized to [0, 1] where 1 = most anomalous
            - is_anomaly: Boolean indicating if sample is anomalous
            - raw_score: Original Isolation Forest decision score
            - confidence: Confidence in the anomaly detection

        Raises:
            RuntimeError: If model is not trained/loaded.
        """
        if not self.is_loaded:
            raise RuntimeError("Model not trained or loaded. Call fit() or load() first.")

        assert self._model is not None  # Type narrowing for mypy

        # Ensure 2D input
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Get raw decision scores
        raw_scores = self._model.decision_function(X)

        # Use optimized threshold instead of model's default predict
        # Lower scores = more anomalous in Isolation Forest
        predictions = np.where(raw_scores < self._threshold, -1, 1)

        # Normalize scores to [0, 1] where 1 = most anomalous
        # Isolation Forest: lower scores = more anomalous
        # We invert and scale to [0, 1]
        normalized_scores = self._normalize_scores(raw_scores)

        # For single sample
        if len(X) == 1:
            return {
                "score": float(normalized_scores[0]),
                "is_anomaly": bool(predictions[0] == -1),
                "raw_score": float(raw_scores[0]),
                "confidence": self._calculate_confidence(raw_scores[0]),
            }

        # For batch
        return {
            "scores": normalized_scores.tolist(),
            "is_anomaly": (predictions == -1).tolist(),
            "raw_scores": raw_scores.tolist(),
            "confidence": [self._calculate_confidence(s) for s in raw_scores],
        }

    def _normalize_scores(self, raw_scores: np.ndarray) -> np.ndarray:
        """
        Normalize Isolation Forest scores to [0, 1].

        Isolation Forest decision_function returns scores where:
        - Negative values = anomalies
        - Positive values = normal
        - Zero = threshold

        We transform to: 0 = normal, 1 = anomaly
        """
        # Typical score range is approximately [-0.5, 0.5]
        # We use sigmoid-like transformation
        normalized = 1 / (1 + np.exp(raw_scores * 5))
        result: np.ndarray = np.clip(normalized, 0, 1)
        return result

    def _calculate_confidence(self, raw_score: float) -> float:
        """
        Calculate confidence in the anomaly detection.

        Higher absolute scores = higher confidence.
        """
        # Distance from threshold indicates confidence
        distance = abs(raw_score - self._threshold)
        confidence = min(1.0, distance * 5)  # Scale factor
        return float(confidence)

    def save(self, path: Path | str | None = None) -> None:
        """
        Save the trained model to disk.

        Args:
            path: Output path. Uses default if not specified.
        """
        if not self.is_loaded:
            raise RuntimeError("No trained model to save.")

        save_path = Path(path) if path else self.model_path

        # Create directory if needed
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Save model and metadata
        model_data = {
            "model": self._model,
            "threshold": self._threshold,
            "threshold_info": self._threshold_info,
            "contamination": self.contamination,
            "n_estimators": self.n_estimators,
            "version": ARTIFACT_VERSION,
        }

        joblib.dump(model_data, save_path)
        logger.info("anomaly_detector_saved", path=str(save_path))

    def load(self, path: Path | str | None = None) -> "AnomalyDetector":
        """
        Load a trained model from disk.

        Args:
            path: Model path. Uses default if not specified.

        Every field is read with ``.get()`` so that a 1.0.0 artifact — which has
        no ``threshold_info`` — loads unchanged.

        Returns:
            self

        Raises:
            FileNotFoundError: If model file doesn't exist.
        """
        load_path = Path(path) if path else self.model_path

        if not load_path.exists():
            raise FileNotFoundError(f"Model not found at {load_path}")

        model_data = joblib.load(load_path)

        self._model = model_data["model"]
        self._threshold = model_data.get("threshold", 0.0)
        self._threshold_info = model_data.get("threshold_info") or {}
        # The hyperparameters travel with the artifact: without restoring them
        # get_params() would report the constructor's defaults for a model that
        # was trained with entirely different ones.
        self.contamination = model_data.get("contamination", self.contamination)
        self.n_estimators = model_data.get("n_estimators", self.n_estimators)
        self._is_fitted = True

        logger.info(
            "anomaly_detector_loaded",
            path=str(load_path),
            threshold=self._threshold,
            version=model_data.get("version", "unknown"),
            has_operating_point=bool(self._threshold_info),
        )

        return self

    def get_params(self) -> dict[str, Any]:
        """Get model parameters."""
        return {
            "contamination": self.contamination,
            "n_estimators": self.n_estimators,
            "max_samples": self.max_samples,
            "random_state": self.random_state,
            "is_fitted": self._is_fitted,
            "threshold": self._threshold,
        }

    def optimize_threshold_for_recall(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
        target_recall: float = 0.8,
    ) -> float:
        """
        Optimize threshold to achieve target recall for anomaly detection.

        For security systems, high recall is critical - we prefer false positives
        over missing actual attacks.

        Args:
            X_val: Validation features.
            y_val: Validation labels (0=normal, 1=anomaly/attack).
            target_recall: Minimum recall to achieve (default 0.8 = 80%).

        Returns:
            Optimized threshold value.
        """
        if not self.is_loaded:
            raise RuntimeError("Model not trained. Call fit() first.")

        assert self._model is not None

        from sklearn.metrics import precision_recall_curve

        # Get raw decision scores (lower = more anomalous in Isolation Forest)
        raw_scores = self._model.decision_function(X_val)

        # Invert scores so higher = more anomalous (for precision_recall_curve)
        anomaly_scores = -raw_scores

        # Calculate precision-recall curve
        precisions, recalls, thresholds = precision_recall_curve(y_val, anomaly_scores)

        # Find threshold that achieves target recall with best precision
        valid_indices = np.where(recalls[:-1] >= target_recall)[0]

        if len(valid_indices) == 0:
            # Target recall not achievable, use threshold with max recall
            logger.warning(
                "target_recall_not_achievable",
                target=target_recall,
                max_recall=float(recalls.max()),
            )
            optimal_idx = np.argmax(recalls[:-1])
        else:
            # Among thresholds meeting target recall, pick highest precision
            precision_at_valid = precisions[:-1][valid_indices]
            optimal_idx = valid_indices[np.argmax(precision_at_valid)]

        # Convert back to original score space (negate again)
        optimal_threshold = -thresholds[optimal_idx]

        logger.info(
            "threshold_optimized",
            original_threshold=float(self._threshold),
            optimal_threshold=float(optimal_threshold),
            recall_at_threshold=float(recalls[optimal_idx]),
            precision_at_threshold=float(precisions[optimal_idx]),
            target_recall=target_recall,
        )

        self._threshold = optimal_threshold
        return float(optimal_threshold)

    def optimize_threshold_fbeta(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
        beta: float = 2.0,
        n_candidates: int | None = None,
    ) -> float:
        """
        Pick the decision-function threshold maximising F-beta on ``X_val``.

        The alternative to :meth:`optimize_threshold_for_recall`: instead of
        pinning recall to a number the operator has to guess, it makes the
        precision/recall trade-off a single explicit knob.

        * ``beta > 1`` penalises false negatives — fewer missed attacks.
        * ``beta < 1`` penalises false positives — fewer benign alarms.

        Every operating point the validation set actually offers is evaluated,
        so the threshold is the exact maximiser; see
        :func:`ml.evaluation.metrics.best_fbeta_threshold`. The decision rule at
        threshold ``t`` is the same one :meth:`predict` applies::

            score(x) <  t  →  anomaly (positive class, 1)
            score(x) >= t  →  normal  (negative class, 0)

        Args:
            X_val: Validation features of shape ``(n_samples, n_features)``.
            y_val: Validation labels. **Two shapes are accepted**, told apart by
                dtype: a numeric array of 0/1 (0 = normal, 1 = attack), which is
                what the event track produces, or an array of string class names,
                in which case anything other than ``"normal"`` counts as the
                positive class. The flow track's detector only ever sees the
                latter; here both occur, so both are supported.
            beta: Recall weight relative to precision.
            n_candidates: Optional cap on the operating points evaluated. None
                evaluates all of them.

        Returns:
            The selected threshold, also stored on the instance.

        Raises:
            RuntimeError: The model has not been trained or loaded.
        """
        if not self.is_loaded:
            raise RuntimeError("Model not trained. Call fit() first.")

        assert self._model is not None

        scores = self._model.decision_function(X_val)
        y_binary = _binarize_labels(y_val)

        best_t, best_fbeta = best_fbeta_threshold(
            scores, y_binary, beta=beta, n_candidates=n_candidates
        )

        self._threshold = best_t
        y_pred_best = (scores < best_t).astype(int)
        self._threshold_info = self._compute_operating_info(y_binary, y_pred_best, best_fbeta, beta)

        info = self._threshold_info
        logger.info(
            "threshold_optimized_fbeta",
            beta=beta,
            threshold=best_t,
            fbeta=round(best_fbeta, 6),
            precision=info["precision"],
            recall=info["recall"],
            fpr=info["fpr"],
            fnr=info["fnr"],
        )
        return best_t

    def get_operating_point_info(self) -> dict[str, Any]:
        """
        Return the metrics measured at the active threshold.

        Returns:
            Dictionary with ``threshold``, ``beta``, ``f<beta>_score``,
            ``precision``, ``recall``, ``fpr``, ``fnr`` and the raw
            ``tp``/``fp``/``tn``/``fn`` counts. A copy, so callers cannot mutate
            the detector's state.

        Raises:
            RuntimeError: :meth:`optimize_threshold_fbeta` has never run and no
                saved operating point was restored, so no metrics exist. Fitting
                alone is not enough — the percentile threshold ``fit`` computes
                has never been measured against any label.
        """
        if not self._threshold_info:
            raise RuntimeError(
                "No operating-point info available. "
                "Call optimize_threshold_fbeta() on a validation set first."
            )
        return dict(self._threshold_info)

    def _compute_operating_info(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        fbeta: float,
        beta: float,
    ) -> dict[str, Any]:
        """Derive the full confusion-matrix diagnostics at the chosen threshold."""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
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
