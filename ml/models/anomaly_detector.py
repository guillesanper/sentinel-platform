"""
Anomaly detection model using Isolation Forest.

Detects unusual patterns in security events that may indicate
malicious activity or novel attack vectors.
"""

import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
import structlog

logger = structlog.get_logger()

# Default model path
DEFAULT_MODEL_PATH = Path("models/anomaly_detector.joblib")


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

        logger.info(
            "anomaly_detector_initialized",
            contamination=contamination,
            n_estimators=n_estimators,
        )

    @property
    def is_loaded(self) -> bool:
        """Check if a trained model is available."""
        return self._is_fitted and self._model is not None

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "AnomalyDetector":
        """
        Train the anomaly detector.

        Args:
            X: Training data of shape (n_samples, n_features).
            y: Ignored. Present for API consistency.

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

        # Ensure 2D input
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Get raw decision scores
        raw_scores = self._model.decision_function(X)
        predictions = self._model.predict(X)  # 1 = normal, -1 = anomaly

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
        return np.clip(normalized, 0, 1)

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
            "contamination": self.contamination,
            "n_estimators": self.n_estimators,
            "version": "1.0.0",
        }

        joblib.dump(model_data, save_path)
        logger.info("anomaly_detector_saved", path=str(save_path))

    def load(self, path: Path | str | None = None) -> "AnomalyDetector":
        """
        Load a trained model from disk.

        Args:
            path: Model path. Uses default if not specified.

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
        self._is_fitted = True

        logger.info(
            "anomaly_detector_loaded",
            path=str(load_path),
            threshold=self._threshold,
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
