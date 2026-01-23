"""
Attack classification model using XGBoost.

Multi-class classifier that identifies specific attack types
from security events.
"""

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import xgboost as xgb
import structlog

logger = structlog.get_logger()

# Default model path
DEFAULT_MODEL_PATH = Path("models/attack_classifier.joblib")

# Attack type labels
ATTACK_TYPES = [
    "normal",
    "brute_force",
    "credential_stuffing",
    "privilege_escalation",
    "lateral_movement",
    "data_exfiltration",
    "reconnaissance",
]

# Label to index mapping
LABEL_TO_IDX = {label: idx for idx, label in enumerate(ATTACK_TYPES)}
IDX_TO_LABEL = {idx: label for idx, label in enumerate(ATTACK_TYPES)}


class AttackClassifier:
    """
    Multi-class attack classifier using XGBoost.

    Classifies security events into one of the following categories:
    - normal: Benign activity
    - brute_force: Multiple failed login attempts
    - credential_stuffing: Multiple users from same IP
    - privilege_escalation: Unauthorized privilege gain
    - lateral_movement: Internal network traversal
    - data_exfiltration: Unusual data transfers
    - reconnaissance: Port scanning, enumeration

    Attributes:
        n_estimators: Number of boosting rounds.
        max_depth: Maximum tree depth.
        is_loaded: Whether a trained model has been loaded.

    Example:
        >>> classifier = AttackClassifier()
        >>> classifier.fit(X_train, y_train)
        >>> result = classifier.predict(X_test[0])
        >>> print(f"Attack type: {result['attack_type']}")
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        random_state: int = 42,
        model_path: Path | str | None = None,
    ) -> None:
        """
        Initialize the attack classifier.

        Args:
            n_estimators: Number of boosting rounds.
            max_depth: Maximum depth of each tree.
            learning_rate: Boosting learning rate.
            random_state: Random seed for reproducibility.
            model_path: Path for saving/loading the model.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH

        self._model: xgb.XGBClassifier | None = None
        self._is_fitted = False
        self._classes = ATTACK_TYPES

        logger.info(
            "attack_classifier_initialized",
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_classes=len(ATTACK_TYPES),
        )

    @property
    def is_loaded(self) -> bool:
        """Check if a trained model is available."""
        return self._is_fitted and self._model is not None

    @property
    def classes(self) -> list[str]:
        """Get list of attack type classes."""
        return self._classes.copy()

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray | list[str],
        eval_set: list[tuple[np.ndarray, np.ndarray]] | None = None,
    ) -> "AttackClassifier":
        """
        Train the attack classifier.

        Args:
            X: Training data of shape (n_samples, n_features).
            y: Target labels (either indices or string labels).
            eval_set: Optional validation set for early stopping.

        Returns:
            self
        """
        logger.info(
            "attack_classifier_training",
            n_samples=X.shape[0],
            n_features=X.shape[1],
        )

        # Convert string labels to indices if needed
        y_encoded = self._encode_labels(y)

        # Initialize model
        self._model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
            objective="multi:softprob",
            num_class=len(ATTACK_TYPES),
            use_label_encoder=False,
            eval_metric="mlogloss",
            n_jobs=-1,
        )

        # Fit with optional early stopping
        fit_params: dict[str, Any] = {}
        if eval_set:
            # Encode eval set labels
            encoded_eval_set = [
                (X_eval, self._encode_labels(y_eval))
                for X_eval, y_eval in eval_set
            ]
            fit_params["eval_set"] = encoded_eval_set
            fit_params["verbose"] = False

        self._model.fit(X, y_encoded, **fit_params)
        self._is_fitted = True

        logger.info(
            "attack_classifier_trained",
            feature_importances=self._get_top_features(5),
        )

        return self

    def predict(self, X: np.ndarray) -> dict[str, Any]:
        """
        Predict attack type for a single sample or batch.

        Args:
            X: Feature vector(s) of shape (n_features,) or (n_samples, n_features).

        Returns:
            Dictionary containing:
            - attack_type: Predicted attack type string
            - confidence: Confidence in the prediction (probability)
            - probabilities: Dict of all class probabilities

        Raises:
            RuntimeError: If model is not trained/loaded.
        """
        if not self.is_loaded:
            raise RuntimeError("Model not trained or loaded. Call fit() or load() first.")

        # Ensure 2D input
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Get predictions and probabilities
        predictions = self._model.predict(X)
        probabilities = self._model.predict_proba(X)

        # For single sample
        if len(X) == 1:
            pred_idx = int(predictions[0])
            probs = probabilities[0]

            return {
                "attack_type": IDX_TO_LABEL[pred_idx],
                "confidence": float(probs[pred_idx]),
                "probabilities": {
                    label: float(probs[idx])
                    for label, idx in LABEL_TO_IDX.items()
                },
            }

        # For batch
        results = []
        for i, (pred_idx, probs) in enumerate(zip(predictions, probabilities)):
            pred_idx = int(pred_idx)
            results.append({
                "attack_type": IDX_TO_LABEL[pred_idx],
                "confidence": float(probs[pred_idx]),
                "probabilities": {
                    label: float(probs[idx])
                    for label, idx in LABEL_TO_IDX.items()
                },
            })

        return {"results": results}

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get class probabilities.

        Args:
            X: Feature vector(s).

        Returns:
            Probability array of shape (n_samples, n_classes).
        """
        if not self.is_loaded:
            raise RuntimeError("Model not trained or loaded.")

        if X.ndim == 1:
            X = X.reshape(1, -1)

        return self._model.predict_proba(X)

    def _encode_labels(self, y: np.ndarray | list) -> np.ndarray:
        """Convert string labels to integer indices."""
        if isinstance(y, np.ndarray) and y.dtype in (np.int32, np.int64):
            return y

        encoded = []
        for label in y:
            if isinstance(label, str):
                encoded.append(LABEL_TO_IDX.get(label, 0))
            else:
                encoded.append(int(label))

        return np.array(encoded, dtype=np.int32)

    def _get_top_features(self, top_k: int = 5) -> list[tuple[int, float]]:
        """Get top feature importances."""
        if not self.is_loaded:
            return []

        importances = self._model.feature_importances_
        top_indices = np.argsort(importances)[-top_k:][::-1]

        return [(int(idx), float(importances[idx])) for idx in top_indices]

    def get_feature_importance(self) -> dict[int, float]:
        """Get all feature importances."""
        if not self.is_loaded:
            return {}

        return {
            i: float(imp)
            for i, imp in enumerate(self._model.feature_importances_)
        }

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
            "classes": self._classes,
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "version": "1.0.0",
        }

        joblib.dump(model_data, save_path)
        logger.info("attack_classifier_saved", path=str(save_path))

    def load(self, path: Path | str | None = None) -> "AttackClassifier":
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
        self._classes = model_data.get("classes", ATTACK_TYPES)
        self._is_fitted = True

        logger.info("attack_classifier_loaded", path=str(load_path))

        return self

    def get_params(self) -> dict[str, Any]:
        """Get model parameters."""
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "random_state": self.random_state,
            "is_fitted": self._is_fitted,
            "classes": self._classes,
        }
