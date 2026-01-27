"""
Attack classification model using XGBoost.

Multi-class classifier that identifies specific attack types
from security events.
"""

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import structlog
import xgboost as xgb

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
IDX_TO_LABEL = dict(enumerate(ATTACK_TYPES))


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
        class_weight: str | None = "balanced",
    ) -> None:
        """
        Initialize the attack classifier.

        Args:
            n_estimators: Number of boosting rounds.
            max_depth: Maximum depth of each tree.
            learning_rate: Boosting learning rate.
            random_state: Random seed for reproducibility.
            model_path: Path for saving/loading the model.
            class_weight: Strategy for class weighting. 'balanced' computes weights
                inversely proportional to class frequencies. None disables weighting.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        self.class_weight = class_weight

        self._model: xgb.XGBClassifier | None = None
        self._is_fitted = False
        self._classes = ATTACK_TYPES

        logger.info(
            "attack_classifier_initialized",
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_classes=len(ATTACK_TYPES),
            class_weight=class_weight,
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

        # Detect number of classes from data
        n_classes = len(np.unique(y_encoded))
        self._n_classes = n_classes

        # Use binary or multi-class objective based on number of classes
        if n_classes == 2:
            objective = "binary:logistic"
            eval_metric = "logloss"
            model_params = {
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
                "learning_rate": self.learning_rate,
                "random_state": self.random_state,
                "objective": objective,
                "use_label_encoder": False,
                "eval_metric": eval_metric,
                "n_jobs": -1,
            }
        else:
            objective = "multi:softprob"
            eval_metric = "mlogloss"
            model_params = {
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
                "learning_rate": self.learning_rate,
                "random_state": self.random_state,
                "objective": objective,
                "num_class": n_classes,
                "use_label_encoder": False,
                "eval_metric": eval_metric,
                "n_jobs": -1,
            }

        logger.info("classifier_config", n_classes=n_classes, objective=objective)

        # Initialize model
        self._model = xgb.XGBClassifier(**model_params)

        # Fit with optional early stopping
        fit_params: dict[str, Any] = {}
        if eval_set:
            # Encode eval set labels
            encoded_eval_set = [
                (X_eval, self._encode_labels(y_eval)) for X_eval, y_eval in eval_set
            ]
            fit_params["eval_set"] = encoded_eval_set
            fit_params["verbose"] = False

        # Calculate sample weights for class balancing
        if self.class_weight == "balanced":
            from sklearn.utils.class_weight import compute_sample_weight

            sample_weight = compute_sample_weight("balanced", y_encoded)
            fit_params["sample_weight"] = sample_weight
            logger.info(
                "class_weighting_applied",
                strategy=self.class_weight,
                weight_range=(float(sample_weight.min()), float(sample_weight.max())),
            )

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

        assert self._model is not None  # Type narrowing for mypy

        # Ensure 2D input
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Get predictions and probabilities
        predictions = self._model.predict(X)
        probabilities = self._model.predict_proba(X)

        # Get number of classes from model
        n_classes = getattr(self, "_n_classes", len(ATTACK_TYPES))

        # For single sample
        if len(X) == 1:
            pred_idx = int(np.asarray(predictions).flatten()[0])
            probs = probabilities[0]

            # Build probabilities dict only for classes in model
            prob_dict = {}
            for idx in range(n_classes):
                if idx < len(probs):
                    prob_dict[IDX_TO_LABEL[idx]] = float(probs[idx])

            return {
                "attack_type": IDX_TO_LABEL[pred_idx],
                "confidence": float(probs[pred_idx]) if pred_idx < len(probs) else 0.0,
                "probabilities": prob_dict,
            }

        # For batch
        results = []
        for pred, probs in zip(predictions, probabilities, strict=True):
            idx = int(np.asarray(pred).flatten()[0]) if hasattr(pred, "__len__") else int(pred)
            prob_dict = {}
            for class_idx in range(n_classes):
                if class_idx < len(probs):
                    prob_dict[IDX_TO_LABEL[class_idx]] = float(probs[class_idx])

            results.append(
                {
                    "attack_type": IDX_TO_LABEL[idx],
                    "confidence": float(probs[idx]) if idx < len(probs) else 0.0,
                    "probabilities": prob_dict,
                }
            )

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

        assert self._model is not None  # Type narrowing for mypy

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
        if not self.is_loaded or self._model is None:
            return []

        importances = self._model.feature_importances_
        top_indices = np.argsort(importances)[-top_k:][::-1]

        return [(int(idx), float(importances[idx])) for idx in top_indices]

    def get_feature_importance(self) -> dict[int, float]:
        """Get all feature importances."""
        if not self.is_loaded or self._model is None:
            return {}

        return {i: float(imp) for i, imp in enumerate(self._model.feature_importances_)}

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
            "n_classes": getattr(self, "_n_classes", len(ATTACK_TYPES)),
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
        self._n_classes = model_data.get("n_classes", len(ATTACK_TYPES))
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
