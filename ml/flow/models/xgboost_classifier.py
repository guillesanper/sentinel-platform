"""
Stage 2 attack-type classifier for the flow-based track.

Receives the flows Stage 1 flagged as anomalous and names the attack family
among the canonical taxonomy of :mod:`ml.flow.labels`. Supports a
randomised hyperparameter search and gradient-boosting early stopping on a
held-out validation set. Class imbalance — CICIDS2017 is ~80% BENIGN, and
Botnet is three orders of magnitude rarer than DoS — is handled with
inverse-frequency sample weights rather than resampling, so no synthetic flow
ever reaches the trees.

Named ``FlowAttackClassifier``, not ``AttackClassifier``: the event-based
classifier in ``ml.models.classifier`` maps any unknown label to ``normal``
*silently*, so feeding it CICIDS labels would yield a model that predicts
``normal`` for everything without raising anything. Separate names, separate
modules, no accidental import.
"""

from pathlib import Path
from typing import Any, Final

import joblib
import numpy as np
import structlog
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from ml.flow.config import XGBoostConfig

logger = structlog.get_logger()


# ============================================================================
# Persistence contract
# ============================================================================

MODEL_FILE: Final[str] = "xgboost.joblib"
ENCODER_FILE: Final[str] = "xgb_label_encoder.joblib"

# Multiclass eval metrics and their binary counterparts. Used when Stage 1
# routes only two attack families and the booster switches objective.
_BINARY_EQUIVALENT_METRIC: Final[dict[str, str]] = {
    "mlogloss": "logloss",
    "merror": "error",
}


class FlowAttackClassifier:
    """
    XGBoost multi-class classifier over CICFlowMeter features.

    Attributes:
        config: Hyperparameters and search space.
        random_seed: Seed forwarded to XGBoost and the CV splitter.

    Example:
        >>> clf = FlowAttackClassifier(XGBoostConfig())
        >>> clf.fit(X_train, y_train, X_val, y_val)   # doctest: +SKIP
        >>> clf.predict(X_test)                        # doctest: +SKIP
    """

    def __init__(self, config: XGBoostConfig, random_seed: int = 42) -> None:
        """
        Args:
            config: XGBoost hyperparameters and RandomizedSearchCV space.
            random_seed: Seed for the booster and the cross-validation folds.
        """
        self.config = config
        self.random_seed = random_seed
        self._model: XGBClassifier | None = None
        self._label_encoder: LabelEncoder = LabelEncoder()
        self._encoder_fitted: bool = False
        self._best_params: dict[str, Any] = {}
        self._n_classes: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def tune_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray) -> dict[str, Any]:
        """
        Search for hyperparameters with ``RandomizedSearchCV``.

        A ``StratifiedKFold`` splitter keeps the attack-class distribution in
        every fold. The search itself uses no early stopping; :meth:`fit`
        applies it afterwards on the real validation set.

        **The folds are shuffled, so this search is not temporally clean.**
        ``X_train`` is already the chronologically early slice of each block, but
        within it the folds are drawn at random, which lets near-duplicates from
        one attack burst sit on both sides of a fold boundary — the very leakage
        ``ml.flow.loader`` partitions to avoid. The consequence is bounded: it
        can bias *which* hyperparameters win, never the reported metrics, which
        are measured on a test set this function never sees. Making the folds
        temporal (``order_idx`` does not reach here) would be the strictly
        correct thing; it is not done because the reference runs use
        ``--skip-tuning`` and the gain would be invisible next to the effects
        documented in ``docs/ml/PARTICION-TEMPORAL.md``.

        Args:
            X_train: Feature matrix of the anomalous training flows.
            y_train: Their canonical string attack labels.

        Returns:
            The best hyperparameter values found, also stored on the instance
            so that the next :meth:`fit` picks them up.
        """
        y_enc = self._fit_and_encode(y_train)
        sample_weight = self._compute_sample_weights(y_enc)

        base_model = XGBClassifier(
            **self._objective_params(),
            tree_method=self.config.tree_method,
            n_jobs=self.config.n_jobs,
            random_state=self.random_seed,
        )

        cv = StratifiedKFold(
            n_splits=self.config.cv_folds,
            shuffle=True,
            random_state=self.random_seed,
        )

        search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=self.config.search_space,
            n_iter=self.config.n_iter_search,
            scoring="f1_macro",
            cv=cv,
            # 1, not -1: XGBoost already parallelises internally, and nesting
            # the two thread pools oversubscribes every core.
            n_jobs=1,
            random_state=self.random_seed,
            verbose=0,
            refit=False,
        )

        logger.info(
            "flow_attack_classifier_tuning_started",
            n_iter=self.config.n_iter_search,
            cv_folds=self.config.cv_folds,
            n_samples=int(X_train.shape[0]),
        )
        search.fit(X_train, y_enc, sample_weight=sample_weight)

        self._best_params = dict(search.best_params_)
        logger.info(
            "flow_attack_classifier_tuned",
            best_params=self._best_params,
            f1_macro=round(float(search.best_score_), 6),
        )
        return self._best_params

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> "FlowAttackClassifier":
        """
        Train the final booster with early stopping on the validation set.

        Hyperparameters found by :meth:`tune_hyperparameters`, if it ran, take
        precedence over the config defaults.

        Validation rows whose class never appeared among the training anomalies
        are dropped from the early-stopping eval set: the label encoder cannot
        represent them, and keeping them would abort the fit.

        Args:
            X_train: Feature matrix of the anomalous training flows.
            y_train: Their canonical string attack labels.
            X_val: Feature matrix of the anomalous validation flows.
            y_val: Their canonical string attack labels.

        Returns:
            self
        """
        y_enc = self._fit_and_encode(y_train)

        known_mask = np.isin(y_val, self._label_encoder.classes_)
        if not known_mask.all():
            logger.warning(
                "flow_attack_classifier_val_classes_unseen",
                n_excluded=int((~known_mask).sum()),
                labels=sorted({str(v) for v in y_val[~known_mask]}),
                hint="classes absent from the training anomalies cannot be scored",
            )
            X_val = X_val[known_mask]
            y_val = y_val[known_mask]

        y_val_enc = self._label_encoder.transform(y_val).astype(np.int32)
        sample_weight = self._compute_sample_weights(y_enc)

        params = self._merged_params()
        self._model = XGBClassifier(
            **self._objective_params(),
            early_stopping_rounds=self.config.early_stopping_rounds,
            tree_method=self.config.tree_method,
            n_jobs=self.config.n_jobs,
            random_state=self.random_seed,
            **params,
        )

        logger.info(
            "flow_attack_classifier_training",
            n_classes=self._n_classes,
            n_samples=int(X_train.shape[0]),
            params=params,
        )
        self._model.fit(
            X_train,
            y_enc,
            sample_weight=sample_weight,
            eval_set=[(X_val, y_val_enc)],
            verbose=False,
        )
        logger.info(
            "flow_attack_classifier_trained",
            best_iteration=int(self._model.best_iteration),
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict canonical attack labels for anomalous flows.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.

        Returns:
            String labels of shape ``(n_samples,)``.

        Raises:
            RuntimeError: The classifier has not been fitted or loaded.
        """
        self._check_fitted()
        assert self._model is not None

        y_enc = self._model.predict(X)
        labels: np.ndarray = self._label_encoder.inverse_transform(y_enc)
        return labels

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return per-class probability estimates.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.

        Returns:
            Array of shape ``(n_samples, n_classes)``, columns ordered as
            :attr:`classes_`.

        Raises:
            RuntimeError: The classifier has not been fitted or loaded.
        """
        self._check_fitted()
        assert self._model is not None

        probs: np.ndarray = self._model.predict_proba(X)
        return probs

    def get_feature_importance(self, feature_names: list[str] | None = None) -> dict[str, float]:
        """
        Return feature importances, highest first.

        Args:
            feature_names: Column names in the same order as the training
                matrix. When None, keys fall back to ``f0``, ``f1``, ...

        Returns:
            ``{feature_name: importance}`` sorted by importance descending.

        Raises:
            RuntimeError: The classifier has not been fitted or loaded.
            ValueError: ``feature_names`` has a different length than the
                model's feature count — which would silently mislabel every
                importance, so it fails loudly instead.
        """
        self._check_fitted()
        assert self._model is not None

        importances: np.ndarray = self._model.feature_importances_
        n = len(importances)

        if feature_names is None:
            names = [f"f{i}" for i in range(n)]
        elif len(feature_names) != n:
            raise ValueError(
                f"feature_names has {len(feature_names)} entries but the model was "
                f"trained on {n} features."
            )
        else:
            names = list(feature_names)

        pairs = zip(names, importances.tolist(), strict=True)
        return dict(sorted(pairs, key=lambda kv: kv[1], reverse=True))

    @property
    def classes_(self) -> np.ndarray:
        """
        String class labels in label-encoder order.

        Raises:
            RuntimeError: The encoder has not been fitted.
        """
        if not self._encoder_fitted:
            raise RuntimeError("Classifier has not been fitted yet.")
        classes: np.ndarray = self._label_encoder.classes_
        return classes

    @property
    def is_fitted(self) -> bool:
        """Whether the classifier can predict (fitted or loaded from disk)."""
        return self._model is not None

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, directory: Path | str) -> None:
        """
        Serialise the booster and its label encoder.

        Both go together on purpose: the booster predicts integers, and without
        the encoder that produced them the integers are meaningless.

        Args:
            directory: Target directory, created recursively if absent.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        joblib.dump(self._model, directory / MODEL_FILE)
        joblib.dump(self._label_encoder, directory / ENCODER_FILE)

        logger.info("flow_attack_classifier_saved", path=str(directory))

    def load(self, directory: Path | str) -> "FlowAttackClassifier":
        """
        Restore the booster and its label encoder.

        Args:
            directory: Directory previously written by :meth:`save`.

        Returns:
            self

        Raises:
            FileNotFoundError: Either artifact is missing.
        """
        directory = Path(directory)

        self._model = joblib.load(directory / MODEL_FILE)
        self._label_encoder = joblib.load(directory / ENCODER_FILE)
        self._encoder_fitted = True
        self._n_classes = len(self._label_encoder.classes_)

        logger.info(
            "flow_attack_classifier_loaded",
            path=str(directory),
            n_classes=self._n_classes,
        )
        return self

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fit_and_encode(self, y: np.ndarray) -> np.ndarray:
        """
        Fit the label encoder on first use and return integer-encoded labels.

        Fitted once and only once: :meth:`tune_hyperparameters` and :meth:`fit`
        both call this, and a re-fit between them would renumber the classes
        under the tuned parameters.
        """
        if not self._encoder_fitted:
            self._label_encoder.fit(y)
            self._encoder_fitted = True
            self._n_classes = len(self._label_encoder.classes_)
        encoded: np.ndarray = self._label_encoder.transform(y).astype(np.int32)
        return encoded

    def _compute_sample_weights(self, y_enc: np.ndarray) -> np.ndarray:
        """
        Balanced inverse-frequency weights, normalised to mean 1.

        ``w_c = 1 / (freq_c * n_classes)``, i.e. sklearn's ``'balanced'``
        strategy expressed per sample.
        """
        _, counts = np.unique(y_enc, return_counts=True)
        freq = counts / counts.sum()
        weight_per_class = 1.0 / (freq * self._n_classes)
        weight_per_class /= weight_per_class.mean()
        weights: np.ndarray = weight_per_class[y_enc]
        return weights

    def _objective_params(self) -> dict[str, Any]:
        """
        Pick the XGBoost objective and eval metric for the routed class count.

        Forcing ``multi:softprob`` unconditionally — as the original TFA code
        did, where Stage 2 always saw all six attack families — breaks the day
        of the first two-class run: with two classes ``predict()`` returns a
        ``(n, 2)`` probability matrix instead of class indices, and
        ``inverse_transform`` rejects it. Two classes is not a hypothetical: a
        day-holdout split, or any deployment where Stage 1 routes only two
        families, lands here. The eval metric has to follow the objective, or
        early stopping scores a multiclass loss against a binary booster.
        """
        if self._n_classes > 2:
            return {
                "objective": "multi:softprob",
                "num_class": self._n_classes,
                "eval_metric": self.config.eval_metric,
            }
        return {
            "objective": "binary:logistic",
            "eval_metric": _BINARY_EQUIVALENT_METRIC.get(
                self.config.eval_metric, self.config.eval_metric
            ),
        }

    def _merged_params(self) -> dict[str, Any]:
        """Config defaults overridden by whatever the search found."""
        defaults: dict[str, Any] = {
            "n_estimators": self.config.n_estimators,
            "max_depth": self.config.max_depth,
            "learning_rate": self.config.learning_rate,
            "subsample": self.config.subsample,
            "colsample_bytree": self.config.colsample_bytree,
            "min_child_weight": self.config.min_child_weight,
            "gamma": self.config.gamma,
            "reg_alpha": self.config.reg_alpha,
            "reg_lambda": self.config.reg_lambda,
        }
        return {**defaults, **self._best_params}

    def _check_fitted(self) -> None:
        """Raise when the classifier cannot predict yet."""
        if self._model is None:
            raise RuntimeError("Model must be fit or loaded before calling this method.")
