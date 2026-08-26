"""
Isolation Forest → XGBoost cascade. The production path of the flow track.

Training:
    1. Fit the Isolation Forest on the BENIGN flows only.
    2. Tune its threshold for F-beta on the validation set.
    3. Route the flows it flags as anomalous to Stage 2.
    4. Optionally search XGBoost hyperparameters on those anomalies.
    5. Fit XGBoost with early stopping on the anomalous validation flows.

Inference:
    1. Stage 1 produces a binary anomaly mask.
    2. Stage 2 names the attack family for the masked flows.
    3. Everything else is BENIGN.

Measured on CICIDS2017 (1M flows, **random split**): accuracy 0.9936, F1-macro
0.9516. Those figures are an optimistic bound — with the temporal split this
track now defaults to, expect them to come down, and that drop is the
correction, not a regression.
"""

from pathlib import Path

import numpy as np
import structlog
from sklearn.metrics import accuracy_score, f1_score

from ml.flow.config import FlowConfig
from ml.flow.cv import TemporalIndex
from ml.flow.labels import BENIGN_LABEL
from ml.flow.models.isolation_forest import FlowAnomalyDetector
from ml.flow.models.xgboost_classifier import FlowAttackClassifier
from ml.flow.pipelines.base import BaseCascadePipeline

logger = structlog.get_logger()

# Sub-directories written by save() and expected by load().
IF_SUBDIR = "isolation_forest"
XGB_SUBDIR = "xgboost"

# When Stage 1 flags no validation flow at all, XGBoost still needs a non-empty
# eval_set. This fraction of the training anomalies stands in for one.
_PSEUDO_VAL_FRACTION = 0.10


class CascadeXGBPipeline(BaseCascadePipeline):
    """
    Two-stage cascade: :class:`FlowAnomalyDetector` then
    :class:`FlowAttackClassifier`.

    Example:
        >>> pipeline = CascadeXGBPipeline(FlowConfig())
        >>> pipeline.train(                                  # doctest: +SKIP
        ...     X_train, y_train, X_val, y_val,
        ...     skip_tuning=True, feature_names=feature_names,
        ... )
        >>> pipeline.save("models/flow/cascade_xgb")         # doctest: +SKIP
    """

    def __init__(
        self,
        config: FlowConfig | None = None,
        feature_names: list[str] | None = None,
    ) -> None:
        """
        Args:
            config: Track configuration. Defaults to ``FlowConfig()``.
            feature_names: Column order of the training matrix. Can also be
                supplied later via :meth:`train`; one of the two is required
                before :meth:`save`.
        """
        self._config = config or FlowConfig()
        self._feature_names = list(feature_names) if feature_names is not None else None
        self._if_model = FlowAnomalyDetector(
            config=self._config.isolation_forest,
            random_seed=self._config.random_seed,
        )
        self._xgb = FlowAttackClassifier(
            config=self._config.xgboost,
            random_seed=self._config.random_seed,
        )

    @property
    def name(self) -> str:
        """Identifier used in log events and output paths."""
        return "cascade_xgb"

    # ------------------------------------------------------------------
    # BaseCascadePipeline contract
    # ------------------------------------------------------------------

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
    ) -> "CascadeXGBPipeline":
        """
        Train Stage 1 and Stage 2 in sequence.

        Args:
            X_train: Full training feature matrix, all classes.
            y_train: Canonical string labels for the training set.
            X_val: Validation feature matrix.
            y_val: Canonical string labels for the validation set.
            beta: F-beta weight for the Stage 1 threshold search.
            skip_tuning: Skip the ``RandomizedSearchCV`` pass on Stage 2.
            feature_names: Column order of ``X_train``; overrides whatever was
                given to the constructor.
            train_index: Chronological index over the rows of ``X_train``. Only
                the Stage 2 search reads it, and only when ``skip_tuning`` is
                False; it is narrowed here by the Stage 1 anomaly mask, since
                what reaches Stage 2 is the routed subset.

        Returns:
            self

        Raises:
            RuntimeError: Stage 1 routed no training flow to Stage 2, which
                leaves the classifier nothing to learn from.
        """
        if feature_names is not None:
            self._feature_names = list(feature_names)
        self._check_feature_names(X_train)

        logger.info("cascade_stage1_training", pipeline=self.name, n_train=int(len(y_train)))
        self._if_model.fit(X_train, y_train)
        self._if_model.optimize_threshold(X_val, y_val, beta=beta)

        anomaly_train = self._if_model.predict_binary(X_train).astype(bool)
        X_attack_train = X_train[anomaly_train]
        y_attack_train = y_train[anomaly_train]
        # Deliberately on the same three lines as the matrix it describes. An
        # index narrowed by any other mask still has a plausible length, still
        # produces plausible folds, and yields hyperparameters chosen from
        # nonsense — with no exception anywhere in the run.
        attack_train_index = train_index.mask(anomaly_train) if train_index is not None else None

        anomaly_val = self._if_model.predict_binary(X_val).astype(bool)
        X_attack_val = X_val[anomaly_val]
        y_attack_val = y_val[anomaly_val]

        logger.info(
            "cascade_stage1_routing",
            pipeline=self.name,
            n_train_anomalous=int(anomaly_train.sum()),
            n_train=int(len(y_train)),
            train_anomalous_pct=round(100.0 * anomaly_train.sum() / len(y_train), 2),
            n_val_anomalous=int(anomaly_val.sum()),
            n_val=int(len(y_val)),
            val_anomalous_pct=round(100.0 * anomaly_val.sum() / len(y_val), 2),
        )

        if X_attack_train.shape[0] == 0:
            raise RuntimeError(
                f"[{self.name}] Stage 1 routed 0 anomalous samples to Stage 2. "
                "Check the contamination parameter or the F-beta threshold."
            )

        logger.info("cascade_stage2_training", pipeline=self.name, stage2="xgboost")
        if not skip_tuning:
            self._report_fold_protocol(attack_train_index)
            self._xgb.tune_hyperparameters(
                X_attack_train,
                y_attack_train,
                train_index=attack_train_index,
            )

        if X_attack_val.shape[0] == 0:
            logger.warning(
                "cascade_stage2_pseudo_validation",
                pipeline=self.name,
                fraction=_PSEUDO_VAL_FRACTION,
                hint="Stage 1 flagged no validation flow; early stopping falls back "
                "to a slice of the training anomalies",
            )
            n_pseudo = max(1, int(_PSEUDO_VAL_FRACTION * X_attack_train.shape[0]))
            X_attack_val = X_attack_train[:n_pseudo]
            y_attack_val = y_attack_train[:n_pseudo]

        self._xgb.fit(X_attack_train, y_attack_train, X_attack_val, y_attack_val)
        logger.info("cascade_trained", pipeline=self.name)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Run end-to-end cascade inference.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.

        Returns:
            Canonical string labels of shape ``(n_samples,)``.
        """
        predictions = np.full(X.shape[0], BENIGN_LABEL, dtype=object)
        anomaly_mask = self._if_model.predict_binary(X).astype(bool)
        if anomaly_mask.sum() > 0:
            predictions[anomaly_mask] = self._xgb.predict(X[anomaly_mask])
        labels: np.ndarray = predictions.astype(str)
        return labels

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        *,
        labels: list[str] | None = None,
    ) -> dict[str, float]:
        """
        Score the cascade end to end.

        Args:
            X_test: Test feature matrix.
            y_test: Ground-truth canonical string labels.
            labels: Canonical class list the F1 scores average over. Without it
                sklearn averages over the union of ``y_test`` and ``y_pred``, so
                a class with no test support vanishes from the macro instead of
                scoring 0.0 — the same headline computed two different ways
                depending on the caller. See
                :meth:`BaseCascadePipeline.evaluate`.

        Returns:
            ``accuracy``, ``f1_macro`` and ``f1_weighted``, plus the Stage 1
            operating point (``if_precision``, ``if_recall``, ``if_fpr``,
            ``if_fnr``) when a threshold search has run.

        Note:
            The Stage 1 figures are measured **on ``X_test``**, not on the
            validation set the threshold was chosen from. Reporting the
            selection-set numbers next to test accuracy — which is what this
            method used to do — overstated Stage 1 recall by 18 points on
            CICIDS2017, and Stage 1 recall is the cascade's hard ceiling. See
            :meth:`~ml.flow.models.isolation_forest.FlowAnomalyDetector.measure_operating_point`.
        """
        y_pred = self.predict(X_test)

        metrics: dict[str, float] = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "f1_macro": float(
                f1_score(y_test, y_pred, average="macro", labels=labels, zero_division=0)
            ),
            "f1_weighted": float(
                f1_score(y_test, y_pred, average="weighted", labels=labels, zero_division=0)
            ),
        }

        try:
            # Only for the beta, and to establish that a threshold was in fact
            # chosen — without a search the operating point means nothing.
            selection = self._if_model.get_operating_point_info()
        except RuntimeError:
            # No threshold search ran; the end-to-end metrics stand on their own.
            pass
        else:
            op = self._if_model.measure_operating_point(X_test, y_test, beta=selection["beta"])
            metrics["if_precision"] = op["precision"]
            metrics["if_recall"] = op["recall"]
            metrics["if_fpr"] = op["fpr"]
            metrics["if_fnr"] = op["fnr"]

        self._log_evaluation(metrics)
        return metrics

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, directory: Path | str) -> None:
        """
        Persist both stages and the feature-name contract.

        Layout::

            <directory>/feature_names.json    training-time column order
            <directory>/isolation_forest/     forest, threshold, operating point
            <directory>/xgboost/              booster and label encoder

        Args:
            directory: Target directory, created recursively if absent.

        Raises:
            RuntimeError: No feature names are known — see
                :meth:`BaseCascadePipeline._save_feature_names`.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        self._save_feature_names(directory)
        self._if_model.save(directory / IF_SUBDIR)
        self._xgb.save(directory / XGB_SUBDIR)

        logger.info("cascade_saved", pipeline=self.name, path=str(directory))

    def load(self, directory: Path | str) -> "CascadeXGBPipeline":
        """
        Restore a saved cascade, feature-name contract included.

        Args:
            directory: Directory previously written by :meth:`save`.

        Returns:
            self

        Raises:
            FileNotFoundError: An artifact or the feature-name contract is
                missing.
        """
        directory = Path(directory)

        self._load_feature_names(directory)
        self._if_model.load(directory / IF_SUBDIR)
        self._xgb.load(directory / XGB_SUBDIR)

        logger.info("cascade_loaded", pipeline=self.name, path=str(directory))
        return self

    # ------------------------------------------------------------------
    # Component accessors
    # ------------------------------------------------------------------

    @property
    def if_model(self) -> FlowAnomalyDetector:
        """Stage 1 anomaly detector."""
        return self._if_model

    @property
    def stage2_classifier(self) -> FlowAttackClassifier:
        """Stage 2 attack classifier."""
        return self._xgb

    def get_feature_importance(self) -> dict[str, float]:
        """
        Stage 2 feature importances, keyed by feature name when known.

        Returns:
            ``{feature_name: importance}``, highest first.
        """
        return self._xgb.get_feature_importance(self._feature_names)

    def get_if_operating_point(self) -> dict:
        """
        Stage 1 threshold diagnostics.

        Returns:
            Precision, recall, FPR, FNR and the confusion-matrix counts.

        Raises:
            RuntimeError: No threshold search has run.
        """
        return self._if_model.get_operating_point_info()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _report_fold_protocol(self, train_index: TemporalIndex | None) -> None:
        """
        Say, once per run, what the Stage 2 search is about to cut its folds on.

        Only the cascade knows both facts at once: whether an index arrived, and
        which partitioning protocol the run declared. So the two cases are
        separated here rather than inside the classifier, which sees only
        ``None``:

        * A temporal protocol without an index is a **warning**. The run claims
          a chronological partition and its hyperparameter search silently is
          not one.
        * ``--split random`` without an index is expected and logged at info.
          There is no temporal claim to violate, and a warning would be noise
          that trains people to ignore the other one.
        """
        if train_index is not None:
            return

        if self._config.split_strategy == "random":
            logger.info(
                "xgb_tuning_folds_random_by_protocol",
                pipeline=self.name,
                hint="random split makes no chronological claim; shuffled folds match it",
            )
            return

        logger.warning(
            "xgb_tuning_temporal_folds_unavailable",
            pipeline=self.name,
            split_strategy=self._config.split_strategy,
            hint=(
                "no train_index reached train(); Stage 2 tunes on shuffled folds, which "
                "can bias which hyperparameters win. Pass "
                "train_index=build_temporal_index(train_df, config)"
            ),
        )

    def _check_feature_names(self, X: np.ndarray) -> None:
        """
        Warn or fail early on a feature-name list that cannot describe ``X``.

        Raises:
            ValueError: The list length disagrees with the matrix width. Left
                unchecked this surfaces as silently wrong feature importances
                and an unservable artifact.
        """
        if self._feature_names is None:
            logger.warning(
                "cascade_feature_names_missing",
                pipeline=self.name,
                hint="save() will refuse; pass feature_names=get_feature_columns(df)",
            )
            return

        if len(self._feature_names) != X.shape[1]:
            raise ValueError(
                f"[{self.name}] feature_names has {len(self._feature_names)} entries but "
                f"X has {X.shape[1]} columns."
            )
