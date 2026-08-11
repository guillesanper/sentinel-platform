"""
Isolation Forest → NumPy MLP cascade. **Experimental, never served.**

Same Stage 1 as :mod:`ml.flow.pipelines.cascade_xgb` — literally the same
detector with the same configuration — and a from-scratch NumPy MLP for Stage 2
instead of XGBoost. Measured on CICIDS2017 (1M flows, random split): accuracy
0.9696, F1-macro **0.7508**, against 0.9936 / 0.9516 for the XGBoost cascade.

That gap is why this module exists as a reference implementation only: it is not
re-exported from ``ml.flow.pipelines``, nothing in the serving path imports it,
and it is excluded from coverage. Because Stage 1 is shared, its threshold
diagnostics are identical to the XGBoost cascade's — do not report them as two
independent results.

Two consequences of the MLP's API shape are worth knowing:

* ``NeuralNetwork`` needs ``n_features`` and ``n_classes`` at construction, and
  neither is known until Stage 1 has routed. The network is therefore built
  inside :meth:`train`, not in ``__init__``.
* ``skip_tuning`` is accepted for interface compatibility and does nothing:
  there is no hyperparameter search for this network. Its search space is the
  fixed architecture plus the defaults in ``NeuralNetworkConfig``.
"""

from pathlib import Path

import numpy as np
import structlog
from sklearn.metrics import accuracy_score, f1_score

from ml.flow.config import FlowConfig
from ml.flow.labels import BENIGN_LABEL, CLASS_ORDER
from ml.flow.models.isolation_forest import FlowAnomalyDetector
from ml.flow.models.neural_network import NeuralNetwork
from ml.flow.pipelines.base import BaseCascadePipeline

logger = structlog.get_logger()

# Sub-directories written by save() and expected by load().
IF_SUBDIR = "isolation_forest"
NN_SUBDIR = "neural_network"

# Stand-in eval set when Stage 1 flags no validation flow at all.
_PSEUDO_VAL_FRACTION = 0.10


class CascadeMLPPipeline(BaseCascadePipeline):
    """
    Two-stage cascade: :class:`FlowAnomalyDetector` then :class:`NeuralNetwork`.

    Kept for its academic value. Use :class:`CascadeXGBPipeline` for anything
    that has to serve traffic.
    """

    def __init__(
        self,
        config: FlowConfig | None = None,
        feature_names: list[str] | None = None,
    ) -> None:
        """
        Args:
            config: Track configuration. Defaults to ``FlowConfig()``.
            feature_names: Column order of the training matrix. Required, here
                or via :meth:`train`, before :meth:`save`.
        """
        self._config = config or FlowConfig()
        self._feature_names = list(feature_names) if feature_names is not None else None
        self._if_model = FlowAnomalyDetector(
            config=self._config.isolation_forest,
            random_seed=self._config.random_seed,
        )
        # Built in train(), once Stage 1 has fixed the input dimensionality.
        self._nn: NeuralNetwork | None = None

    @property
    def name(self) -> str:
        """Identifier used in log events and output paths."""
        return "cascade_mlp"

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
    ) -> "CascadeMLPPipeline":
        """
        Train Stage 1 and Stage 2 in sequence.

        Args:
            X_train: Full training feature matrix, all classes.
            y_train: Canonical string labels for the training set.
            X_val: Validation feature matrix.
            y_val: Canonical string labels for the validation set.
            beta: F-beta weight for the Stage 1 threshold search.
            skip_tuning: Accepted for interface compatibility; no effect.
            feature_names: Column order of ``X_train``.

        Returns:
            self

        Raises:
            RuntimeError: Stage 1 routed no training flow to Stage 2.
        """
        if skip_tuning:
            logger.info("cascade_skip_tuning_ignored", pipeline=self.name)
        if feature_names is not None:
            self._feature_names = list(feature_names)

        logger.info("cascade_stage1_training", pipeline=self.name, n_train=int(len(y_train)))
        self._if_model.fit(X_train, y_train)
        self._if_model.optimize_threshold(X_val, y_val, beta=beta)

        anomaly_train = self._if_model.predict_binary(X_train).astype(bool)
        X_attack_train = X_train[anomaly_train]
        y_attack_train = y_train[anomaly_train]

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

        if X_attack_val.shape[0] == 0:
            logger.warning(
                "cascade_stage2_pseudo_validation",
                pipeline=self.name,
                fraction=_PSEUDO_VAL_FRACTION,
            )
            n_pseudo = max(1, int(_PSEUDO_VAL_FRACTION * X_attack_train.shape[0]))
            X_attack_val = X_attack_train[:n_pseudo]
            y_attack_val = y_attack_train[:n_pseudo]

        n_features = X_attack_train.shape[1]
        # BENIGN is included on purpose: Stage 2 gets a chance to overrule a
        # Stage 1 false positive instead of being forced to name an attack.
        attack_classes = list(CLASS_ORDER)
        n_classes = len(attack_classes)

        logger.info(
            "cascade_stage2_training",
            pipeline=self.name,
            stage2="neural_network",
            n_features=n_features,
            n_classes=n_classes,
        )

        self._nn = NeuralNetwork(
            config=self._config.neural_network,
            n_features=n_features,
            n_classes=n_classes,
            random_seed=self._config.random_seed,
        )
        # Pre-fitting the encoder on the full taxonomy keeps the output layer
        # aligned with CLASS_ORDER even when a class is absent from the routed
        # anomalies of this particular run.
        self._nn._label_encoder.fit(attack_classes)
        self._nn._encoder_fitted = True
        self._nn.fit(X_attack_train, y_attack_train, X_attack_val, y_attack_val)

        logger.info("cascade_trained", pipeline=self.name)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Run end-to-end cascade inference.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.

        Returns:
            Canonical string labels of shape ``(n_samples,)``.

        Raises:
            RuntimeError: The cascade has not been trained or loaded.
        """
        if self._nn is None:
            raise RuntimeError(f"[{self.name}] must be trained or loaded before predict().")

        predictions = np.full(X.shape[0], BENIGN_LABEL, dtype=object)
        anomaly_mask = self._if_model.predict_binary(X).astype(bool)
        if anomaly_mask.sum() > 0:
            predictions[anomaly_mask] = self._nn.predict(X[anomaly_mask])
        labels: np.ndarray = predictions.astype(str)
        return labels

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict[str, float]:
        """
        Score the cascade end to end.

        Args:
            X_test: Test feature matrix.
            y_test: Ground-truth canonical string labels.

        Returns:
            ``accuracy``, ``f1_macro``, ``f1_weighted`` and the Stage 1
            operating point when available.

        Note:
            Stage 1 is measured on ``X_test``, not on the validation set its
            threshold was selected from — see
            :meth:`~ml.flow.pipelines.cascade_xgb.CascadeXGBPipeline.evaluate`.
        """
        y_pred = self.predict(X_test)

        metrics: dict[str, float] = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
            "f1_weighted": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        }

        try:
            selection = self._if_model.get_operating_point_info()
        except RuntimeError:
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
            <directory>/neural_network/       weights, BN statistics, encoder,
                                              scaler

        Args:
            directory: Target directory, created recursively if absent.

        Raises:
            RuntimeError: The cascade has not been trained, or no feature names
                are known.
        """
        if self._nn is None:
            raise RuntimeError(f"[{self.name}] must be trained before save().")

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        self._save_feature_names(directory)
        self._if_model.save(directory / IF_SUBDIR)
        self._nn.save(directory / NN_SUBDIR)

        logger.info("cascade_saved", pipeline=self.name, path=str(directory))

    def load(self, directory: Path | str) -> "CascadeMLPPipeline":
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

        names = self._load_feature_names(directory)
        self._if_model.load(directory / IF_SUBDIR)

        # Placeholder dimensions: load() overwrites every weight matrix, and
        # the true shapes come back with them.
        self._nn = NeuralNetwork(
            config=self._config.neural_network,
            n_features=len(names),
            n_classes=len(CLASS_ORDER),
            random_seed=self._config.random_seed,
        )
        self._nn.load(directory / NN_SUBDIR)
        self._nn.n_features = self._nn._params["W1"].shape[0]

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
    def stage2_classifier(self) -> NeuralNetwork:
        """
        Stage 2 neural network.

        Raises:
            RuntimeError: The cascade has not been trained or loaded.
        """
        if self._nn is None:
            raise RuntimeError(f"[{self.name}] has not been trained yet.")
        return self._nn

    def get_training_history(self) -> dict[str, list[float]]:
        """
        Per-epoch training history of the network.

        Returns:
            ``train_loss``, ``train_acc``, ``val_loss`` and ``val_acc``, each a
            list of length ``n_epochs_run``.

        Raises:
            RuntimeError: The cascade has not been trained.
        """
        if self._nn is None:
            raise RuntimeError(f"[{self.name}] must be trained before accessing history.")
        return self._nn.history

    def get_if_operating_point(self) -> dict:
        """
        Stage 1 threshold diagnostics.

        Returns:
            Precision, recall, FPR, FNR and the confusion-matrix counts.

        Raises:
            RuntimeError: No threshold search has run.
        """
        return self._if_model.get_operating_point_info()
