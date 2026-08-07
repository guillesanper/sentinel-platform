"""
From-scratch NumPy MLP — the alternative Stage 2 classifier. **Experimental.**

Measured result on CICIDS2017 (1M flows, random split): **F1-macro 0.7508**
against **0.9516** for the XGBoost Stage 2 of ``cascade_xgb``, at an accuracy of
0.9696 against 0.9936. That gap is the reason this module exists only as a
reference implementation and **never enters the serving path**: nothing in
``ml/flow/serving.py`` or the API imports it, and ``ml.flow.models.__init__``
deliberately does not re-export it.

What it is worth keeping for is that everything below is hand-derived: the
forward pass, the full chain-rule backpropagation, Batch Normalisation with its
exact closed-form gradient, inverted Dropout, Adam with bias correction and L2
decay, cross-entropy with label smoothing, focal weighting, class-balanced
batching, and ReduceLROnPlateau. No autodiff framework is involved anywhere.

Architecture::

    Input → Dense(256) → BatchNorm → ReLU → Dropout(0.15)
          → Dense(128) → BatchNorm → ReLU → Dropout(0.15)
          → Dense(64)  → BatchNorm → ReLU → Dropout(0.10)
          → Dense(n_classes) → Softmax

Ported verbatim from the CICIDS2017 TFA: the numerics are untouched. Only the
logging (stdlib ``logging`` → structlog), the type annotations, and the progress
bar (``tqdm`` is not a project dependency) differ from the original. It is a
frozen academic artifact — the lint suppressions in ``pyproject.toml`` exist so
that it is not refactored, and it should not be.
"""

from pathlib import Path
from typing import Any, Final

import joblib
import numpy as np
import structlog
from sklearn.metrics import f1_score as _sk_f1
from sklearn.preprocessing import LabelEncoder, StandardScaler

from ml.flow.config import NeuralNetworkConfig

logger = structlog.get_logger()

# Exponential decay factor for the BatchNorm running statistics.
_BN_MOMENTUM: Final[float] = 0.9
# Numerical stability constant inside the BatchNorm normalisation.
_BN_EPS: Final[float] = 1e-5

# One structured log line every N epochs; the original printed a tqdm bar.
_LOG_EVERY_N_EPOCHS: Final[int] = 10


class NeuralNetwork:
    """
    Multi-class MLP implemented with NumPy alone.

    Attributes:
        config: Architecture and optimisation hyperparameters.
        n_features: Input dimensionality, fixed at construction time.
        n_classes: Output dimensionality, fixed at construction time.
        history: Per-epoch ``train_loss``/``train_acc``/``val_loss``/``val_acc``.
    """

    def __init__(
        self,
        config: NeuralNetworkConfig,
        n_features: int,
        n_classes: int,
        random_seed: int = 42,
    ) -> None:
        """
        Args:
            config: Architecture and optimiser hyperparameters.
            n_features: Number of input features.
            n_classes: Number of output classes.
            random_seed: Seed for weight initialisation, dropout and batching.
        """
        self.config = config
        self.n_features = n_features
        self.n_classes = n_classes
        self.random_seed = random_seed
        self._rng = np.random.default_rng(random_seed)

        self._label_encoder: LabelEncoder = LabelEncoder()
        self._encoder_fitted: bool = False
        self._scaler: StandardScaler = StandardScaler()
        self._class_weights: np.ndarray | None = None

        self._params: dict[str, np.ndarray] = {}
        self._bn_running: dict[str, np.ndarray] = {}
        self._m: dict[str, np.ndarray] = {}  # Adam first moment
        self._v: dict[str, np.ndarray] = {}  # Adam second moment
        self._t: int = 0  # Adam global step

        self._current_lr: float = config.learning_rate
        self._lr_patience_counter: int = 0

        self.history: dict[str, list[float]] = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }
        self._best_weights: dict[str, np.ndarray] | None = None

        self._init_params()
        self._init_adam_state()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_params(self) -> None:
        """He-initialise the weight matrices; BatchNorm parameters start at identity."""
        h1, h2, h3 = self.config.hidden_layers  # [256, 128, 64]
        n = self.n_features

        self._params["W1"] = self._rng.standard_normal((n, h1)) * np.sqrt(2.0 / n)
        self._params["b1"] = np.zeros(h1)
        if self.config.use_batch_norm:
            self._params["gamma1"] = np.ones(h1)
            self._params["beta_bn1"] = np.zeros(h1)
            self._bn_running["mean1"] = np.zeros(h1)
            self._bn_running["var1"] = np.ones(h1)

        self._params["W2"] = self._rng.standard_normal((h1, h2)) * np.sqrt(2.0 / h1)
        self._params["b2"] = np.zeros(h2)
        if self.config.use_batch_norm:
            self._params["gamma2"] = np.ones(h2)
            self._params["beta_bn2"] = np.zeros(h2)
            self._bn_running["mean2"] = np.zeros(h2)
            self._bn_running["var2"] = np.ones(h2)

        self._params["W3"] = self._rng.standard_normal((h2, h3)) * np.sqrt(2.0 / h2)
        self._params["b3"] = np.zeros(h3)
        if self.config.use_batch_norm:
            self._params["gamma3"] = np.ones(h3)
            self._params["beta_bn3"] = np.zeros(h3)
            self._bn_running["mean3"] = np.zeros(h3)
            self._bn_running["var3"] = np.ones(h3)

        self._params["W4"] = self._rng.standard_normal((h3, self.n_classes)) * np.sqrt(2.0 / h3)
        self._params["b4"] = np.zeros(self.n_classes)

    def _init_adam_state(self) -> None:
        """Reset both Adam moment estimates and the global step counter."""
        self._m = {k: np.zeros_like(v) for k, v in self._params.items()}
        self._v = {k: np.zeros_like(v) for k, v in self._params.items()}
        self._t = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> "NeuralNetwork":
        """
        Train with mini-batch Adam and early stopping on validation F1-macro.

        Args:
            X_train: Training feature matrix.
            y_train: Training string class labels.
            X_val: Validation feature matrix.
            y_val: Validation string class labels.

        Returns:
            self
        """
        if self._encoder_fitted:
            known_train_mask = np.isin(y_train, self._label_encoder.classes_)
            if not known_train_mask.all():
                logger.warning(
                    "flow_nn_train_labels_outside_encoder",
                    n_excluded=int((~known_train_mask).sum()),
                    labels=sorted({str(v) for v in y_train[~known_train_mask]}),
                )
                X_train = X_train[known_train_mask]
                y_train = y_train[known_train_mask]
        y_enc = self._fit_and_encode(y_train)
        X_train = self._scaler.fit_transform(X_train)

        if self.config.oversample_min_ratio > 0:
            X_train, y_enc = self._oversample_minority(
                X_train, y_enc, self.config.oversample_min_ratio
            )

        if self.config.use_class_weights:
            counts = np.bincount(y_enc, minlength=self.n_classes).astype(float)
            self._class_weights = len(y_enc) / (self.n_classes * np.maximum(counts, 1.0))
        else:
            self._class_weights = None

        known_mask = np.isin(y_val, self._label_encoder.classes_)
        if not known_mask.all():
            logger.warning(
                "flow_nn_val_classes_unseen",
                n_excluded=int((~known_mask).sum()),
                labels=sorted({str(v) for v in y_val[~known_mask]}),
            )
            X_val = X_val[known_mask]
            y_val = y_val[known_mask]
        y_val_enc = self._label_encoder.transform(y_val).astype(np.int32)
        X_val = self._scaler.transform(X_val)

        n_train = X_train.shape[0]
        bs = self.config.batch_size
        patience = self.config.early_stopping_patience
        min_delta = self.config.early_stopping_min_delta

        best_es_metric = np.inf
        patience_counter = 0
        self._current_lr = self.config.learning_rate
        self._lr_patience_counter = 0

        logger.info(
            "flow_nn_training",
            n_samples=int(n_train),
            n_classes=self.n_classes,
            epochs=self.config.epochs,
            batch_size=bs,
        )

        for epoch in range(1, self.config.epochs + 1):
            if epoch <= self.config.lr_warmup_epochs:
                self._current_lr = self.config.lr_min + (
                    self.config.learning_rate - self.config.lr_min
                ) * epoch / max(self.config.lr_warmup_epochs, 1)

            epoch_loss = 0.0
            epoch_correct = 0
            epoch_samples = 0

            if self.config.use_balanced_batching:
                batch_list = self._balanced_batch_indices(y_enc, bs)
            else:
                idx = self._rng.permutation(n_train)
                batch_list = [idx[s : s + bs] for s in range(0, n_train, bs)]

            for batch_idx in batch_list:
                X_b, y_b = X_train[batch_idx], y_enc[batch_idx]
                probs, cache = self._forward(X_b, training=True)
                y_smooth = self._smooth_labels(y_b)

                p_t = probs[np.arange(len(y_b)), y_b]
                focal_w = (1.0 - p_t) ** self.config.focal_gamma
                if self._class_weights is not None:
                    sample_w = focal_w * self._class_weights[y_b]
                else:
                    sample_w = focal_w

                loss = self._cross_entropy_loss(probs, y_smooth, sample_w)

                epoch_loss += loss * len(X_b)
                epoch_correct += int((probs.argmax(axis=1) == y_b).sum())
                epoch_samples += len(X_b)

                grads = self._backward(cache, y_smooth, sample_w)
                self._adam_update(grads)

            epoch_loss /= epoch_samples
            epoch_acc = epoch_correct / epoch_samples

            val_probs, _ = self._forward(X_val, training=False)
            val_smooth = self._smooth_labels(y_val_enc)
            val_sample_w = (
                self._class_weights[y_val_enc] if self._class_weights is not None else None
            )
            val_loss = self._cross_entropy_loss(val_probs, val_smooth, val_sample_w)
            val_acc = float((val_probs.argmax(axis=1) == y_val_enc).mean())

            self.history["train_loss"].append(epoch_loss)
            self.history["train_acc"].append(epoch_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            val_preds = val_probs.argmax(axis=1)
            val_f1_mac = float(_sk_f1(y_val_enc, val_preds, average="macro", zero_division=0))
            # Negated so that "lower is better" holds for the early-stopping
            # comparison below, as it would for a loss.
            es_metric = -val_f1_mac

            if epoch % _LOG_EVERY_N_EPOCHS == 0 or epoch == 1:
                logger.info(
                    "flow_nn_epoch",
                    epoch=epoch,
                    loss=round(epoch_loss, 6),
                    acc=round(epoch_acc, 6),
                    val_loss=round(val_loss, 6),
                    val_f1_macro=round(val_f1_mac, 6),
                    lr=self._current_lr,
                )

            if es_metric < best_es_metric - min_delta:
                best_es_metric = es_metric
                patience_counter = 0
                self._lr_patience_counter = 0
                self._best_weights = {k: v.copy() for k, v in self._params.items()}
            else:
                patience_counter += 1
                self._lr_patience_counter += 1
                if self._lr_patience_counter >= self.config.lr_scheduler_patience:
                    new_lr = max(
                        self._current_lr * self.config.lr_scheduler_factor,
                        self.config.lr_min,
                    )
                    if new_lr < self._current_lr:
                        self._current_lr = new_lr
                        logger.info("flow_nn_lr_reduced", lr=new_lr, epoch=epoch)
                    self._lr_patience_counter = 0
                if patience_counter >= patience:
                    logger.info(
                        "flow_nn_early_stopped",
                        epoch=epoch,
                        best_val_f1_macro=round(-float(best_es_metric), 6),
                    )
                    break

        if self._best_weights is not None:
            self._params = self._best_weights
            self._best_weights = None
            logger.info(
                "flow_nn_best_weights_restored",
                val_f1_macro=round(-float(best_es_metric), 6),
            )

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict string class labels.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.

        Returns:
            String labels of shape ``(n_samples,)``.

        Raises:
            RuntimeError: The network has not been fitted.
        """
        self._check_fitted()
        X = self._scaler.transform(X)
        probs, _ = self._forward(X, training=False)
        top_idx = probs.argmax(axis=1)
        threshold = self.config.confidence_threshold
        if threshold > 0:
            top_probs = probs[np.arange(len(probs)), top_idx]
            low_conf = top_probs < threshold
            if low_conf.any():
                probs_fb = probs.copy()
                probs_fb[np.arange(len(probs))[low_conf], top_idx[low_conf]] = 0.0
                top_idx[low_conf] = probs_fb[low_conf].argmax(axis=1)
        labels: np.ndarray = self._label_encoder.inverse_transform(top_idx)
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
            RuntimeError: The network has not been fitted.
        """
        self._check_fitted()
        X = self._scaler.transform(X)
        probs, _ = self._forward(X, training=False)
        return probs

    @property
    def classes_(self) -> np.ndarray:
        """
        String class labels in label-encoder order.

        Raises:
            RuntimeError: The encoder has not been fitted.
        """
        if not self._encoder_fitted:
            raise RuntimeError("Network has not been fitted yet.")
        classes: np.ndarray = self._label_encoder.classes_
        return classes

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, directory: Path | str) -> None:
        """
        Persist weights, BatchNorm running statistics, encoder and scaler.

        Args:
            directory: Target directory, created recursively if absent.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.savez(directory / "nn_weights.npz", **self._params)
        np.savez(directory / "nn_bn_running.npz", **self._bn_running)
        joblib.dump(self._label_encoder, directory / "nn_label_encoder.joblib")
        joblib.dump(self._scaler, directory / "nn_scaler.joblib")
        logger.info("flow_nn_saved", path=str(directory))

    def load(self, directory: Path | str) -> "NeuralNetwork":
        """
        Restore weights, running statistics, encoder and scaler.

        Args:
            directory: Directory previously written by :meth:`save`.

        Returns:
            self
        """
        directory = Path(directory)
        weights = np.load(directory / "nn_weights.npz")
        self._params = {k: weights[k] for k in weights.files}
        bn_data = np.load(directory / "nn_bn_running.npz")
        self._bn_running = {k: bn_data[k] for k in bn_data.files}
        self._label_encoder = joblib.load(directory / "nn_label_encoder.joblib")
        self._encoder_fitted = True
        self.n_classes = len(self._label_encoder.classes_)
        scaler_path = directory / "nn_scaler.joblib"
        if scaler_path.exists():
            self._scaler = joblib.load(scaler_path)
        self._init_adam_state()
        logger.info("flow_nn_loaded", path=str(directory))
        return self

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def _forward(self, X: np.ndarray, training: bool) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Run the full forward pass, recording everything backprop will need.

        Returns:
            ``(probs, cache)`` — class probabilities after Softmax, and the
            intermediate activations required by :meth:`_backward`.
        """
        cache: dict[str, Any] = {"X": X}

        # Layer 1 — Dense(256)
        z1 = X @ self._params["W1"] + self._params["b1"]
        cache["z1"] = z1
        if self.config.use_batch_norm:
            bn1_out, bn1_cache = self._bn_forward(z1, "1", training)
        else:
            bn1_out, bn1_cache = z1, None
        cache["bn1_out"] = bn1_out
        cache["bn1_cache"] = bn1_cache
        a1 = np.maximum(0.0, bn1_out)
        cache["a1"] = a1
        d1, mask1 = self._dropout_forward(a1, self.config.dropout_rates[0], training)
        cache["d1"] = d1
        cache["mask1"] = mask1

        # Layer 2 — Dense(128)
        z2 = d1 @ self._params["W2"] + self._params["b2"]
        cache["z2"] = z2
        if self.config.use_batch_norm:
            bn2_out, bn2_cache = self._bn_forward(z2, "2", training)
        else:
            bn2_out, bn2_cache = z2, None
        cache["bn2_out"] = bn2_out
        cache["bn2_cache"] = bn2_cache
        a2 = np.maximum(0.0, bn2_out)
        cache["a2"] = a2
        d2, mask2 = self._dropout_forward(a2, self.config.dropout_rates[1], training)
        cache["d2"] = d2
        cache["mask2"] = mask2

        # Layer 3 — Dense(64)
        z3 = d2 @ self._params["W3"] + self._params["b3"]
        cache["z3"] = z3
        if self.config.use_batch_norm:
            bn3_out, bn3_cache = self._bn_forward(z3, "3", training)
        else:
            bn3_out, bn3_cache = z3, None
        cache["bn3_out"] = bn3_out
        cache["bn3_cache"] = bn3_cache
        a3 = np.maximum(0.0, bn3_out)
        cache["a3"] = a3
        d3, mask3 = self._dropout_forward(a3, self.config.dropout_rates[2], training)
        cache["d3"] = d3
        cache["mask3"] = mask3

        # Layer 4 — Dense(n_classes) + Softmax
        z4 = d3 @ self._params["W4"] + self._params["b4"]
        probs = self._softmax(z4)
        cache["z4"] = z4
        cache["probs"] = probs

        return probs, cache

    # ------------------------------------------------------------------
    # Backward pass
    # ------------------------------------------------------------------

    def _backward(
        self,
        cache: dict[str, Any],
        y_smooth: np.ndarray,
        sample_weights: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """
        Full backpropagation via the chain rule.

        The combined gradient of Softmax + cross-entropy with respect to the
        pre-Softmax logits is ``(probs - y_smooth) / N``, which sidesteps the
        numerically awkward explicit Softmax Jacobian.

        Args:
            cache: Output of :meth:`_forward` with ``training=True``.
            y_smooth: Label-smoothed one-hot targets, shape ``(N, n_classes)``.
            sample_weights: Optional per-sample weights, rescaled to mean 1.

        Returns:
            Parameter name → gradient.
        """
        N = cache["X"].shape[0]
        grads: dict[str, np.ndarray] = {}

        # Combined Softmax + cross-entropy gradient
        d_z4 = (cache["probs"] - y_smooth) / N
        if sample_weights is not None:
            w_norm = sample_weights / (sample_weights.mean() + 1e-12)
            d_z4 = d_z4 * w_norm[:, None]

        # Layer 4 (output)
        grads["W4"] = cache["d3"].T @ d_z4
        grads["b4"] = d_z4.sum(axis=0)
        d_d3 = d_z4 @ self._params["W4"].T

        # Dropout 3
        d_a3 = self._dropout_backward(d_d3, cache["mask3"])

        # ReLU 3 — gate: bn3_out > 0
        d_bn3 = d_a3 * (cache["bn3_out"] > 0)

        # BatchNorm 3
        if self.config.use_batch_norm and cache["bn3_cache"] is not None:
            d_z3, grads["gamma3"], grads["beta_bn3"] = self._bn_backward(d_bn3, cache["bn3_cache"])
        else:
            d_z3 = d_bn3

        # Layer 3
        grads["W3"] = cache["d2"].T @ d_z3
        grads["b3"] = d_z3.sum(axis=0)
        d_d2 = d_z3 @ self._params["W3"].T

        # Dropout 2
        d_a2 = self._dropout_backward(d_d2, cache["mask2"])

        # ReLU 2 — gate: bn2_out > 0
        d_bn2 = d_a2 * (cache["bn2_out"] > 0)

        # BatchNorm 2
        if self.config.use_batch_norm and cache["bn2_cache"] is not None:
            d_z2, grads["gamma2"], grads["beta_bn2"] = self._bn_backward(d_bn2, cache["bn2_cache"])
        else:
            d_z2 = d_bn2

        # Layer 2
        grads["W2"] = cache["d1"].T @ d_z2
        grads["b2"] = d_z2.sum(axis=0)
        d_d1 = d_z2 @ self._params["W2"].T

        # Dropout 1
        d_a1 = self._dropout_backward(d_d1, cache["mask1"])

        # ReLU 1 — gate: bn1_out > 0 (ReLU is applied after BatchNorm)
        d_bn1 = d_a1 * (cache["bn1_out"] > 0)

        # BatchNorm 1
        if self.config.use_batch_norm and cache["bn1_cache"] is not None:
            d_z1, grads["gamma1"], grads["beta_bn1"] = self._bn_backward(d_bn1, cache["bn1_cache"])
        else:
            d_z1 = d_bn1

        # Layer 1
        grads["W1"] = cache["X"].T @ d_z1
        grads["b1"] = d_z1.sum(axis=0)

        # L2 weight decay on the weight matrices only — never on biases or the
        # BatchNorm scale/shift, which have no reason to be pulled to zero.
        wd = self.config.weight_decay
        for key in ("W1", "W2", "W3", "W4"):
            grads[key] = grads[key] + wd * self._params[key]

        return grads

    # ------------------------------------------------------------------
    # Adam optimiser
    # ------------------------------------------------------------------

    def _adam_update(self, grads: dict[str, np.ndarray]) -> None:
        """
        Update every parameter with Adam and bias correction (Kingma & Ba, 2015).

        ``m_t = b1*m_{t-1} + (1-b1)*g_t``,
        ``v_t = b2*v_{t-1} + (1-b2)*g_t^2``,
        ``theta_t = theta_{t-1} - lr * m_hat / (sqrt(v_hat) + eps)``.
        """
        if self.config.grad_clip_norm > 0:
            total_norm = np.sqrt(sum(np.sum(g**2) for g in grads.values()))
            if total_norm > self.config.grad_clip_norm:
                scale = self.config.grad_clip_norm / (total_norm + 1e-8)
                grads = {k: v * scale for k, v in grads.items()}
        self._t += 1
        lr = self._current_lr
        b1 = self.config.beta1
        b2 = self.config.beta2
        eps = self.config.epsilon

        for key, g in grads.items():
            self._m[key] = b1 * self._m[key] + (1.0 - b1) * g
            self._v[key] = b2 * self._v[key] + (1.0 - b2) * (g * g)

            m_hat = self._m[key] / (1.0 - b1**self._t)
            v_hat = self._v[key] / (1.0 - b2**self._t)

            self._params[key] -= lr * m_hat / (np.sqrt(v_hat) + eps)

    # ------------------------------------------------------------------
    # BatchNorm helpers
    # ------------------------------------------------------------------

    def _bn_forward(
        self, z: np.ndarray, layer_id: str, training: bool
    ) -> tuple[np.ndarray, dict[str, Any] | None]:
        """
        BatchNorm forward for one layer.

        In training mode it normalises over the mini-batch and updates the
        exponential moving averages; at inference it uses the stored statistics.
        """
        gamma = self._params[f"gamma{layer_id}"]
        beta = self._params[f"beta_bn{layer_id}"]

        if training:
            mu = z.mean(axis=0)
            var = z.var(axis=0)
            z_hat = (z - mu) / np.sqrt(var + _BN_EPS)
            out = gamma * z_hat + beta

            self._bn_running[f"mean{layer_id}"] = (
                _BN_MOMENTUM * self._bn_running[f"mean{layer_id}"] + (1.0 - _BN_MOMENTUM) * mu
            )
            self._bn_running[f"var{layer_id}"] = (
                _BN_MOMENTUM * self._bn_running[f"var{layer_id}"] + (1.0 - _BN_MOMENTUM) * var
            )
            return out, {"z_hat": z_hat, "var": var, "gamma": gamma, "N": z.shape[0]}

        z_hat = (z - self._bn_running[f"mean{layer_id}"]) / np.sqrt(
            self._bn_running[f"var{layer_id}"] + _BN_EPS
        )
        return gamma * z_hat + beta, None

    def _bn_backward(
        self, d_out: np.ndarray, cache: dict[str, Any]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        BatchNorm backward via the exact chain-rule expression.

        The derivation accounts for mu and sigma^2 depending on every input in
        the mini-batch, giving the closed form::

            dL/dz = (1/(N*sigma)) * (N*dL/dz_hat - sum(dL/dz_hat)
                                     - z_hat*sum(dL/dz_hat * z_hat))

        Returns:
            ``(d_z, d_gamma, d_beta)``.
        """
        z_hat = cache["z_hat"]
        var = cache["var"]
        gamma = cache["gamma"]
        N = cache["N"]
        inv_std = 1.0 / np.sqrt(var + _BN_EPS)

        d_gamma = (d_out * z_hat).sum(axis=0)
        d_beta = d_out.sum(axis=0)

        d_z_hat = d_out * gamma
        d_z = (inv_std / N) * (
            N * d_z_hat - d_z_hat.sum(axis=0) - z_hat * (d_z_hat * z_hat).sum(axis=0)
        )
        return d_z, d_gamma, d_beta

    # ------------------------------------------------------------------
    # Dropout helpers
    # ------------------------------------------------------------------

    def _dropout_forward(
        self, a: np.ndarray, rate: float, training: bool
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """
        Inverted dropout: surviving activations are scaled by ``1/(1-rate)``, so
        inference needs no adjustment at all.
        """
        if not training or rate == 0.0:
            return a, None
        keep_prob = 1.0 - rate
        mask = (self._rng.random(a.shape) < keep_prob).astype(a.dtype) / keep_prob
        return a * mask, mask

    def _dropout_backward(self, d_out: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
        """Propagate the gradient through inverted dropout; dropped units get zero."""
        if mask is None:
            return d_out
        gated: np.ndarray = d_out * mask
        return gated

    # ------------------------------------------------------------------
    # Static utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _softmax(z: np.ndarray) -> np.ndarray:
        """Row-wise softmax, shifted by the row max for numerical stability."""
        z_shift = z - z.max(axis=1, keepdims=True)
        exp_z = np.exp(z_shift)
        result: np.ndarray = exp_z / exp_z.sum(axis=1, keepdims=True)
        return result

    def _smooth_labels(self, y: np.ndarray) -> np.ndarray:
        """
        Turn integer labels into label-smoothed one-hot vectors.

        ``y_smooth[k] = (1 - eps) * delta_{k,c} + eps / K``, with ``eps`` the
        smoothing factor, ``K`` the class count and ``c`` the true class.
        """
        eps = self.config.label_smoothing
        k = self.n_classes
        y_oh = np.zeros((len(y), k), dtype=np.float64)
        y_oh[np.arange(len(y)), y] = 1.0
        return (1.0 - eps) * y_oh + eps / k

    @staticmethod
    def _cross_entropy_loss(
        probs: np.ndarray,
        y_smooth: np.ndarray,
        sample_weights: np.ndarray | None = None,
    ) -> float:
        """Weighted mean cross-entropy, supporting per-sample class weights."""
        per_sample = -np.sum(y_smooth * np.log(np.clip(probs, 1e-12, 1.0)), axis=1)
        if sample_weights is not None:
            return float(np.sum(per_sample * sample_weights) / sample_weights.sum())
        return float(np.mean(per_sample))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _balanced_batch_indices(self, y: np.ndarray, batch_size: int) -> list[np.ndarray]:
        """Build mini-batches holding roughly equal counts of every class."""
        classes = np.unique(y)
        per_class = max(1, batch_size // len(classes))
        idx_by_class = {c: np.where(y == c)[0] for c in classes}
        # Capped at the natural number of batches, so an enormous majority class
        # does not make one epoch traverse the dataset several times over.
        n_batches = min(
            max(len(v) for v in idx_by_class.values()) // per_class,
            max(1, len(y) // batch_size),
        )
        n_batches = max(1, n_batches)
        batches = []
        for _ in range(n_batches):
            batch_parts = []
            for c in classes:
                pool = idx_by_class[c]
                chosen = (
                    self._rng.choice(pool, per_class, replace=True)
                    if len(pool) < per_class
                    else self._rng.choice(pool, per_class, replace=False)
                )
                batch_parts.append(chosen)
            batch = np.concatenate(batch_parts)
            self._rng.shuffle(batch)
            batches.append(batch)
        return batches

    def _oversample_minority(
        self, X: np.ndarray, y: np.ndarray, min_ratio: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Repeat rare-class rows with Gaussian noise (sigma=0.01, scaled space)."""
        counts = np.bincount(y, minlength=self.n_classes).astype(float)
        target = max(int(min_ratio * counts.max()), int(counts.min()))
        X_parts, y_parts = [X], [y]
        for c in range(self.n_classes):
            deficit = target - int(counts[c])
            # A class with no rows at all cannot be oversampled from. The TFA
            # never hit this because all seven classes were always present;
            # here a day-holdout split, minority filtering or a small sample
            # can easily leave one of CLASS_ORDER empty.
            if deficit <= 0 or counts[c] == 0:
                continue
            idx = self._rng.choice(np.where(y == c)[0], deficit, replace=True)
            noise = self._rng.standard_normal((deficit, X.shape[1])) * 0.01
            X_parts.append(X[idx] + noise)
            y_parts.append(np.full(deficit, c, dtype=np.int32))
        if len(X_parts) == 1:
            return X, y
        return np.vstack(X_parts), np.concatenate(y_parts)

    def _fit_and_encode(self, y: np.ndarray) -> np.ndarray:
        """Fit the label encoder on first use and return integer-encoded labels."""
        if not self._encoder_fitted:
            self._label_encoder.fit(y)
            self._encoder_fitted = True
        encoded: np.ndarray = self._label_encoder.transform(y).astype(np.int32)
        return encoded

    def _check_fitted(self) -> None:
        """Raise when the network cannot predict yet."""
        if not self._encoder_fitted:
            raise RuntimeError("Network must be fit before calling this method.")
