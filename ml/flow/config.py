"""
Configuration dataclasses for the flow-based track.

Ported from the CICIDS2017 TFA ``config.py`` with three deliberate changes:

* The module-level singleton ``cfg`` is gone. Every function that needs a
  configuration takes ``config: FlowConfig | None = None`` and does
  ``config = config or FlowConfig()``. A mutable global shared by loader,
  models and pipelines makes runs order-dependent and untestable.
* ``Config`` → ``FlowConfig``. A bare ``Config`` is far too generic in a repo
  that already carries ``pydantic-settings`` application settings.
* New ``split_strategy`` field, backing the temporal partitioning that replaces
  the TFA's row-wise random split.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

# Dataset partitioning protocols.
#   temporal    — 70/15/15 chronological cut inside each temporal block.
#                 Default: no future→past leakage, block coverage kept.
#   random      — the TFA's stratified row-wise split. Retained only to
#                 quantify the leakage by comparing against `temporal`.
#   day-holdout — train on Mon-Thu, test on Fri. Honest generalisation to an
#                 unseen day, but only for classes present on both sides.
SPLIT_STRATEGIES: Final[tuple[str, ...]] = ("temporal", "random", "day-holdout")

# What closes the (capture_day, …) tuple that defines a temporal block. Applies
# to `temporal` and to the train/val cut of `day-holdout`; `random` ignores it.
#
#   sub-label — block by the raw CICIDS2017 label. Every attack tool is cut
#               chronologically on its own, so all of them appear in train, val
#               and test. Removes the near-duplicate leakage and nothing else.
#   class     — block by the collapsed canonical label. Because ATTACK_GROUPS
#               merges tools that ran *sequentially* within a day (Wednesday:
#               Hulk → slowloris → Slowhttptest → GoldenEye, all `DoS`), this
#               additionally holds entire tools out of training. That is a
#               legitimate question — generalisation to an unseen tool — but it
#               is a different one, and only the multi-tool classes are actually
#               asked it, so its F1-macro averages incomparable quantities.
#
# Measured on CICIDS2017 (200k flows, xgb, no tuning): F1-macro 0.7721 with
# `sub-label` against 0.2973 with `class`, both under the same chronological
# guarantee. See docs/ml/PARTICION-TEMPORAL.md.
TEMPORAL_BLOCKS: Final[tuple[str, ...]] = ("sub-label", "class")


# ============================================================================
# Stage 1 — Isolation Forest
# ============================================================================


@dataclass
class IsolationForestConfig:
    """
    Default hyperparameters for the Stage 1 anomaly detector.

    Note on ``contamination``: the forest is fitted on **BENIGN rows only**, so
    this is not "the expected attack rate in CICIDS2017" — it is the fraction of
    *benign* traffic sklearn is told to treat as outlying. It only sets
    ``offset_``, which shifts ``decision_function`` by a constant, and
    :meth:`~ml.flow.models.isolation_forest.FlowAnomalyDetector.optimize_threshold`
    then picks its own threshold on that shifted scale. The value is therefore
    inert for every metric this track reports; changing it moves the raw scores
    and the selected threshold by the same amount.
    """

    n_estimators: int = 200
    max_samples: str = "auto"  # "auto" → min(256, n_samples)
    contamination: float = 0.15  # see the class docstring: absorbed by the threshold search
    max_features: float = 1.0
    bootstrap: bool = False
    n_jobs: int = -1


# ============================================================================
# Stage 2 — XGBoost
# ============================================================================


@dataclass
class XGBoostConfig:
    """
    Default hyperparameters and RandomizedSearchCV space for the Stage 2
    gradient-boosted classifier.
    """

    # Defaults, used when hyperparameter tuning is skipped.
    n_estimators: int = 300
    max_depth: int = 6
    learning_rate: float = 0.1
    subsample: float = 0.9
    colsample_bytree: float = 0.9
    min_child_weight: int = 1
    gamma: float = 0.0
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0
    eval_metric: str = "mlogloss"
    early_stopping_rounds: int = 20
    tree_method: str = "hist"  # works on both CPU and GPU
    n_jobs: int = -1

    # Search space for RandomizedSearchCV.
    search_space: dict[str, Any] = field(
        default_factory=lambda: {
            "n_estimators": [200, 300, 500],
            "max_depth": [4, 6, 8],
            "learning_rate": [0.05, 0.1, 0.2],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
        }
    )
    n_iter_search: int = 20
    cv_folds: int = 3


# ============================================================================
# Stage 2 (alternative) — NumPy neural network
# ============================================================================


@dataclass
class NeuralNetworkConfig:
    """
    Hyperparameters for the from-scratch NumPy MLP.

    Kept for its academic value only: it scores F1-macro 0.7508 against 0.9516
    for XGBoost, so it never enters the serving path.
    """

    # Architecture
    hidden_layers: list[int] = field(default_factory=lambda: [256, 128, 64])
    dropout_rates: list[float] = field(default_factory=lambda: [0.15, 0.15, 0.10])
    use_batch_norm: bool = True

    # Optimisation
    learning_rate: float = 1e-3
    beta1: float = 0.9  # Adam — first-order moment
    beta2: float = 0.999  # Adam — second-order moment
    epsilon: float = 1e-8  # Adam — numerical stability term
    weight_decay: float = 1e-4  # L2 regularisation applied inside Adam

    # Training
    epochs: int = 100
    batch_size: int = 512
    label_smoothing: float = 0.05
    early_stopping_patience: int = 20
    early_stopping_min_delta: float = 1e-4
    use_class_weights: bool = True

    # LR scheduling (ReduceLROnPlateau)
    lr_scheduler_patience: int = 5
    lr_scheduler_factor: float = 0.5
    lr_min: float = 1e-5

    # v2 improvements
    use_balanced_batching: bool = True
    focal_gamma: float = 2.0
    oversample_min_ratio: float = 0.15
    lr_warmup_epochs: int = 5
    grad_clip_norm: float = 1.0
    confidence_threshold: float = 0.0  # below it, fall back to the runner-up class


# ============================================================================
# Track configuration
# ============================================================================


@dataclass
class FlowConfig:
    """
    Configuration for the flow-based training track.

    Instantiate explicitly and pass it down; there is no global instance.

    Example:
        >>> config = FlowConfig(sample_size=20_000, split_strategy="random")
        >>> config.train_ratio
        0.7
    """

    # --- Paths ---
    data_path: Path = field(default_factory=lambda: Path("data"))
    output_path: Path = field(default_factory=lambda: Path("models/flow"))

    # --- Reproducibility ---
    random_seed: int = 42

    # --- Sampling ---
    sample_size: int = 1_000_000  # 0 → use the full dataset

    # --- Dataset partitioning ---
    split_strategy: str = "temporal"
    temporal_block: str = "sub-label"
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # --- Minority class filtering ---
    min_class_samples: int = 100  # absolute minimum samples per class
    min_class_ratio: float = 0.0005  # relative minimum (fraction of the total)

    # --- F-beta threshold for the Stage 1 detector ---
    beta: float = 2.0  # >1 penalises false negatives → higher Stage 1 recall

    # --- Model sub-configurations ---
    isolation_forest: IsolationForestConfig = field(default_factory=IsolationForestConfig)
    xgboost: XGBoostConfig = field(default_factory=XGBoostConfig)
    neural_network: NeuralNetworkConfig = field(default_factory=NeuralNetworkConfig)

    def __post_init__(self) -> None:
        self.data_path = Path(self.data_path)
        self.output_path = Path(self.output_path)
        _validate_ratios(self.train_ratio, self.val_ratio, self.test_ratio)
        _validate_split_strategy(self.split_strategy)
        _validate_temporal_block(self.temporal_block)

    @property
    def eda_path(self) -> Path:
        """Directory for exploratory data analysis figures."""
        return self.output_path / "eda"

    @property
    def cascade_xgb_path(self) -> Path:
        """Directory for the Isolation Forest → XGBoost cascade artifacts."""
        return self.output_path / "cascade_xgb"

    @property
    def cascade_mlp_path(self) -> Path:
        """Directory for the Isolation Forest → NumPy MLP cascade artifacts."""
        return self.output_path / "cascade_mlp"

    @property
    def comparison_path(self) -> Path:
        """Directory for cross-cascade comparison outputs."""
        return self.output_path / "comparison"

    @property
    def results_file(self) -> Path:
        """Plain-text run summary."""
        return self.output_path / "results.txt"

    def create_output_dirs(self) -> None:
        """Create the output directory tree if it does not exist."""
        for path in (
            self.eda_path,
            self.cascade_xgb_path,
            self.cascade_mlp_path,
            self.comparison_path,
        ):
            path.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Validation helpers
# ============================================================================


def _validate_ratios(train: float, val: float, test: float) -> None:
    """Raise when the three split ratios do not add up to 1.0."""
    total = round(train + val + test, 10)
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"train_ratio + val_ratio + test_ratio must be 1.0, got {total:.10f}")


def _validate_split_strategy(strategy: str) -> None:
    """Raise when the split strategy is not one of SPLIT_STRATEGIES."""
    if strategy not in SPLIT_STRATEGIES:
        raise ValueError(
            f"Unknown split_strategy {strategy!r}. Expected one of {list(SPLIT_STRATEGIES)}"
        )


def _validate_temporal_block(block: str) -> None:
    """Raise when the temporal block key is not one of TEMPORAL_BLOCKS."""
    if block not in TEMPORAL_BLOCKS:
        raise ValueError(
            f"Unknown temporal_block {block!r}. Expected one of {list(TEMPORAL_BLOCKS)}"
        )
