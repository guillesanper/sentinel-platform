"""
Training CLI for the flow-based track.

The flow counterpart of :mod:`ml.training.train`. Ported from the CICIDS2017
TFA's ``main.py`` with ``argparse`` swapped for click, the global ``cfg``
singleton replaced by an explicit :class:`~ml.flow.config.FlowConfig`, and the
row-wise random split replaced by the temporal partition that
:mod:`ml.flow.loader` implements.

It stays a separate module rather than a flag on ``ml.training.train`` because
the two tracks share no label space. ``AttackClassifier._encode_labels`` maps an
unknown label to ``normal`` *silently*, so a single CLI able to feed CICIDS
labels to the event-track classifier would be able to produce a model that
predicts ``normal`` for everything without raising anything anywhere.

Two things about this file are load bearing:

* **Every import of :mod:`ml.evaluation.plots` sits inside an ``if`` branch.**
  ``api/Dockerfile`` copies ``ml/`` wholesale while installing only
  ``requirements.txt``, and matplotlib/seaborn live in ``requirements-dev.txt``.
  A module-level figure import here breaks the production image. A subprocess
  test guards it.
* **The feature matrix comes from
  :func:`~ml.flow.loader.get_feature_columns`, never from a hand-written column
  list.** ``capture_day`` and ``order_idx`` are provenance, not features; leaking
  either produces a model that detects the capture schedule, with spectacular
  metrics, no error, and no value.

Usage:
    python -m ml.training.train_cicids --data-dir api/csvs --pipeline xgb
    python -m ml.training.train_cicids --data-dir api/csvs --split random \\
        --sample-size 200000 --skip-tuning --output-dir /tmp/flow-random
"""

import json
import logging
import random
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, TypeAlias

import click
import numpy as np
import pandas as pd
import structlog
from sklearn.metrics import classification_report

from ml.evaluation.metrics import compare_models
from ml.flow.config import SPLIT_STRATEGIES, FlowConfig
from ml.flow.labels import BENIGN_LABEL, CLASS_ORDER
from ml.flow.loader import (
    DEFAULT_TEST_DAYS,
    LABEL_COL,
    clean_dataset,
    filter_minority_classes,
    get_feature_columns,
    load_cicids2017,
    split_dataset_by_strategy,
    stratified_sample,
)
from ml.flow.pipelines.cascade_xgb import CascadeXGBPipeline

if TYPE_CHECKING:
    from ml.flow.pipelines.cascade_mlp import CascadeMLPPipeline

logger = structlog.get_logger()

# The two concrete cascades, rather than BaseCascadePipeline: the base class
# carries the training and persistence contract, but `if_model`,
# `stage2_classifier` and `get_if_operating_point` — which the comparison and
# the figures both read — are defined on the subclasses.
#
# Spelled as a string because ``cascade_mlp`` is imported lazily, inside
# :func:`_make_pipeline`. It is the experimental branch of the track — never
# served, excluded from coverage — and importing it eagerly would both make a
# `--pipeline xgb` run pay for 800 lines of hand-written NumPy MLP and attach it
# as an attribute of ``ml.flow.pipelines``, which the package deliberately does
# not re-export.
Cascade: TypeAlias = "CascadeXGBPipeline | CascadeMLPPipeline"


# ============================================================================
# CLI contract
# ============================================================================

# Written at the root of --output-dir, next to the cascade sub-directories.
SUMMARY_FILE: Final[str] = "training_summary.json"

# Cascades selectable with --pipeline.
PIPELINE_CHOICES: Final[tuple[str, ...]] = ("xgb", "mlp", "both")

# Human-facing names, matching the defaults of ml.evaluation.plots.
CASCADE_DISPLAY_NAMES: Final[dict[str, str]] = {
    "xgb": "Cascade XGB",
    "mlp": "Cascade MLP",
}

# Option defaults are read off the dataclass rather than repeated as literals,
# so `--help` and FlowConfig cannot drift apart.
_CONFIG_DEFAULTS: Final[FlowConfig] = FlowConfig()


# ============================================================================
# Prepared data
# ============================================================================


@dataclass(frozen=True)
class _Dataset:
    """
    The three splits as model-ready arrays, plus the two name lists.

    ``feature_names`` is the column order of every matrix here and the contract
    persisted alongside the saved cascade; ``labels`` is the subset of
    :data:`~ml.flow.labels.CLASS_ORDER` that actually survived cleaning,
    minority filtering and sampling, in canonical order.
    """

    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    labels: list[str]


# ============================================================================
# Setup helpers
# ============================================================================


def _set_seeds(seed: int) -> None:
    """Seed the two global RNGs the training path draws from."""
    random.seed(seed)
    np.random.seed(seed)


def _build_config(
    *,
    data_dir: str,
    output_dir: str,
    sample_size: int,
    split: str,
    beta: float,
    seed: int,
) -> FlowConfig:
    """
    Turn the CLI options into a :class:`FlowConfig`.

    Args:
        data_dir: Directory holding the CICIDS2017 CSVs.
        output_dir: Root for every artifact this run writes.
        sample_size: Target stratified sample size; 0 uses the full dataset.
        split: Partitioning protocol, one of
            :data:`~ml.flow.config.SPLIT_STRATEGIES`.
        beta: F-beta weight for the Stage 1 threshold search.
        seed: Global random seed.

    Returns:
        A validated configuration — ``FlowConfig.__post_init__`` rejects an
        unknown split strategy and ratios that do not sum to 1.

    Note:
        ``test_days`` is deliberately not a config field: it belongs to the
        day-holdout split function, not to the track.
    """
    return FlowConfig(
        data_path=Path(data_dir),
        output_path=Path(output_dir),
        sample_size=sample_size,
        split_strategy=split,
        beta=beta,
        random_seed=seed,
    )


# ============================================================================
# Data preparation
# ============================================================================


def _prepare_dataset(config: FlowConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Load, clean, filter and sample the dataset.

    Args:
        config: Track configuration; supplies ``data_path``, the minority-class
            thresholds and ``sample_size``.

    Returns:
        ``(frame, stats)`` — the frame still carrying ``capture_day`` and
        ``order_idx``, and a JSON-serialisable record of what each stage
        removed.
    """
    raw = load_cicids2017(config.data_path)
    cleaned = clean_dataset(raw)
    filtered, removed = filter_minority_classes(cleaned, config)
    sampled = stratified_sample(filtered, config)

    stats: dict[str, Any] = {
        "n_rows_loaded": int(len(raw)),
        "n_rows_cleaned": int(len(cleaned)),
        "n_rows_after_minority_filter": int(len(filtered)),
        "n_rows_sampled": int(len(sampled)),
        "removed_classes": removed,
    }
    logger.info("flow_training_dataset_ready", **stats)
    return sampled, stats


def _build_matrices(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> _Dataset:
    """
    Project the three splits onto the feature space.

    Args:
        train_df: Training split, as returned by the partitioning functions.
        val_df: Validation split.
        test_df: Test split.

    Returns:
        The matrices plus the feature and class name lists.

    Note:
        The column list comes from :func:`~ml.flow.loader.get_feature_columns`,
        the single source of truth for the feature space. Building it here by
        hand is what would let ``capture_day`` or ``order_idx`` reach the model.
    """
    feature_cols = get_feature_columns(train_df)

    y_train = train_df[LABEL_COL].to_numpy()
    y_val = val_df[LABEL_COL].to_numpy()
    y_test = test_df[LABEL_COL].to_numpy()

    present = set(y_train) | set(y_val) | set(y_test)
    labels = [cls for cls in CLASS_ORDER if cls in present]

    logger.info(
        "flow_training_matrices_built",
        n_features=len(feature_cols),
        n_train=int(len(y_train)),
        n_val=int(len(y_val)),
        n_test=int(len(y_test)),
        labels=labels,
    )

    return _Dataset(
        X_train=train_df[feature_cols].to_numpy(dtype=np.float32),
        y_train=y_train,
        X_val=val_df[feature_cols].to_numpy(dtype=np.float32),
        y_val=y_val,
        X_test=test_df[feature_cols].to_numpy(dtype=np.float32),
        y_test=y_test,
        feature_names=feature_cols,
        labels=labels,
    )


# ============================================================================
# Cascade execution
# ============================================================================


def _make_pipeline(
    kind: str,
    config: FlowConfig,
    feature_names: list[str],
) -> tuple[Cascade, Path]:
    """
    Build the requested cascade and the directory it owns.

    Args:
        kind: ``"xgb"`` or ``"mlp"``.
        config: Track configuration.
        feature_names: Training-time column order, recorded now so that
            :meth:`~ml.flow.pipelines.base.BaseCascadePipeline.save` can persist
            the contract.

    Returns:
        ``(pipeline, output_dir)``.

    Raises:
        ValueError: ``kind`` is neither cascade.
    """
    if kind == "xgb":
        return (
            CascadeXGBPipeline(config, feature_names=feature_names),
            config.cascade_xgb_path,
        )
    if kind == "mlp":
        # Lazy on purpose — see the note on the Cascade alias.
        from ml.flow.pipelines.cascade_mlp import CascadeMLPPipeline

        return (
            CascadeMLPPipeline(config, feature_names=feature_names),
            config.cascade_mlp_path,
        )
    raise ValueError(f"Unknown pipeline {kind!r}. Expected one of {list(PIPELINE_CHOICES)}")


def _run_cascade(
    kind: str,
    dataset: _Dataset,
    config: FlowConfig,
    *,
    skip_tuning: bool,
    plots: bool,
) -> tuple[Cascade, dict[str, float], Path]:
    """
    Train, evaluate, save and optionally plot one cascade.

    The save happens before the figures on purpose: plotting is the only step
    here that can fail for reasons unrelated to the model, and a run that has
    already produced a servable artifact should not lose it to a broken figure.

    Args:
        kind: ``"xgb"`` or ``"mlp"``.
        dataset: Prepared matrices.
        config: Track configuration.
        skip_tuning: Skip the Stage 2 hyperparameter search.
        plots: Generate the per-cascade figures.

    Returns:
        ``(pipeline, test_metrics, output_dir)``.
    """
    started = time.perf_counter()
    pipeline, output_dir = _make_pipeline(kind, config, dataset.feature_names)
    display = CASCADE_DISPLAY_NAMES[kind]

    print("\n" + "=" * 60)
    print(f"{display} — training")
    print("=" * 60)

    logger.info(
        "flow_training_cascade_started",
        pipeline=pipeline.name,
        beta=config.beta,
        skip_tuning=skip_tuning,
    )
    pipeline.train(
        dataset.X_train,
        dataset.y_train,
        dataset.X_val,
        dataset.y_val,
        beta=config.beta,
        skip_tuning=skip_tuning,
        feature_names=dataset.feature_names,
    )

    metrics = pipeline.evaluate(dataset.X_test, dataset.y_test)
    pipeline.save(output_dir)

    y_pred = pipeline.predict(dataset.X_test)
    report = classification_report(dataset.y_test, y_pred, labels=dataset.labels, zero_division=0)
    print(f"\n--- {display}: classification report ---\n{report}")

    if plots:
        try:
            _plot_cascade(pipeline, dataset, y_pred, output_dir, display=display, beta=config.beta)
        except Exception as exc:  # figures must not sink a saved run
            logger.warning(
                "flow_training_cascade_plots_skipped",
                pipeline=pipeline.name,
                error=str(exc),
            )

    elapsed = time.perf_counter() - started
    print(
        f"\n{display} — accuracy {metrics['accuracy']:.4f} | "
        f"F1-macro {metrics['f1_macro']:.4f} | F1-weighted {metrics['f1_weighted']:.4f} "
        f"({elapsed:.1f}s)"
    )
    logger.info(
        "flow_training_cascade_finished",
        pipeline=pipeline.name,
        elapsed_seconds=round(elapsed, 1),
        **{key: round(float(value), 6) for key, value in metrics.items()},
    )
    return pipeline, metrics, output_dir


def _cascade_proba(
    pipeline: Cascade,
    X: np.ndarray,
    labels: list[str],
) -> np.ndarray:
    """
    Build a full ``(n, n_classes)`` probability matrix for a cascade.

    A cascade has no native ``predict_proba``: Stage 2 only ever sees the flows
    Stage 1 flagged. Flows Stage 1 called benign therefore get probability 1.0
    on BENIGN, and the rest get Stage 2's softmax projected onto the canonical
    class index. It is what makes AUC-ROC and the ROC/PR comparison figures
    possible.

    Args:
        pipeline: A trained cascade exposing ``if_model`` and
            ``stage2_classifier``.
        X: Feature matrix.
        labels: Canonical class order defining the column layout.

    Returns:
        Probability matrix of shape ``(len(X), len(labels))``.
    """
    label_to_idx = {label: i for i, label in enumerate(labels)}
    proba = np.zeros((X.shape[0], len(labels)), dtype=np.float32)

    benign_idx = label_to_idx.get(BENIGN_LABEL, 0)
    anomaly_mask = pipeline.if_model.predict_binary(X).astype(bool)
    proba[~anomaly_mask, benign_idx] = 1.0

    if anomaly_mask.sum() > 0:
        stage2 = pipeline.stage2_classifier
        stage2_proba = stage2.predict_proba(X[anomaly_mask])
        for column, cls in enumerate(stage2.classes_):
            idx = label_to_idx.get(str(cls))
            if idx is not None:
                proba[anomaly_mask, idx] = stage2_proba[:, column]

    return proba


def _run_comparison(
    xgb: Cascade,
    mlp: Cascade,
    dataset: _Dataset,
    config: FlowConfig,
    *,
    plots: bool,
) -> dict[str, Any]:
    """
    Score both cascades side by side on the test set.

    Args:
        xgb: Trained XGBoost cascade.
        mlp: Trained MLP cascade.
        dataset: Prepared matrices.
        config: Track configuration.
        plots: Generate the comparison figures.

    Returns:
        The dict from :func:`~ml.evaluation.metrics.compare_models`.

    Note:
        Both cascades share Stage 1 — the same detector with the same
        configuration — so their Stage 1 diagnostics are identical by
        construction. Do not report them as two independent results.
    """
    print("\n" + "=" * 60)
    print("Cross-cascade comparison")
    print("=" * 60)

    y_score_a = _cascade_proba(xgb, dataset.X_test, dataset.labels)
    y_score_b = _cascade_proba(mlp, dataset.X_test, dataset.labels)

    comparison = compare_models(
        xgb,
        mlp,
        dataset.X_test,
        dataset.y_test,
        seed=config.random_seed,
        labels=dataset.labels,
        y_score_a=y_score_a,
        y_score_b=y_score_b,
    )

    if plots:
        try:
            _plot_comparison(
                comparison,
                dataset,
                y_score_a,
                y_score_b,
                config.comparison_path,
            )
        except Exception as exc:  # figures are not the deliverable
            logger.warning("flow_training_comparison_plots_skipped", error=str(exc))

    return comparison


# ============================================================================
# Figures — every matplotlib import in this file lives below this line
# ============================================================================


def _run_eda(frame: pd.DataFrame, config: FlowConfig) -> None:
    """
    Generate the exploratory figures for the sampled dataset.

    Args:
        frame: The sampled frame, provenance columns included.
        config: Track configuration; the figures land in ``config.eda_path``.
    """
    from ml.evaluation.plots import run_eda

    feature_cols = get_feature_columns(frame)
    run_eda(frame[feature_cols], frame[LABEL_COL], config.eda_path)


def _plot_split_distribution(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: FlowConfig,
) -> None:
    """
    Draw the rows-per-``(capture_day, class)``-block figure.

    The only place where a class missing from a split, or a block too small to
    yield a meaningful test slice, is visible at a glance. Neither condition
    raises, and both quietly invalidate the per-class metrics.

    Args:
        train_df: Training split.
        val_df: Validation split.
        test_df: Test split.
        config: Track configuration; the figure lands in ``config.output_path``.
    """
    from ml.evaluation.plots import plot_split_distribution

    plot_split_distribution(train_df, val_df, test_df, config.output_path)


def _plot_cascade(
    pipeline: Cascade,
    dataset: _Dataset,
    y_pred: np.ndarray,
    output_dir: Path,
    *,
    display: str,
    beta: float,
) -> None:
    """
    Draw the per-cascade figures next to the saved artifacts.

    Args:
        pipeline: The trained cascade.
        dataset: Prepared matrices.
        y_pred: Test-set predictions, already computed by the caller.
        output_dir: The cascade's own directory.
        display: Human-facing cascade name, used in the figure titles.
        beta: F-beta weight the Stage 1 threshold was chosen with.
    """
    from ml.evaluation.plots import (
        plot_confusion_matrix,
        plot_feature_importance,
        plot_pr_tradeoff,
        plot_threshold_curve,
        plot_training_curves,
    )

    plot_confusion_matrix(
        dataset.y_test,
        y_pred,
        dataset.labels,
        output_dir,
        title=f"{display} — Confusion Matrix",
    )

    # Narrowed on the XGBoost cascade alone: the else branch is the MLP one, and
    # naming it would mean importing it just to run an isinstance check.
    if isinstance(pipeline, CascadeXGBPipeline):
        plot_feature_importance(pipeline.get_feature_importance(), output_dir)
    else:
        plot_training_curves(pipeline.get_training_history(), output_dir)

    detector = pipeline.if_model
    threshold = detector.threshold
    if threshold is None:
        logger.warning("flow_training_threshold_curves_skipped", pipeline=pipeline.name)
        return

    scores = detector.decision_scores(dataset.X_val)
    y_binary_val = (dataset.y_val != BENIGN_LABEL).astype(int)
    plot_threshold_curve(scores, y_binary_val, threshold, output_dir, beta=beta)
    plot_pr_tradeoff(scores, y_binary_val, threshold, output_dir, beta=beta)


def _plot_comparison(
    comparison: dict[str, Any],
    dataset: _Dataset,
    y_score_a: np.ndarray,
    y_score_b: np.ndarray,
    output_dir: Path,
) -> None:
    """
    Draw the four cross-cascade comparison figures.

    Args:
        comparison: Result of :func:`~ml.evaluation.metrics.compare_models`.
        dataset: Prepared matrices.
        y_score_a: Probability matrix of the XGBoost cascade.
        y_score_b: Probability matrix of the MLP cascade.
        output_dir: Directory for the four PNGs.
    """
    from ml.evaluation.plots import (
        plot_f1_per_class,
        plot_metrics_bar,
        plot_pr_curves,
        plot_roc_curves,
    )

    plot_metrics_bar(comparison, output_dir)
    plot_f1_per_class(comparison["model_a"], comparison["model_b"], output_dir)
    plot_roc_curves(dataset.y_test, y_score_a, y_score_b, dataset.labels, output_dir)
    plot_pr_curves(dataset.y_test, y_score_a, y_score_b, dataset.labels, output_dir)


# ============================================================================
# Run record
# ============================================================================


def _operating_point(pipeline: Cascade) -> dict[str, Any] | None:
    """
    Stage 1 threshold diagnostics, or None when no search ever ran.

    Returns None rather than zeros for the same reason the event track does: a
    fabricated operating point in the training record is worse than an absent
    one.
    """
    try:
        return dict(pipeline.get_if_operating_point())
    except RuntimeError:
        return None


def _build_summary(
    *,
    config: FlowConfig,
    data_stats: dict[str, Any],
    dataset: _Dataset,
    test_days: tuple[str, ...],
    skip_tuning: bool,
    mlflow_requested: bool,
    trained: dict[str, Cascade],
    metrics: dict[str, dict[str, float]],
    model_paths: dict[str, str],
    comparison: dict[str, Any] | None,
    elapsed: float,
) -> dict[str, Any]:
    """
    Assemble the JSON record of the run.

    The ``split`` block is the point of the file: an F1-macro from this track is
    uninterpretable without knowing which partitioning protocol produced it, and
    the temporal and random numbers are expected to differ.
    """
    return {
        "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "track": "flow",
        "data": {
            "data_dir": str(config.data_path),
            **data_stats,
            "train_size": int(len(dataset.y_train)),
            "val_size": int(len(dataset.y_val)),
            "test_size": int(len(dataset.y_test)),
            "n_features": len(dataset.feature_names),
            "classes": dataset.labels,
        },
        "split": {
            "strategy": config.split_strategy,
            "train_ratio": config.train_ratio,
            "val_ratio": config.val_ratio,
            "test_ratio": config.test_ratio,
            # Only meaningful for day-holdout; null everywhere else rather than
            # a value that reads as if it had been applied.
            "test_days": (list(test_days) if config.split_strategy == "day-holdout" else None),
        },
        "parameters": {
            "sample_size": config.sample_size,
            "seed": config.random_seed,
            "beta": config.beta,
            "skip_tuning": skip_tuning,
            "pipelines": sorted(metrics),
        },
        "metrics": metrics,
        "stage1_operating_point": {
            pipeline.name: _operating_point(pipeline) for pipeline in trained.values()
        },
        "comparison": comparison["comparison"] if comparison is not None else None,
        "model_paths": model_paths,
        # MLflow tracking is out of scope for the flow track. Recorded so a run
        # invoked with --mlflow cannot be mistaken later for a tracked one.
        "mlflow": {"requested": mlflow_requested, "logged": False},
        "elapsed_seconds": round(elapsed, 1),
    }


# ============================================================================
# Entry point
# ============================================================================


@click.command()
@click.option(
    "--data-dir",
    type=click.Path(exists=False),
    default=str(_CONFIG_DEFAULTS.data_path),
    show_default=True,
    help="Directory holding the CICIDS2017 CSV files.",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    default=str(_CONFIG_DEFAULTS.output_path),
    show_default=True,
    help="Root directory for the cascades, figures and the run summary.",
)
@click.option(
    "--sample-size",
    type=int,
    default=_CONFIG_DEFAULTS.sample_size,
    show_default=True,
    help="Target stratified sample size. 0 uses the full dataset.",
)
@click.option(
    "--pipeline",
    type=click.Choice(list(PIPELINE_CHOICES)),
    default="xgb",
    show_default=True,
    help="Cascade to train. 'both' also runs the cross-cascade comparison.",
)
@click.option(
    "--split",
    type=click.Choice(list(SPLIT_STRATEGIES)),
    default=_CONFIG_DEFAULTS.split_strategy,
    show_default=True,
    help=(
        "Partitioning protocol. 'temporal' cuts each (capture day, class) block "
        "chronologically; 'random' is the TFA's leaky row-wise split, kept to "
        "quantify the leakage; 'day-holdout' reserves whole capture days."
    ),
)
@click.option(
    "--test-days",
    multiple=True,
    default=DEFAULT_TEST_DAYS,
    show_default=True,
    help="Capture days held out by --split day-holdout. Repeatable; ignored otherwise.",
)
@click.option(
    "--beta",
    type=float,
    default=_CONFIG_DEFAULTS.beta,
    show_default=True,
    help="F-beta weight for the Stage 1 threshold. >1 penalises missed attacks.",
)
@click.option(
    "--skip-tuning/--tune",
    default=False,
    show_default=True,
    help="Skip the Stage 2 RandomizedSearchCV pass and use the config defaults.",
)
@click.option(
    "--plots/--no-plots",
    default=False,
    show_default=True,
    help="Generate figures. Requires matplotlib and seaborn (requirements-dev.txt).",
)
@click.option(
    "--eda/--no-eda",
    default=False,
    show_default=True,
    help="Generate the exploratory figures before training. Same dependency.",
)
@click.option(
    "--seed",
    type=int,
    default=_CONFIG_DEFAULTS.random_seed,
    show_default=True,
    help="Global random seed.",
)
@click.option(
    "--mlflow/--no-mlflow",
    default=False,
    show_default=True,
    help="Accepted for symmetry with the event track. NO-OP: see --help notes.",
)
def main(
    data_dir: str,
    output_dir: str,
    sample_size: int,
    pipeline: str,
    split: str,
    test_days: tuple[str, ...],
    beta: float,
    skip_tuning: bool,
    plots: bool,
    eda: bool,
    seed: int,
    mlflow: bool,
) -> None:
    """
    Train the CICIDS2017 flow cascades.

    Stage 1 is an Isolation Forest fitted on benign flows with an F-beta tuned
    threshold; Stage 2 names the attack family for whatever Stage 1 flags. The
    artifacts land in <output-dir>/cascade_xgb (and /cascade_mlp), in the layout
    ml.flow.serving.CascadeInferenceService expects, feature-name contract
    included.

    --mlflow is a documented no-op. MLflow tracking is out of scope for this
    track; the flag exists so invocations stay symmetric with
    `python -m ml.training.train`, and the run summary records that nothing was
    tracked rather than leaving it ambiguous.

    Examples:

        # Default protocol, over the real CSVs
        python -m ml.training.train_cicids --data-dir api/csvs --pipeline xgb

        # Quantify the leakage: same run, the TFA's random split
        python -m ml.training.train_cicids --data-dir api/csvs --split random \\
            --sample-size 200000 --skip-tuning --output-dir /tmp/flow-random

        # Both cascades, with figures
        python -m ml.training.train_cicids --data-dir api/csvs --pipeline both --plots
    """
    structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.INFO))

    config = _build_config(
        data_dir=data_dir,
        output_dir=output_dir,
        sample_size=sample_size,
        split=split,
        beta=beta,
        seed=seed,
    )
    config.create_output_dirs()
    _set_seeds(config.random_seed)

    started = time.perf_counter()
    logger.info(
        "flow_training_started",
        data_dir=str(config.data_path),
        output_dir=str(config.output_path),
        sample_size=config.sample_size,
        pipeline=pipeline,
        split=config.split_strategy,
        beta=config.beta,
        skip_tuning=skip_tuning,
        seed=config.random_seed,
    )

    if mlflow:
        logger.warning(
            "flow_training_mlflow_not_implemented",
            hint="MLflow tracking is out of scope for the flow track; nothing was logged",
        )
        print("[warning] --mlflow is a no-op for the flow track: nothing will be tracked.")

    # --- Data ---------------------------------------------------------------
    frame, data_stats = _prepare_dataset(config)

    if eda:
        _run_eda(frame, config)

    train_df, val_df, test_df = split_dataset_by_strategy(frame, config, test_days=test_days)
    dataset = _build_matrices(train_df, val_df, test_df)

    if plots:
        try:
            _plot_split_distribution(train_df, val_df, test_df, config)
        except Exception as exc:  # the audit figure is not the deliverable
            logger.warning("flow_training_split_figure_skipped", error=str(exc))

    # --- Cascades -----------------------------------------------------------
    kinds = ("xgb", "mlp") if pipeline == "both" else (pipeline,)

    trained: dict[str, Cascade] = {}
    metrics: dict[str, dict[str, float]] = {}
    model_paths: dict[str, str] = {}

    for kind in kinds:
        cascade, cascade_metrics, path = _run_cascade(
            kind,
            dataset,
            config,
            skip_tuning=skip_tuning,
            plots=plots,
        )
        trained[kind] = cascade
        metrics[cascade.name] = cascade_metrics
        model_paths[cascade.name] = str(path)

    comparison: dict[str, Any] | None = None
    if "xgb" in trained and "mlp" in trained:
        comparison = _run_comparison(
            trained["xgb"],
            trained["mlp"],
            dataset,
            config,
            plots=plots,
        )

    # --- Run record ---------------------------------------------------------
    elapsed = time.perf_counter() - started
    summary = _build_summary(
        config=config,
        data_stats=data_stats,
        dataset=dataset,
        test_days=tuple(test_days),
        skip_tuning=skip_tuning,
        mlflow_requested=mlflow,
        trained=trained,
        metrics=metrics,
        model_paths=model_paths,
        comparison=comparison,
        elapsed=elapsed,
    )

    summary_path = config.output_path / SUMMARY_FILE
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print("\n" + "=" * 60)
    print(f"Flow training complete in {elapsed:.1f}s — split: {config.split_strategy}")
    for name, cascade_metrics in metrics.items():
        print(
            f"  {name:14s} accuracy {cascade_metrics['accuracy']:.4f} | "
            f"F1-macro {cascade_metrics['f1_macro']:.4f} | "
            f"F1-weighted {cascade_metrics['f1_weighted']:.4f}"
        )
    print(f"Summary: {summary_path}")
    print("=" * 60)

    logger.info(
        "flow_training_finished",
        elapsed_seconds=round(elapsed, 1),
        split=config.split_strategy,
        summary=str(summary_path),
    )


if __name__ == "__main__":
    main()
