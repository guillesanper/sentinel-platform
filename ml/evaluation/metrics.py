"""
Model-agnostic classification metrics.

Pure numpy + scikit-learn: usable by both the event-based track
(``ml/models/``) and the flow-based track (``ml/flow/``). Nothing here imports
matplotlib — figures live in :mod:`ml.evaluation.plots`.

Ported from the CICIDS2017 TFA (``evaluation/metrics.py``); the pipeline-specific
naming (``compare_pipelines`` / ``pipeline_a`` / ``pipeline_b``) was generalised
to ``compare_models`` / ``model_a`` / ``model_b`` because both tracks use it.
"""

from typing import Any

import numpy as np
import structlog
from sklearn.metrics import accuracy_score, f1_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelBinarizer

logger = structlog.get_logger()

# Scalar metrics compared side by side by compare_models().
SCALAR_METRICS: tuple[str, ...] = ("accuracy", "f1_macro", "f1_weighted")

# Deltas below this are reported as a tie rather than a winner.
_TIE_EPSILON: float = 1e-6


# ============================================================================
# Public API
# ============================================================================


def best_fbeta_threshold(
    scores: np.ndarray,
    y_true: np.ndarray,
    beta: float = 1.0,
    n_candidates: int | None = None,
) -> tuple[float, float]:
    """
    Find the anomaly-score threshold that maximises F-beta.

    Shared by both tracks' detectors, which agree on the decision rule —
    **lower score means more anomalous**::

        score(x) <  t  →  positive (anomaly, 1)
        score(x) >= t  →  negative (normal, 0)

    Every distinct operating point of the validation set is evaluated in a
    single vectorised pass, so the result is the **exact** maximiser rather than
    the best of a fixed grid. Both detectors previously scanned 300 thresholds
    spread uniformly over ``[min(score), max(score)]``, which is the wrong
    spacing for Isolation Forest — its scores bunch tightly around the boundary,
    so a uniform grid spends most of its resolution on empty range and can miss
    the optimum entirely. It also cost one full ``fbeta_score`` pass per
    candidate.

    Args:
        scores: Decision-function scores, shape ``(n,)``.
        y_true: Binary ground truth, shape ``(n,)``; 1 = anomaly.
        beta: Recall weight relative to precision. ``beta > 1`` favours recall.
        n_candidates: Optional cap on the number of operating points evaluated.
            None — the default — evaluates all of them. When set, candidates are
            thinned evenly **along the curve**, which is quantile spacing in the
            score distribution, not the uniform range spacing it replaces.

    Returns:
        ``(threshold, fbeta)``. When ``y_true`` carries no positive, no
        threshold can be scored: the median score is returned with an F-beta of
        0.0 and a warning is logged.
    """
    scores = np.asarray(scores, dtype=float)
    y_true = np.asarray(y_true)

    if scores.size == 0 or int(y_true.sum()) == 0:
        logger.warning(
            "fbeta_threshold_undefined",
            n_samples=int(scores.size),
            n_positive=int(y_true.sum()) if y_true.size else 0,
            hint="no positive class in y_true; returning the median score unoptimised",
        )
        return (float(np.median(scores)) if scores.size else 0.0), 0.0

    # precision_recall_curve thresholds on `>=`, so it is fed the negated
    # scores; a candidate `thr` there means `-score >= thr`, i.e. `score <= -thr`.
    precision, recall, neg_thresholds = precision_recall_curve(y_true, -scores)

    # The curve's trailing point is (precision=1, recall=0) and carries no
    # threshold: it is the degenerate "flag nothing" operating point, which has
    # undefined precision in practice and must stay out of the search.
    precision, recall = precision[:-1], recall[:-1]

    if n_candidates is not None and 0 < n_candidates < len(neg_thresholds):
        keep = np.unique(np.linspace(0, len(neg_thresholds) - 1, n_candidates).round().astype(int))
        precision, recall, neg_thresholds = precision[keep], recall[keep], neg_thresholds[keep]

    beta_sq = beta * beta
    denominator = beta_sq * precision + recall
    fbeta = np.divide(
        (1.0 + beta_sq) * precision * recall,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )

    best = int(fbeta.argmax())

    # `score <= -thr` is what the curve selected; the detectors apply a strict
    # `<`. Nudging to the next representable float makes the two identical
    # instead of differing on exact ties at the boundary.
    threshold = float(np.nextafter(-neg_thresholds[best], np.inf))
    return threshold, float(fbeta[best])


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray | None = None,
    labels: list[str] | None = None,
) -> dict[str, Any]:
    """
    Compute a comprehensive set of classification metrics.

    Args:
        y_true: Ground-truth labels, shape (n,).
        y_pred: Predicted labels, shape (n,).
        y_score: Class probability matrix (softmax output), shape
            (n, n_classes). Required for AUC-ROC; ignored when None.
        labels: Canonical class order. Defaults to the sorted union of
            y_true and y_pred.

    Returns:
        Dict with keys:
            'accuracy'          — overall accuracy (float)
            'f1_macro'          — macro-averaged F1 (float)
            'f1_weighted'       — weighted-averaged F1 (float)
            'f1_per_class'      — per-class F1 (dict[str, float])
            'auc_roc_macro'     — macro OvR AUC-ROC (only if y_score given)
            'auc_roc_per_class' — per-class OvR AUC-ROC (same condition)
    """
    if labels is None:
        labels = sorted(set(np.unique(y_true)).union(set(np.unique(y_pred))))

    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(
            f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
        ),
        "f1_weighted": float(
            f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)
        ),
        "f1_per_class": {
            str(cls): float(
                f1_score(y_true, y_pred, labels=[cls], average="micro", zero_division=0)
            )
            for cls in labels
        },
    }

    if y_score is not None:
        metrics.update(_compute_auc_roc(y_true, y_score))

    return metrics


def compare_models(
    model_a: Any,
    model_b: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    n_cv_splits: int = 0,
    seed: int = 42,
    labels: list[str] | None = None,
    y_score_a: np.ndarray | None = None,
    y_score_b: np.ndarray | None = None,
) -> dict[str, Any]:
    """
    Quantitatively compare two trained models on held-out test data.

    When ``n_cv_splits`` > 1, StratifiedKFold partitioning of (X_test, y_test)
    estimates metric variance without retraining either model.

    Args:
        model_a: Any trained object exposing ``predict(X) -> np.ndarray``.
        model_b: Same, for the second model.
        X_test: Test feature matrix, shape (n_test, n_features).
        y_test: Ground-truth labels, shape (n_test,).
        n_cv_splits: Number of StratifiedKFold folds for variance estimation.
            0 or 1 disables cross-validation.
        seed: Random state passed to StratifiedKFold.
        labels: Canonical class order for metric computation.
        y_score_a: Probability matrix for model A, shape (n_test, n_classes).
            When provided, AUC-ROC metrics are included in the model_a results.
        y_score_b: Same, for model B.

    Returns:
        Dict with keys:
            'model_a'    — metrics dict from :func:`compute_metrics`
            'model_b'    — same for model B
            'comparison' — per-scalar-metric winner and delta, plus per-class F1
            'cv_model_a' — mean/std per metric across folds (only if
                           n_cv_splits > 1)
            'cv_model_b' — same for model B
    """
    logger.info("model_comparison_started", n_test_samples=len(y_test))

    metrics_a = _eval_model(model_a, X_test, y_test, labels, y_score=y_score_a)
    metrics_b = _eval_model(model_b, X_test, y_test, labels, y_score=y_score_b)

    result: dict[str, Any] = {
        "model_a": metrics_a,
        "model_b": metrics_b,
        "comparison": _build_comparison(metrics_a, metrics_b),
    }

    if n_cv_splits > 1:
        logger.info("cross_validation_started", n_splits=n_cv_splits)
        result["cv_model_a"] = _cv_metric_stats(model_a, X_test, y_test, n_cv_splits, seed, labels)
        result["cv_model_b"] = _cv_metric_stats(model_b, X_test, y_test, n_cv_splits, seed, labels)

    _log_comparison(result["comparison"])
    return result


# ============================================================================
# Private helpers
# ============================================================================


def _eval_model(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    labels: list[str] | None,
    y_score: np.ndarray | None = None,
) -> dict[str, Any]:
    """Predict with the model and compute metrics."""
    y_pred = model.predict(X)
    return compute_metrics(y, y_pred, y_score=y_score, labels=labels)


def _compute_auc_roc(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, Any]:
    """Compute One-vs-Rest AUC-ROC per class plus the macro average."""
    lb = LabelBinarizer()
    y_bin = lb.fit_transform(y_true)

    # LabelBinarizer returns (n, 1) for binary problems; expand to (n, 2).
    if y_bin.shape[1] == 1:
        y_bin = np.hstack([1 - y_bin, y_bin])

    auc_per_class: dict[str, float] = {}
    for i, cls in enumerate(lb.classes_):
        col = i if i < y_score.shape[1] else y_score.shape[1] - 1
        try:
            auc_per_class[str(cls)] = float(roc_auc_score(y_bin[:, i], y_score[:, col]))
        except ValueError as exc:
            logger.debug("auc_roc_skipped", label=str(cls), error=str(exc))
            auc_per_class[str(cls)] = float("nan")

    valid = [v for v in auc_per_class.values() if not np.isnan(v)]
    macro = float(np.mean(valid)) if valid else float("nan")

    return {
        "auc_roc_per_class": auc_per_class,
        "auc_roc_macro": macro,
    }


def _cv_metric_stats(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int,
    seed: int,
    labels: list[str] | None,
) -> dict[str, dict[str, float]]:
    """
    Evaluate an already-trained model across StratifiedKFold test folds.

    Partitions (X, y) into ``n_splits`` folds and runs prediction on each fold
    independently. No retraining occurs; this yields variance estimates for the
    model's inference performance on the provided dataset.

    Args:
        model: Trained model exposing ``predict(X)``.
        X: Feature matrix to partition (typically the test set).
        y: Matching labels.
        n_splits: Number of folds.
        seed: StratifiedKFold random state.
        labels: Canonical class order for metric computation.

    Returns:
        Dict mapping metric name → {'mean': float, 'std': float}.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_metrics: dict[str, list[float]] = {key: [] for key in SCALAR_METRICS}

    for fold_idx, (_, fold_rows) in enumerate(skf.split(X, y)):
        m = _eval_model(model, X[fold_rows], y[fold_rows], labels)
        for key in fold_metrics:
            fold_metrics[key].append(float(m[key]))
        logger.debug(
            "cv_fold_evaluated",
            fold=fold_idx + 1,
            n_splits=n_splits,
            f1_macro=round(m["f1_macro"], 4),
            accuracy=round(m["accuracy"], 4),
        )

    return {
        key: {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0),
        }
        for key, vals in fold_metrics.items()
    }


def _build_comparison(
    metrics_a: dict[str, Any],
    metrics_b: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Build a per-metric winner/delta comparison table."""
    comparison: dict[str, dict[str, Any]] = {}

    for key in SCALAR_METRICS:
        va = metrics_a.get(key)
        vb = metrics_b.get(key)
        if va is None or vb is None:
            continue
        delta = float(vb) - float(va)
        if delta > _TIE_EPSILON:
            winner = "model_b"
        elif delta < -_TIE_EPSILON:
            winner = "model_a"
        else:
            winner = "tie"
        comparison[key] = {
            "model_a": float(va),
            "model_b": float(vb),
            "delta": delta,  # positive → B wins
            "winner": winner,
        }

    f1_per_class_a = metrics_a.get("f1_per_class", {})
    f1_per_class_b = metrics_b.get("f1_per_class", {})
    comparison["f1_per_class"] = {
        cls: {
            "model_a": f1_per_class_a.get(cls, 0.0),
            "model_b": f1_per_class_b.get(cls, 0.0),
            "delta": f1_per_class_b.get(cls, 0.0) - f1_per_class_a.get(cls, 0.0),
        }
        for cls in set(f1_per_class_a) | set(f1_per_class_b)
    }

    return comparison


def _log_comparison(comparison: dict[str, dict[str, Any]]) -> None:
    """Emit one structured event per compared metric."""
    for metric, row in comparison.items():
        if metric == "f1_per_class":
            for cls, cls_row in sorted(row.items()):
                logger.info(
                    "model_comparison_f1_per_class",
                    label=cls,
                    model_a=round(cls_row["model_a"], 4),
                    model_b=round(cls_row["model_b"], 4),
                    delta=round(cls_row["delta"], 4),
                )
        else:
            logger.info(
                "model_comparison_metric",
                metric=metric,
                model_a=round(row["model_a"], 4),
                model_b=round(row["model_b"], 4),
                delta=round(row["delta"], 4),
                winner=row["winner"],
            )
